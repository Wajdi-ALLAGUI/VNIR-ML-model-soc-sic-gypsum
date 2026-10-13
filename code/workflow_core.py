"""Core modelling utilities for the revised VNIR soil-spectroscopy workflow.

The only analytical input is the raw spectral CSV. Spectral scenarios,
cross-validation partitions, model fitting, response scaling, pooled metrics,
and bootstrap intervals are all generated from that file at runtime.
"""

from __future__ import annotations

import os
import hashlib
import json
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from pipeline_preprocessing import (
    UpperHullContinuumRemoval,
    common_range_columns,
    wavelength_from_column,
)


RANDOM_STATE = 42
RF_TREES = 260
SG_WINDOW_LENGTH = 11
SG_POLYORDER = 3
SG_DERIVATIVE = 1
BOOTSTRAP_ITERATIONS = 1000

_STATELESS_CR_CACHE: dict[int, np.ndarray] = {}

TARGETS = ("SOC", "SIC", "Gypsum")
TARGET_COLUMNS = {"SOC": "SOC %", "SIC": "SIC %", "Gypsum": "Gypsum %"}
SCENARIOS = ("RAW", "RAW_SG_MSC", "CR")
VALIDATIONS = ("GroupKFold_Plot", "GroupKFold_Oasis", "KFold", "LOOCV")
METRICS = ("R2", "RMSE", "MAE", "Bias", "CCC", "RPD")

MANAGEMENT_LABELS = {1: "Modern", 2: "Traditional"}
TOPOGRAPHY_LABELS = {1: "Upslope", 2: "Midslope", 3: "Downslope"}
STUDY_AREA_BY_OASIS = {
    "Bouhlel": "Deguache",
    "Degache": "Deguache",
    "Lahwar": "Tozeur",
    "Machtla": "Nefta",
    "Nefta": "Nefta",
    "Tozeur": "Tozeur",
}


@dataclass(frozen=True)
class AnalysisData:
    raw: pd.DataFrame
    working: pd.DataFrame
    spectral_columns: tuple[str, ...]
    wavelengths: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    sample_id: np.ndarray
    plot: np.ndarray
    oasis: np.ndarray
    study_area: np.ndarray
    oasis_type: np.ndarray
    management: np.ndarray
    slope_category: np.ndarray
    topography: np.ndarray
    depth: np.ndarray


@dataclass(frozen=True)
class EvaluationConfig:
    scenario: str
    path: str
    stratum: str
    validation: str
    model_family: str
    mode: str

    @property
    def model_flow(self) -> str:
        return f"{self.model_family}_{self.mode}"

    @property
    def key(self) -> str:
        return "__".join(
            [
                self.scenario,
                self.path,
                self.stratum,
                self.validation,
                self.model_family,
                self.mode,
            ]
        )


class SavitzkyGolayMSC(BaseEstimator, TransformerMixin):
    """Savitzky-Golay first derivative followed by calibration-fitted MSC."""

    def __init__(
        self,
        window_length: int = SG_WINDOW_LENGTH,
        polyorder: int = SG_POLYORDER,
        deriv: int = SG_DERIVATIVE,
    ) -> None:
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def _sg(self, X: np.ndarray) -> np.ndarray:
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
            mode="interp",
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SavitzkyGolayMSC":
        matrix = check_array(X, dtype=float, ensure_all_finite=True)
        if self.window_length % 2 != 1:
            raise ValueError("Savitzky-Golay window_length must be odd.")
        if self.window_length <= self.polyorder:
            raise ValueError("Savitzky-Golay window_length must exceed polyorder.")
        if matrix.shape[1] < self.window_length:
            raise ValueError("Savitzky-Golay window exceeds the number of spectral bands.")
        transformed = self._sg(matrix)
        self.reference_ = np.mean(transformed, axis=0)
        self.n_features_in_ = matrix.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["reference_", "n_features_in_"])
        matrix = check_array(X, dtype=float, ensure_all_finite=True)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("MSC input has a different number of bands than its calibration data.")
        transformed = self._sg(matrix)
        design = np.column_stack([self.reference_, np.ones_like(self.reference_)])
        corrected = np.empty_like(transformed)
        for row_index, spectrum in enumerate(transformed):
            slope, intercept = np.linalg.lstsq(design, spectrum, rcond=None)[0]
            if not np.isfinite(slope) or abs(slope) <= 1e-12:
                raise ValueError(f"MSC slope is undefined for spectrum row {row_index}.")
            corrected[row_index] = (spectrum - intercept) / slope
        return corrected


def load_analysis_data(raw_path: Path) -> AnalysisData:
    raw = pd.read_csv(raw_path)
    required = {
        "identifiant",
        "sites",
        "OasisType",
        "Slope_Category",
        *TARGET_COLUMNS.values(),
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw input is missing required columns: {missing}")
    if len(raw) != 216:
        raise ValueError(f"Expected 216 samples, found {len(raw)}.")
    if raw["identifiant"].duplicated().any():
        raise ValueError("Sample identifiers must be unique.")

    plot_series = raw["sites"].astype(str)
    if not plot_series.eq(plot_series.str.strip()).all():
        raise ValueError("Plot values contain leading or trailing whitespace; values were not normalized.")
    plot = plot_series.to_numpy()
    oasis = plot_series.str.replace(r"\d+$", "", regex=True).to_numpy()
    if np.any(oasis == plot):
        bad = sorted(set(plot[oasis == plot]))
        raise ValueError(f"Cannot derive an oasis stem from plot values: {bad}")
    study_area = pd.Series(oasis).map(STUDY_AREA_BY_OASIS)
    if study_area.isna().any():
        bad = sorted(pd.Series(oasis)[study_area.isna()].unique())
        raise ValueError(f"No approved Study_Area mapping for oasis values: {bad}")

    oasis_type = pd.to_numeric(raw["OasisType"], errors="raise").astype(int).to_numpy()
    slope_category = (
        pd.to_numeric(raw["Slope_Category"], errors="raise").astype(int).to_numpy()
    )
    management = pd.Series(oasis_type).map(MANAGEMENT_LABELS)
    topography = pd.Series(slope_category).map(TOPOGRAPHY_LABELS)
    if management.isna().any() or topography.isna().any():
        raise ValueError("Unexpected OasisType or Slope_Category code in raw input.")

    depth_columns = [column for column in raw.columns if str(column).strip().lower() == "depth"]
    if len(depth_columns) != 1:
        raise ValueError(f"Expected one depth column, found {depth_columns}.")
    depth = pd.to_numeric(raw[depth_columns[0]], errors="raise").to_numpy(dtype=float)

    spectral_cols = common_range_columns(raw)
    wavelengths = np.asarray(
        [float(wavelength_from_column(column)) for column in spectral_cols], dtype=float
    )
    X = raw[spectral_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    Y = np.column_stack(
        [
            pd.to_numeric(raw[TARGET_COLUMNS[target]], errors="raise").to_numpy(dtype=float)
            for target in TARGETS
        ]
    )
    if not np.isfinite(Y).all():
        raise ValueError("Target variables contain missing or non-finite values.")

    # Preserve every raw field exactly and add only the approved Study_Area column.
    working = raw.copy()
    working["Study_Area"] = study_area.to_numpy()

    data = AnalysisData(
        raw=raw,
        working=working,
        spectral_columns=tuple(spectral_cols),
        wavelengths=wavelengths,
        X=X,
        Y=Y,
        sample_id=raw["identifiant"].astype(str).to_numpy(),
        plot=plot,
        oasis=oasis,
        study_area=study_area.astype(str).to_numpy(),
        oasis_type=oasis_type,
        management=management.astype(str).to_numpy(),
        slope_category=slope_category,
        topography=topography.astype(str).to_numpy(),
        depth=depth,
    )
    validate_sampling_hierarchy(data)
    return data


def validate_sampling_hierarchy(data: AnalysisData) -> None:
    mapping = pd.DataFrame(
        {
            "Plot": data.plot,
            "Oasis": data.oasis,
            "Study_Area": data.study_area,
            "OasisType": data.oasis_type,
            "Slope_Category": data.slope_category,
        }
    ).drop_duplicates()
    if mapping["Plot"].nunique() != 18 or mapping["Oasis"].nunique() != 6:
        raise ValueError("The approved hierarchy must contain 18 plots and 6 oases.")
    if set(mapping["Study_Area"]) != {"Tozeur", "Deguache", "Nefta"}:
        raise ValueError("The approved hierarchy must contain the three Study_Area values.")
    for area, group in mapping.groupby("Study_Area"):
        by_oasis = group.groupby("Oasis")["OasisType"].first()
        if len(by_oasis) != 2 or set(by_oasis.to_numpy()) != {1, 2}:
            raise ValueError(
                f"Study_Area {area} does not contain one OasisType 1 and one OasisType 2 oasis."
            )


def approved_site_mapping(data: AnalysisData) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "Plot": data.plot,
            "Oasis": data.oasis,
            "Study_Area": data.study_area,
            "OasisType": data.oasis_type,
            "Management": data.management,
            "Slope_Category": data.slope_category,
            "Topography": data.topography,
        }
    )
    counts = rows.groupby("Plot", sort=True).size().rename("N_Samples")
    mapping = rows.drop_duplicates().set_index("Plot").join(counts).reset_index()
    return mapping.sort_values("Plot").reset_index(drop=True)


def make_master_fold_assignments(data: AnalysisData) -> pd.DataFrame:
    n_samples = len(data.sample_id)
    assignment = pd.DataFrame(
        {
            "Sample_ID": data.sample_id,
            "Plot": data.plot,
            "Oasis": data.oasis,
        }
    )
    dummy = np.zeros((n_samples, 1), dtype=float)

    plot_fold = np.full(n_samples, -1, dtype=int)
    plot_cv = GroupKFold(n_splits=6)
    for fold, (_, test_idx) in enumerate(
        plot_cv.split(dummy, groups=data.plot), start=1
    ):
        plot_fold[test_idx] = fold
    assignment["GroupKFold_Plot"] = plot_fold

    oasis_fold = np.full(n_samples, -1, dtype=int)
    oasis_cv = GroupKFold(n_splits=6)
    for fold, (_, test_idx) in enumerate(
        oasis_cv.split(dummy, groups=data.oasis), start=1
    ):
        oasis_fold[test_idx] = fold
    assignment["GroupKFold_Oasis"] = oasis_fold

    random_fold = np.full(n_samples, -1, dtype=int)
    random_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (_, test_idx) in enumerate(random_cv.split(dummy), start=1):
        random_fold[test_idx] = fold
    assignment["KFold"] = random_fold

    loo_fold = np.full(n_samples, -1, dtype=int)
    for fold, (_, test_idx) in enumerate(LeaveOneOut().split(dummy), start=1):
        loo_fold[test_idx] = fold
    assignment["LOOCV"] = loo_fold

    for validation in VALIDATIONS:
        if (assignment[validation] < 1).any():
            raise AssertionError(f"Incomplete fold assignment for {validation}.")
    return assignment


def subset_indices(data: AnalysisData, path: str, stratum: str) -> np.ndarray:
    if path == "A_Global":
        mask = np.ones(len(data.sample_id), dtype=bool)
    elif path == "B1_Management":
        mask = data.management == stratum
    elif path == "B2_Topography":
        mask = data.topography == stratum
    else:
        raise ValueError(f"Unsupported modelling path: {path}")
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(f"No samples for path={path}, stratum={stratum}.")
    return indices


def outer_splits(
    subset: np.ndarray,
    fold_assignments: pd.DataFrame,
    validation: str,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    folds = fold_assignments[validation].to_numpy(dtype=int)
    result: list[tuple[np.ndarray, np.ndarray, str]] = []
    for fold in sorted(np.unique(folds[subset])):
        test_idx = subset[folds[subset] == fold]
        train_idx = subset[folds[subset] != fold]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        result.append((train_idx, test_idx, f"{validation}_{fold:03d}"))
    return result


def preprocessing_steps(scenario: str, wavelengths: np.ndarray) -> list[tuple[str, Any]]:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scenario == "RAW":
        return steps
    if scenario == "RAW_SG_MSC":
        steps.append(
            (
                "sg_msc",
                SavitzkyGolayMSC(
                    window_length=SG_WINDOW_LENGTH,
                    polyorder=SG_POLYORDER,
                    deriv=SG_DERIVATIVE,
                ),
            )
        )
        return steps
    if scenario == "CR":
        steps.append(("continuum_removal", UpperHullContinuumRemoval(wavelengths)))
        return steps
    raise ValueError(f"Unsupported spectral scenario: {scenario}")


def make_rf(mode: str, n_jobs: int) -> BaseEstimator:
    forest = RandomForestRegressor(
        n_estimators=RF_TREES,
        criterion="squared_error",
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )
    if mode == "direct":
        return forest
    if mode == "multi":
        return TransformedTargetRegressor(
            regressor=forest,
            transformer=StandardScaler(),
            check_inverse=True,
        )
    raise ValueError(f"Unsupported model mode: {mode}")


def fit_outer_model(
    config: EvaluationConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_plot_groups: np.ndarray,
    wavelengths: np.ndarray,
    n_jobs: int,
) -> tuple[BaseEstimator, dict[str, Any] | None]:
    if config.model_family == "RF":
        model = Pipeline(
            preprocessing_steps(config.scenario, wavelengths)
            + [("model", make_rf(config.mode, n_jobs))]
        )
        model.fit(X_train, y_train)
        return model, None

    if config.model_family != "PLSR":
        raise ValueError(f"Unsupported model family: {config.model_family}")

    unique_plots = np.unique(train_plot_groups)
    if unique_plots.size < 2:
        raise ValueError("Grouped inner PLSR selection requires at least two training plots.")
    inner_cv = GroupKFold(n_splits=min(5, unique_plots.size))
    inner_splits = list(inner_cv.split(X_train, y_train, groups=train_plot_groups))
    minimum_inner_train = min(len(train_idx) for train_idx, _ in inner_splits)
    maximum_components = min(25, X_train.shape[1], minimum_inner_train - 1)
    if maximum_components < 1:
        raise ValueError("No feasible PLSR component count for this outer calibration fold.")
    pipeline = Pipeline(
        preprocessing_steps(config.scenario, wavelengths)
        + [("model", PLSRegression(scale=True, max_iter=500))]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"model__n_components": list(range(1, maximum_components + 1))},
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_train, y_train, groups=train_plot_groups)
    selection = {
        "Selected_n_components": int(search.best_params_["model__n_components"]),
        "Best_inner_CV_RMSE": float(-search.best_score_),
        "Candidate_Min": 1,
        "Candidate_Max": int(maximum_components),
        "Inner_Splitter": "GroupKFold by Plot",
        "Inner_N_Splits": int(inner_cv.n_splits),
        "Inner_N_Plots": int(unique_plots.size),
    }
    return search, selection


def stable_seed(parts: Iterable[object]) -> int:
    digest = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + RANDOM_STATE) % (2**32 - 1)


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    variance_true = float(np.var(y_true, ddof=0))
    variance_pred = float(np.var(y_pred, ddof=0))
    covariance = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))
    denominator = variance_true + variance_pred + (mean_true - mean_pred) ** 2
    return float(2.0 * covariance / denominator) if denominator > 0 else np.nan


def pooled_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    finite = np.isfinite(true) & np.isfinite(predicted)
    true = true[finite]
    predicted = predicted[finite]
    if true.size < 2:
        raise ValueError("At least two finite pooled predictions are required for metrics.")
    residual = predicted - true
    rmse = float(math.sqrt(mean_squared_error(true, predicted)))
    sd_population = float(np.std(true, ddof=0))
    result = {
        "R2": float(r2_score(true, predicted)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(true, predicted)),
        "Bias": float(np.mean(residual)),
        "CCC": concordance_correlation_coefficient(true, predicted),
        "RPD": float(sd_population / rmse) if rmse > 0 else np.nan,
    }
    assert_metric_identity(result)
    return result


def assert_metric_identity(metrics: dict[str, float], tolerance: float = 1e-10) -> None:
    r2 = float(metrics["R2"])
    rpd = float(metrics["RPD"])
    if not np.isfinite(r2) or not np.isfinite(rpd) or rpd <= 0:
        raise AssertionError(f"Non-finite R2/RPD pair: R2={r2}, RPD={rpd}")
    expected_r2 = 1.0 - 1.0 / (rpd**2)
    if abs(r2 - expected_r2) > tolerance:
        raise AssertionError(
            f"Metric identity failed: R2={r2:.15g}, 1-1/RPD^2={expected_r2:.15g}."
        )


def bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    key_parts: Iterable[object],
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    rng = np.random.default_rng(stable_seed(key_parts))
    n = len(true)
    r2_values: list[float] = []
    rmse_values: list[float] = []
    for _ in range(iterations):
        sample = rng.integers(0, n, size=n)
        sampled_true = true[sample]
        sampled_pred = predicted[sample]
        if np.var(sampled_true, ddof=0) <= 0:
            continue
        r2_values.append(float(r2_score(sampled_true, sampled_pred)))
        rmse_values.append(float(math.sqrt(mean_squared_error(sampled_true, sampled_pred))))
    if len(r2_values) < int(iterations * 0.95):
        raise RuntimeError("Too many invalid bootstrap resamples for stable confidence intervals.")
    return {
        "R2_CI_Lower": float(np.percentile(r2_values, 2.5)),
        "R2_CI_Upper": float(np.percentile(r2_values, 97.5)),
        "RMSE_CI_Lower": float(np.percentile(rmse_values, 2.5)),
        "RMSE_CI_Upper": float(np.percentile(rmse_values, 97.5)),
        "Bootstrap_Iterations": int(iterations),
    }


def prediction_metadata(data: AnalysisData, index: int) -> dict[str, Any]:
    return {
        "Sample_ID": str(data.sample_id[index]),
        "Plot": str(data.plot[index]),
        "Oasis": str(data.oasis[index]),
        "Study_Area": str(data.study_area[index]),
        "OasisType": int(data.oasis_type[index]),
        "Management": str(data.management[index]),
        "Slope_Category": int(data.slope_category[index]),
        "Topography": str(data.topography[index]),
        "Depth": float(data.depth[index]),
    }


def _evaluate_outer_fold(
    data: AnalysisData,
    config: EvaluationConfig,
    fold_number: int,
    split: tuple[np.ndarray, np.ndarray, str],
    model_n_jobs: int,
    precomputed_rf_X: np.ndarray | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[float]],
    dict[str, list[float]],
]:
    train_idx, test_idx, fold_id = split
    validation_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    calibration_true = {target: [] for target in TARGETS}
    calibration_pred = {target: [] for target in TARGETS}

    X_train = data.X[train_idx]
    X_test = data.X[test_idx]
    Y_train = data.Y[train_idx]
    Y_test = data.Y[test_idx]

    rf_X_train: np.ndarray | None = None
    rf_X_test: np.ndarray | None = None
    if config.model_family == "RF":
        if precomputed_rf_X is not None:
            rf_X_train = precomputed_rf_X[train_idx]
            rf_X_test = precomputed_rf_X[test_idx]
        else:
            # X preprocessing is target-independent. Fit it once on the outer
            # calibration fold, then reuse the transformed matrices for the
            # three direct forests or the single native multi-output forest.
            feature_pipeline = Pipeline(
                preprocessing_steps(config.scenario, data.wavelengths)
            )
            rf_X_train = np.asarray(
                feature_pipeline.fit_transform(X_train), dtype=float
            )
            rf_X_test = np.asarray(feature_pipeline.transform(X_test), dtype=float)

    if config.mode == "direct":
        for target_index, target in enumerate(TARGETS):
            if config.model_family == "RF":
                if rf_X_train is None or rf_X_test is None:
                    raise AssertionError("RF preprocessing matrices were not created.")
                model = make_rf("direct", model_n_jobs)
                model.fit(rf_X_train, Y_train[:, target_index])
                selection = None
                train_prediction = np.asarray(
                    model.predict(rf_X_train), dtype=float
                ).reshape(-1)
                test_prediction = np.asarray(
                    model.predict(rf_X_test), dtype=float
                ).reshape(-1)
            else:
                model, selection = fit_outer_model(
                    config,
                    X_train,
                    Y_train[:, target_index],
                    data.plot[train_idx],
                    data.wavelengths,
                    model_n_jobs,
                )
                train_prediction = np.asarray(
                    model.predict(X_train), dtype=float
                ).reshape(-1)
                test_prediction = np.asarray(
                    model.predict(X_test), dtype=float
                ).reshape(-1)
            calibration_true[target].extend(Y_train[:, target_index].tolist())
            calibration_pred[target].extend(train_prediction.tolist())
            for local_index, sample_index in enumerate(test_idx):
                row = {
                    "Scenario": config.scenario,
                    "Path": config.path,
                    "Stratum": config.stratum,
                    "Validation": config.validation,
                    "ModelFamily": config.model_family,
                    "Mode": config.mode,
                    "ModelFlow": config.model_flow,
                    "Fold": fold_id,
                    "Target": target,
                    "Observed": float(Y_test[local_index, target_index]),
                    "Predicted": float(test_prediction[local_index]),
                }
                row["Residual"] = row["Predicted"] - row["Observed"]
                row.update(prediction_metadata(data, int(sample_index)))
                validation_rows.append(row)
            if selection is not None:
                component_rows.append(
                    {
                        "Scenario": config.scenario,
                        "Path": config.path,
                        "Stratum": config.stratum,
                        "Validation": config.validation,
                        "Mode": config.mode,
                        "Target": target,
                        "Fold": fold_id,
                        "Outer_Fold_Number": fold_number,
                        "N_Train": int(len(train_idx)),
                        "N_Test": int(len(test_idx)),
                        **selection,
                    }
                )
    elif config.mode == "multi":
        if config.model_family == "RF":
            if rf_X_train is None or rf_X_test is None:
                raise AssertionError("RF preprocessing matrices were not created.")
            model = make_rf("multi", model_n_jobs)
            model.fit(rf_X_train, Y_train)
            selection = None
            train_prediction = np.asarray(model.predict(rf_X_train), dtype=float)
            test_prediction = np.asarray(model.predict(rf_X_test), dtype=float)
        else:
            model, selection = fit_outer_model(
                config,
                X_train,
                Y_train,
                data.plot[train_idx],
                data.wavelengths,
                model_n_jobs,
            )
            train_prediction = np.asarray(model.predict(X_train), dtype=float)
            test_prediction = np.asarray(model.predict(X_test), dtype=float)
        for target_index, target in enumerate(TARGETS):
            calibration_true[target].extend(Y_train[:, target_index].tolist())
            calibration_pred[target].extend(train_prediction[:, target_index].tolist())
            for local_index, sample_index in enumerate(test_idx):
                row = {
                    "Scenario": config.scenario,
                    "Path": config.path,
                    "Stratum": config.stratum,
                    "Validation": config.validation,
                    "ModelFamily": config.model_family,
                    "Mode": config.mode,
                    "ModelFlow": config.model_flow,
                    "Fold": fold_id,
                    "Target": target,
                    "Observed": float(Y_test[local_index, target_index]),
                    "Predicted": float(test_prediction[local_index, target_index]),
                }
                row["Residual"] = row["Predicted"] - row["Observed"]
                row.update(prediction_metadata(data, int(sample_index)))
                validation_rows.append(row)
            if selection is not None:
                component_rows.append(
                    {
                        "Scenario": config.scenario,
                        "Path": config.path,
                        "Stratum": config.stratum,
                        "Validation": config.validation,
                        "Mode": config.mode,
                        "Target": target,
                        "Fold": fold_id,
                        "Outer_Fold_Number": fold_number,
                        "N_Train": int(len(train_idx)),
                        "N_Test": int(len(test_idx)),
                        **selection,
                    }
                )
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")

    return validation_rows, component_rows, calibration_true, calibration_pred


def evaluate_configuration(
    data: AnalysisData,
    fold_assignments: pd.DataFrame,
    config: EvaluationConfig,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    subset = subset_indices(data, config.path, config.stratum)
    splits = outer_splits(subset, fold_assignments, config.validation)
    if not splits:
        raise ValueError(f"No outer splits for {config.key}.")

    validation_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    calibration_true = {target: [] for target in TARGETS}
    calibration_pred = {target: [] for target in TARGETS}

    precomputed_rf_X: np.ndarray | None = None
    if config.model_family == "RF" and config.scenario == "CR":
        if not np.isfinite(data.X).all():
            raise ValueError(
                "Stateless CR caching requires finite raw spectra; missing values "
                "must instead be imputed within each outer fold."
            )
        cache_key = id(data)
        if cache_key not in _STATELESS_CR_CACHE:
            # CR is fitted independently to each spectrum and has no learned
            # cross-sample parameters. With a complete raw matrix, computing it
            # once here is exactly equivalent to repeating it in every fold.
            transformer = UpperHullContinuumRemoval(data.wavelengths)
            _STATELESS_CR_CACHE[cache_key] = np.asarray(
                transformer.fit_transform(data.X), dtype=float
            )
        precomputed_rf_X = _STATELESS_CR_CACHE[cache_key]

    fold_inputs = list(enumerate(splits, start=1))
    if config.model_family == "RF" and n_jobs != 1:
        max_workers = n_jobs if n_jobs > 0 else min(4, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_outer_fold,
                    data,
                    config,
                    fold_number,
                    split,
                    1,
                    precomputed_rf_X,
                )
                for fold_number, split in fold_inputs
            ]
            fold_results = [future.result() for future in futures]
    else:
        fold_results = [
            _evaluate_outer_fold(
                data,
                config,
                fold_number,
                split,
                n_jobs,
                precomputed_rf_X,
            )
            for fold_number, split in fold_inputs
        ]

    for fold_validation, fold_components, fold_true, fold_pred in fold_results:
        validation_rows.extend(fold_validation)
        component_rows.extend(fold_components)
        for target in TARGETS:
            calibration_true[target].extend(fold_true[target])
            calibration_pred[target].extend(fold_pred[target])

    prediction_frame = pd.DataFrame(validation_rows)
    expected_predictions = len(subset) * len(TARGETS)
    if len(prediction_frame) != expected_predictions:
        raise AssertionError(
            f"{config.key}: expected {expected_predictions} validation rows, "
            f"found {len(prediction_frame)}."
        )
    duplicate_key = ["Sample_ID", "Target"]
    if prediction_frame.duplicated(duplicate_key).any():
        raise AssertionError(f"{config.key}: duplicate pooled validation predictions.")

    calibration_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        metrics = pooled_metrics(calibration_true[target], calibration_pred[target])
        calibration_rows.append(
            {
                "Scenario": config.scenario,
                "Path": config.path,
                "Stratum": config.stratum,
                "Validation": config.validation,
                "ModelFamily": config.model_family,
                "Mode": config.mode,
                "ModelFlow": config.model_flow,
                "Phase": "Calibration",
                "Target": target,
                "N_Predictions": int(len(calibration_true[target])),
                "N_Unique_Samples": int(len(subset)),
                "N_Folds": int(len(splits)),
                **metrics,
            }
        )
    return prediction_frame, calibration_rows, component_rows


SUMMARY_KEYS = [
    "Scenario",
    "Path",
    "Stratum",
    "Validation",
    "ModelFamily",
    "Mode",
    "ModelFlow",
    "Target",
]


def summarize_validation_predictions(
    predictions: pd.DataFrame,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(SUMMARY_KEYS, sort=False, dropna=False)
    for key, group in grouped:
        key_values = dict(zip(SUMMARY_KEYS, key))
        true = group["Observed"].to_numpy(dtype=float)
        predicted = group["Predicted"].to_numpy(dtype=float)
        metrics = pooled_metrics(true, predicted)
        intervals = bootstrap_intervals(
            true,
            predicted,
            [*key, "bootstrap"],
            iterations=bootstrap_iterations,
        )
        rows.append(
            {
                **key_values,
                "Phase": "Validation",
                "N_Predictions": int(len(group)),
                "N_Unique_Samples": int(group["Sample_ID"].nunique()),
                "N_Folds": int(group["Fold"].nunique()),
                **metrics,
                **intervals,
            }
        )
    summary = pd.DataFrame(rows)
    assert_summary_metrics(summary)
    return summary.sort_values(SUMMARY_KEYS).reset_index(drop=True)


def assert_summary_metrics(summary: pd.DataFrame) -> None:
    if summary.empty:
        raise AssertionError("Summary object is empty.")
    for row in summary.itertuples(index=False):
        assert_metric_identity({"R2": row.R2, "RPD": row.RPD})
    if summary.duplicated(SUMMARY_KEYS).any():
        raise AssertionError("Summary object contains duplicate configuration-target rows.")


def build_evaluation_plan() -> list[EvaluationConfig]:
    plan: list[EvaluationConfig] = []
    for scenario in SCENARIOS:
        for model_family in ("PLSR", "RF"):
            for mode in ("direct", "multi"):
                plan.append(
                    EvaluationConfig(
                        scenario=scenario,
                        path="A_Global",
                        stratum="ALL",
                        validation="GroupKFold_Plot",
                        model_family=model_family,
                        mode=mode,
                    )
                )

    for model_family in ("PLSR", "RF"):
        for mode in ("direct", "multi"):
            plan.append(
                EvaluationConfig(
                    scenario="CR",
                    path="A_Global",
                    stratum="ALL",
                    validation="GroupKFold_Oasis",
                    model_family=model_family,
                    mode=mode,
                )
            )

    for validation in ("KFold", "LOOCV"):
        for mode in ("direct", "multi"):
            plan.append(
                EvaluationConfig(
                    scenario="CR",
                    path="A_Global",
                    stratum="ALL",
                    validation=validation,
                    model_family="RF",
                    mode=mode,
                )
            )

    path_strata = {
        "B1_Management": ("Modern", "Traditional"),
        "B2_Topography": ("Upslope", "Midslope", "Downslope"),
    }
    for path, strata in path_strata.items():
        for stratum in strata:
            for validation in VALIDATIONS:
                for mode in ("direct", "multi"):
                    plan.append(
                        EvaluationConfig(
                            scenario="CR",
                            path=path,
                            stratum=stratum,
                            validation=validation,
                            model_family="RF",
                            mode=mode,
                        )
                    )

    keys = [config.key for config in plan]
    if len(keys) != len(set(keys)):
        raise AssertionError("Evaluation plan contains duplicate configurations.")
    return plan


def save_run_configuration(path: Path, raw_path: Path, plan: list[EvaluationConfig]) -> None:
    payload = {
        "raw_input": str(raw_path.resolve()),
        "random_state": RANDOM_STATE,
        "rf": {
            "n_estimators": RF_TREES,
            "criterion": "squared_error",
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "multi_target_response_scaling": "StandardScaler fitted on each outer calibration fold",
        },
        "sg": {
            "window_length": SG_WINDOW_LENGTH,
            "polyorder": SG_POLYORDER,
            "deriv": SG_DERIVATIVE,
            "order": "Savitzky-Golay then MSC",
        },
        "plsr": {
            "component_range": "1-25, capped only by inner-fold feasibility",
            "inner_validation": "GroupKFold by Plot on outer calibration data only",
            "scoring": "neg_root_mean_squared_error",
            "scale": True,
        },
        "validation": {
            "GroupKFold_Plot": "6 folds assigned once from 18 recorded plot values",
            "GroupKFold_Oasis": "6 folds assigned once from 6 approved oasis stems",
            "KFold": "5 folds, shuffled, random_state=42",
            "LOOCV": "216 leave-one-sample-out folds",
        },
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "evaluation_plan": [config.__dict__ for config in plan],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_no_precomputed_cr_reference(code_root: Path) -> None:
    forbidden = "cr_spectra"
    violations: list[str] = []
    for script in code_root.glob("*.py"):
        if script.name in {
            "verify_continuum_removal.py",
            "prepare_sections_2_3.py",
            "workflow_core.py",
        }:
            continue
        text = script.read_text(encoding="utf-8").lower()
        if forbidden in text:
            violations.append(script.name)
    if violations:
        raise AssertionError(
            "Production code references a precomputed CR file: " + ", ".join(violations)
        )
