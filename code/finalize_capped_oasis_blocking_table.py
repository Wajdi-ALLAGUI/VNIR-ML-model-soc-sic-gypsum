from __future__ import annotations

import shutil

import sys

from pathlib import Path

import numpy as np

import pandas as pd

from openpyxl import load_workbook

from openpyxl.styles import Alignment, Font, PatternFill

from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import GridSearchCV, GroupKFold

from sklearn.pipeline import Pipeline

from build_sections_5_8 import (  # noqa: E402
    make_paths,
    sort_table,
    write_table_outputs,
)

from workflow_core import (  # noqa: E402
    SUMMARY_KEYS,
    TARGETS,
    EvaluationConfig,
    assert_summary_metrics,
    load_analysis_data,
    make_master_fold_assignments,
    outer_splits,
    prediction_metadata,
    preprocessing_steps,
    summarize_validation_predictions,
)

def fit_capped_oasis_plsr(data, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_base = {
        "scenario": "CR",
        "path": "A_Global",
        "stratum": "ALL",
        "validation": "GroupKFold_Oasis",
        "model_family": "PLSR",
    }
    subset = np.arange(len(data.sample_id), dtype=int)
    predictions: list[dict[str, object]] = []
    components: list[dict[str, object]] = []

    for mode in ("direct", "multi"):
        config = EvaluationConfig(mode=mode, **config_base)
        for fold_number, (train_idx, test_idx, fold_id) in enumerate(
            outer_splits(subset, assignments, "GroupKFold_Oasis"), start=1
        ):
            inner_cv = GroupKFold(n_splits=5)
            train_groups = data.plot[train_idx]
            if mode == "direct":
                fit_targets = [(target, target_index) for target_index, target in enumerate(TARGETS)]
            else:
                fit_targets = [("ALL", None)]

            for fit_target, target_index in fit_targets:
                pipeline = Pipeline(
                    preprocessing_steps("CR", data.wavelengths)
                    + [("model", PLSRegression(scale=True, max_iter=500))]
                )
                search = GridSearchCV(
                    pipeline,
                    {"model__n_components": [1, 2, 3, 4]},
                    cv=inner_cv,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                    refit=True,
                    error_score="raise",
                )
                if mode == "direct":
                    y_train = data.Y[train_idx, int(target_index)]
                else:
                    y_train = data.Y[train_idx]
                search.fit(data.X[train_idx], y_train, groups=train_groups)
                fold_prediction = np.asarray(search.predict(data.X[test_idx]), dtype=float)
                if mode == "direct":
                    fold_prediction = fold_prediction.reshape(-1, 1)
                    output_targets = [(fit_target, int(target_index), 0)]
                else:
                    output_targets = [
                        (target, index, index) for index, target in enumerate(TARGETS)
                    ]

                selected = int(search.best_params_["model__n_components"])
                for target, observed_index, predicted_index in output_targets:
                    components.append(
                        {
                            "Scenario": "CR",
                            "Path": "A_Global",
                            "Stratum": "ALL",
                            "Validation": "GroupKFold_Oasis",
                            "Mode": mode,
                            "Target": target,
                            "Fold": fold_id,
                            "Outer_Fold_Number": fold_number,
                            "N_Train": len(train_idx),
                            "N_Test": len(test_idx),
                            "Selected_n_components": selected,
                            "Candidate_Min": 1,
                            "Candidate_Max": 4,
                            "Inner_Splitter": "GroupKFold by Plot",
                            "Inner_N_Splits": 5,
                            "Inner_N_Plots": len(np.unique(train_groups)),
                        }
                    )
                    for local_index, sample_index in enumerate(test_idx):
                        observed = float(data.Y[sample_index, observed_index])
                        predicted = float(fold_prediction[local_index, predicted_index])
                        row = {
                            "Scenario": "CR",
                            "Path": "A_Global",
                            "Stratum": "ALL",
                            "Validation": "GroupKFold_Oasis",
                            "ModelFamily": "PLSR",
                            "Mode": mode,
                            "ModelFlow": config.model_flow,
                            "Fold": fold_id,
                            "Target": target,
                            "Observed": observed,
                            "Predicted": predicted,
                            "Residual": predicted - observed,
                        }
                        row.update(prediction_metadata(data, int(sample_index)))
                        predictions.append(row)

    prediction_frame = pd.DataFrame(predictions)
    if len(prediction_frame) != 216 * len(TARGETS) * 2:
        raise AssertionError("Incomplete capped oasis-blocked PLSR predictions.")
    return prediction_frame, pd.DataFrame(components)
