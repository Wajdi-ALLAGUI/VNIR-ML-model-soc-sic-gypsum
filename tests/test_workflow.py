"""Synthetic tests only: no experimental records or private inputs."""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "code"))

from pipeline_preprocessing import (
    common_range_columns, continuum_remove_spectrum, upper_hull_indices,
    UpperHullContinuumRemoval,
)
from workflow_core import (
    EvaluationConfig, SavitzkyGolayMSC, bootstrap_intervals,
    build_evaluation_plan, fit_outer_model, make_rf, pooled_metrics,
)


class SyntheticScientificTests(unittest.TestCase):
    def test_pooled_metrics_population_rpd_and_bias(self):
        y = np.array([1.0, 2.0, 4.0, 7.0])
        p = np.array([1.3, 2.1, 3.7, 6.4])
        result = pooled_metrics(y, p)
        rmse = np.sqrt(np.mean((p - y) ** 2))
        self.assertAlmostEqual(result["RMSE"], rmse)
        self.assertAlmostEqual(result["MAE"], np.mean(np.abs(p - y)))
        self.assertAlmostEqual(result["Bias"], np.mean(p - y))
        self.assertAlmostEqual(result["RPD"], np.std(y, ddof=0) / rmse)
        self.assertAlmostEqual(result["R2"], 1.0 - 1.0 / result["RPD"] ** 2)
        ccc = 2 * np.mean((y-y.mean()) * (p-p.mean())) / (
            y.var() + p.var() + (y.mean()-p.mean()) ** 2
        )
        self.assertAlmostEqual(result["CCC"], ccc)

    def test_loocv_is_scored_after_pooling(self):
        y = [1.0, 3.0, 5.0]
        p = [1.5, 2.7, 4.8]
        self.assertTrue(np.isfinite(pooled_metrics(y, p)["R2"]))
        with self.assertRaises(ValueError):
            pooled_metrics(y[:1], p[:1])

    def test_bootstrap_reproducible(self):
        y = np.linspace(0.0, 10.0, 60)
        p = y + np.sin(y) * 0.25
        first = bootstrap_intervals(y, p, ["synthetic"], iterations=100)
        second = bootstrap_intervals(y, p, ["synthetic"], iterations=100)
        self.assertEqual(first, second)
        self.assertLessEqual(first["R2_CI_Lower"], first["R2_CI_Upper"])

    def test_cr_hull_endpoints_and_interpolation(self):
        w = np.array([400.0, 900.0, 1600.0, 2500.0])
        r = np.array([0.8, 0.4, 0.5, 0.9])
        np.testing.assert_array_equal(upper_hull_indices(w, r), [0, 3])
        expected = r / np.interp(w, w[[0, 3]], r[[0, 3]])
        np.testing.assert_allclose(continuum_remove_spectrum(w, r), expected)
        self.assertEqual(expected[0], 1.0)
        self.assertEqual(expected[-1], 1.0)

    def test_cr_independent_per_spectrum_and_range_validation(self):
        w = np.arange(400.0, 2501.0)
        x = np.linspace(0.0, 1.0, len(w))
        a = 0.8 - 0.2 * np.exp(-((x-0.4) / 0.1) ** 2)
        b = 0.7 - 0.1 * np.exp(-((x-0.7) / 0.1) ** 2)
        together = UpperHullContinuumRemoval(w).fit_transform(np.vstack([a, b]))
        alone = UpperHullContinuumRemoval(w).fit_transform(a[None, :])
        np.testing.assert_allclose(together[0], alone[0])
        self.assertTrue(np.all(together <= 1.0 + 1e-12))
        with self.assertRaises(ValueError):
            UpperHullContinuumRemoval(w[:-1]).fit(a[None, :-1])

    def test_common_range_ignores_external_bands(self):
        columns = ["identifier", "X399"] + [f"X{i}" for i in range(400, 2501)] + ["X2501"]
        df = pd.DataFrame(columns=columns)
        found = common_range_columns(df)
        self.assertEqual(len(found), 2101)
        self.assertEqual((found[0], found[-1]), ("X400", "X2500"))

    def test_msc_reference_is_calibration_only(self):
        x = np.linspace(0.0, 4.0, 31)
        training = np.vstack([np.sin(x) + 2, 1.2*np.sin(x) + 2.5, 0.8*np.sin(x) + 1.8])
        heldout = (2.0*np.sin(x) + 7.0)[None, :]
        transform = SavitzkyGolayMSC().fit(training)
        expected = savgol_filter(training, 11, 3, deriv=1, axis=1, mode="interp").mean(axis=0)
        np.testing.assert_allclose(transform.reference_, expected)
        before = transform.reference_.copy()
        predicted = transform.transform(heldout)
        np.testing.assert_array_equal(transform.reference_, before)
        self.assertEqual(predicted.shape, heldout.shape)
        self.assertTrue(np.isfinite(predicted).all())

    def test_rf_native_multioutput_and_parameters(self):
        direct = make_rf("direct", n_jobs=1)
        multi = make_rf("multi", n_jobs=1)
        self.assertIsInstance(direct, RandomForestRegressor)
        self.assertIsInstance(multi, TransformedTargetRegressor)
        self.assertIsInstance(multi.transformer, StandardScaler)
        for forest in (direct, multi.regressor):
            self.assertEqual(forest.n_estimators, 260)
            self.assertEqual(forest.random_state, 42)
            self.assertEqual(forest.max_features, "sqrt")
            self.assertEqual(forest.criterion, "squared_error")
            self.assertEqual(forest.min_samples_leaf, 1)
            self.assertIsNone(forest.max_depth)

    def test_synthetic_direct_and_multitarget_rf_fit(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(24, 5))
        Y = np.column_stack([X[:, 0], X[:, 1]*3, X[:, 2]*10])
        for mode in ("direct", "multi"):
            config = EvaluationConfig("RAW", "A_Global", "ALL", "GroupKFold_Plot", "RF", mode)
            model, selection = fit_outer_model(
                config, X[:20], Y[:20, 0] if mode == "direct" else Y[:20],
                np.repeat(np.arange(5), 4), np.arange(400, 405), 1,
            )
            self.assertIsNone(selection)
            self.assertTrue(np.isfinite(model.predict(X[20:])).all())

    def test_synthetic_nested_plsr_passes_training_groups(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 4))
        y = X[:, 0]*0.8 + rng.normal(scale=0.15, size=20)
        config = EvaluationConfig("RAW", "A_Global", "ALL", "GroupKFold_Plot", "PLSR", "direct")
        search, selection = fit_outer_model(
            config, X, y, np.repeat(np.arange(5), 4), np.arange(400, 404), 1
        )
        self.assertEqual(selection["Inner_N_Splits"], 5)
        self.assertEqual(selection["Inner_N_Plots"], 5)
        self.assertEqual(selection["Candidate_Max"], 4)
        self.assertIn(selection["Selected_n_components"], [1, 2, 3, 4])
        self.assertTrue(np.isfinite(search.predict(X[:2])).all())

    def test_plan_contains_all_60_unique_configurations(self):
        plan = build_evaluation_plan()
        self.assertEqual(len(plan), 60)
        self.assertEqual(len({c.key for c in plan}), 60)
        self.assertEqual({c.validation for c in plan}, {
            "GroupKFold_Plot", "GroupKFold_Oasis", "KFold", "LOOCV"
        })

    def test_final_capped_function_has_exact_grid(self):
        tree = ast.parse((PACKAGE / "code" / "finalize_capped_oasis_blocking_table.py").read_text(encoding="utf-8"))
        grids = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)]
        found = []
        for grid in grids:
            for key, value in zip(grid.keys, grid.values):
                if isinstance(key, ast.Constant) and key.value == "model__n_components":
                    found.append(ast.literal_eval(value))
        self.assertEqual(found, [[1, 2, 3, 4]])

    def test_all_packaged_modules_import_without_private_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            old = os.environ.get("VNIR_OUTPUT_ROOT")
            os.environ["VNIR_OUTPUT_ROOT"] = temporary
            try:
                for path in sorted((PACKAGE / "code").glob("*.py")):
                    importlib.import_module(path.stem)
                supplementary = importlib.import_module("rebuild_submitted_supplementary_figures")
                helper = supplementary.load_reference_module()
                self.assertTrue(callable(helper.plot_shap_figures))
            finally:
                if old is None:
                    os.environ.pop("VNIR_OUTPUT_ROOT", None)
                else:
                    os.environ["VNIR_OUTPUT_ROOT"] = old

    def test_bundled_supplementary_style_renders_synthetic_data(self):
        import matplotlib.pyplot as plt
        import submitted_plot_style as style

        rng = np.random.default_rng(42)
        frame = pd.DataFrame({
            "Management": np.tile(np.repeat(["Modern", "Traditional"], 12), 3),
            "Topography": np.repeat(["Upslope", "Midslope", "Downslope"], 24),
            "Depth": np.tile([0, 5, 10, 30, 60, 90], 12),
            "SOC": rng.uniform(0.1, 3.0, 72),
            "SIC": rng.uniform(0.5, 4.0, 72),
            "Gypsum": rng.uniform(0.1, 30.0, 72),
        })
        count = []

        def inspect_figure(fig, *unused):
            fig.canvas.draw()
            count.append(len(fig.axes))
            plt.close(fig)

        with patch.object(style, "save_figure", side_effect=inspect_figure):
            paths = type("PlotPaths", (), {"figures": PACKAGE / "outputs"})()
            style.plot_management_topography_joint_distributions(frame, paths)
            style.plot_correlation_heatmap(frame, paths)
        self.assertEqual(count, [3, 2])


if __name__ == "__main__":
    unittest.main()
