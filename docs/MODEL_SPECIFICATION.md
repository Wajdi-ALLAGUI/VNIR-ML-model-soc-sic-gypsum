# Exact Scientific Specification

This document describes the CURRENT REVISED workflow, not the earlier submitted
run. The scientific cores are unchanged copies of that revised source.

## Spectral Scenarios

All models use exactly 2,101 bands from 400 through 2500 nm inclusive at 1 nm
spacing. Deterministic band selection is independent of target values.

| Scenario | Model input and operations, in order |
|---|---|
| RAW | Fold-fitted median imputation; raw reflectance |
| RAW_SG_MSC | Fold-fitted median imputation; SG first derivative; MSC |
| CR | Fold-fitted median imputation; independent per-spectrum continuum removal |

SG uses `scipy.signal.savgol_filter(window_length=11, polyorder=3, deriv=1,
axis=1, mode="interp")`; default `delta=1.0` agrees with the 1 nm spacing.
MSC uses the mean SG-transformed calibration spectrum as its reference.
Each SG-transformed spectrum is least-squares regressed on that reference and
an intercept; correction is `(spectrum - intercept) / slope`. The calibration
reference is reused, without refitting, for validation spectra.

CR uses an upper monotone-chain convex hull (least concave majorant), fitted
independently for each spectrum over the full 400-2500 nm interval. Both range
endpoints are retained, interior collinear vertices are removed, and hull
vertices are connected by linear interpolation (`numpy.interp`). Reflectance
is divided by the interpolated continuum. Non-positive/invalid continua raise
an error. The RF implementation caches CR only when the raw matrix is finite;
this sample-wise cache does not learn any cross-sample parameter.

## Models and Responses

All random procedures use seed 42. All RF validation schemes use **260 trees**.

| Model | Exact behaviour |
|---|---|
| Direct RF | Independent `RandomForestRegressor` for each target; response not standardized |
| Multi-target RF | ONE native multi-output `RandomForestRegressor` inside `TransformedTargetRegressor(StandardScaler())` |
| Direct PLSR | Independent `PLSRegression(scale=True, max_iter=500)` and component search for each target |
| Multi-target PLSR | One `PLSRegression` with the SOC/SIC/gypsum response matrix and one component search |

RF: `n_estimators=260`, `criterion="squared_error"`, `max_depth=None`,
`min_samples_leaf=1`, `max_features="sqrt"`, `random_state=42`;
other settings retain the defaults of scikit-learn 1.8.0, including
`bootstrap=True` and `min_samples_split=2`. The response StandardScaler of
multi-target RF is fitted only on outer training data, and predictions are
inverse-transformed to original percent units. This is **not**
`MultiOutputRegressor`, and it is not the unscaled native multi-output model
from the older submitted run. No extra RF feature StandardScaler is added.

The forest recursively selects wavelength splits that reduce squared error.
Direct forests optimize one response. The native multi-output forest shares
tree partitions across standardized responses. PLSR learns latent spectral
components related to the response(s); it does not choose wavelengths by a
top-five SHAP ranking before fitting.

PLSR tuning uses `GridSearchCV` on the ENTIRE preprocessing/model pipeline,
`scoring="neg_root_mean_squared_error"`, `refit=True`, `error_score="raise"`.
Inner validation is `GroupKFold(n_splits=min(5, number_of_training_plots))`
with the training **plot** labels passed as `groups`. Candidates are
`1..min(25, n_features, minimum_inner_training_size - 1)` inclusive.
For multi-target PLSR the RMSE score averages target-wise errors in original
target units; no separate equal-weight response scaler is added to the search.

### Final Oasis/Site-Blocked PLSR Cap

The final CR/global oasis-blocked PLSR comparison uses candidates **1, 2, 3, 4**
for both direct and multi-target PLSR. Its six outer folds leave one oasis out.
Its five-fold inner search still groups by TRAINING PLOT, not by training
oasis. This is stated explicitly rather than implying an oasis-grouped inner
loop that the code does not perform. All preprocessing is refitted within
the inner training pipeline.

The main diagnostics keep the baseline uncapped search for auditability.
`finalize_capped_oasis_blocking_table.py` generates a separate capped summary;
the final Table S4 substitutes these capped PLSR rows and keeps RF unchanged.
Do not use the uncapped oasis PLSR diagnostic rows as final manuscript values.
The display term "site-blocked" in figures refers to the six-oasis blocking
level; the internal identifier remains `GroupKFold_Oasis`.

## Paths and Paired Validation

| Path | Training/evaluation scope | Models/scenarios |
|---|---|---|
| A_Global | All 216 samples | RAW, SG+MSC, CR; PLSR/RF; direct/multi under plot blocking |
| A_Global additional | All samples | CR; PLSR/RF under oasis blocking; RF under random KFold and LOOCV |
| B1_Management | Separate recorded modern/traditional strata | CR RF, direct/multi, four validation schemes |
| B2_Topography | Separate upslope/midslope/downslope strata | CR RF, direct/multi, four validation schemes |

Outer master partitions are assigned once to the complete dataset:
plot-blocked GroupKFold (18 plots, 6 folds), oasis-blocked GroupKFold (6 oases,
6 folds), shuffled KFold (5 folds, seed 42), and sample-wise leave-one-out.
Stratified models intersect these SAME master training/test memberships with
the stratum and skip empty intersections. They do not invent independent
stratum partitions. Consequently a topographic stratum can have only two
nonempty plot-blocked folds even though the global master splitter has six.
Diagnostics record the actual fold and sample counts.

## Performance and Uncertainty

Held-out predictions are concatenated across outer folds; metrics are
calculated ONCE per configuration and target, never as averaged fold metrics.
Calibration diagnostics likewise pool training-fold predictions; calibration
contains repeated training appearances and is not held-out evidence.

| Metric | Definition |
|---|---|
| R2 | `1 - sum((predicted-observed)^2) / sum((observed-mean(observed))^2)` |
| RMSE | Square root of mean squared residual |
| MAE | Mean absolute residual |
| Bias | Mean(predicted - observed) |
| CCC | Lin concordance: `2*cov / (var_y + var_pred + (mean_y-mean_pred)^2)` |
| RPD | Population SD of pooled observed values (`ddof=0`) / RMSE |

CCC also uses population variances/covariance. All error metrics are in
percentage points. R2/CCC/RPD are dimensionless. The unrounded rows are checked
against `R2 = 1 - 1/RPD^2` with tolerance `1e-10`. This identity need not hold
exactly after independently rounding R2 and RPD to two decimal places.

95% bootstrap intervals for R2 and RMSE use 1,000 paired **sample-row**
resamples of pooled observed/predicted pairs and percentile limits 2.5/97.5.
Seeds are derived from configuration keys plus seed 42. This is not a
plot/oasis cluster bootstrap or a repeated refitting bootstrap. Those
intervals should not be described as uncertainty across independent oases.

## Statistics, Interpretation and Graphics

Main distribution-plot inference uses `statsmodels.MixedLM` with formula
`Response ~ C(Depth) + C(Topography) + C(Management)`, a plot random intercept,
REML, Powell optimizer, maximum 2,000 iterations, and joint term Wald tests.
The code raises on failure rather than silently replacing the analysis.
Retained supplementary distribution styling uses separate Kruskal-Wallis
factor tests; this is distinct from the main mixed-effects omnibus tests.

For each target and CR/SG+MSC scenario, SHAP chooses the best global RF mode by
plot-blocked pooled R2 (RMSE breaks ties). The chosen model is refitted on all
216 samples and explained using `shap.TreeExplainer`; mean absolute SHAP
values rank wavelengths. Multi-output SHAP values are multiplied by the
target scaler to recover original target units. Exactly five wavelengths per
target/scenario are reported; full wavelength importance files are generated
privately. This is post hoc interpretation, not held-out performance or
causal attribution. Selection by the same CV results is not an independent
estimate of the winner's selection-adjusted performance.

PCA is descriptive: computed CR spectra, median imputation, feature
StandardScaler, and two principal components. Mean spectra for display use
the whole dataset. These whole-dataset visualization fits are not used to
preprocess held-out model predictions.

Figures use the retained submitted designs, approved black typography,
PNG/vector PDF export, and final SHAP styling. Final tables use the six metrics
in the order R2, RMSE, MAE, Bias, CCC, RPD. The package does not contain the
submitted manuscript, historical comparison reports, or experimental outputs.
