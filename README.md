# VNIR ML model: predicting SOC, SIC and gypsum from soil spectra

Python workflow for visible–near-infrared (VNIR, 400–2500 nm) spectroscopy of dryland agricultural soils (gypsiferous oasis soils, southern Tunisia). It predicts **soil organic carbon (SOC), soil inorganic carbon (SIC) and gypsum** and tests how well the models transfer to new plots and new oases.

**Manuscript:** Allagui W, Zayani H, Bernardini L, Pichelin P, Brahim N, Bol R, Aichi H, Fouad Y, Wanek W (2026). Validation-aware VNIR spectroscopy reveals target-specific transferability of SOC, SIC and gypsum predictions in dryland agricultural soils. *Soil Systems*, under revision. [Preprint](https://doi.org/10.20944/preprints202609.0396.v1)

## What the workflow does

| Step | Method |
| --- | --- |
| Preprocessing | Wavelength selection, Savitzky–Golay, MSC (fitted on training data only), per-spectrum continuum removal |
| Models | PLSR with nested grouped component selection; random forest (one model per target and native multi-output) |
| Validation | LOOCV, plot-blocked (18 plots) and oasis-blocked (6 oases) grouped cross-validation |
| Metrics | Pooled R², RMSE, MAE, bias, CCC, RPD; paired-bootstrap 95% intervals; ΔR² between random and blocked validation |
| Interpretation | SHAP values of the best random-forest models (top wavelengths); mixed-effects models of soil properties by depth, topography and management |
| Outputs | Manuscript figures 1–5, supplementary figures and tables |

60 model configurations in total. The key idea: random cross-validation overestimates accuracy when samples from the same site are correlated, so every model is also tested on oases it has never seen.

**No experimental data, sample-level predictions, fitted models, manuscript files, or result tables are included.** Reproducing the numbers requires the private input dataset. Without it, anyone can read the full code and run the synthetic unit tests (14 tests, all passing).

## Setup

The reference environment is Python 3.14.0. Exact installed library versions
are recorded in `requirements.txt`; those pins reflect the working environment,
not a claim that all older/newer versions give identical results.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_workflow.py --check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_workflow.py --check
.venv/bin/python -m unittest discover -s tests -v
```

## Run With Private Data

Keep the input outside this repository. There is deliberately no default input
path and no data download. The schema and study-specific checks are documented
in [INPUT_SCHEMA.md](docs/INPUT_SCHEMA.md).

```powershell
.\.venv\Scripts\python.exe run_workflow.py --raw "C:\private\raw_spectra.csv" --output "C:\private\vnir_outputs" --n-jobs -1
```

Use `--models-only` to stop after fitting and pooled diagnostics. Use
`--postprocess-only` with the same private input and completed output folder to
produce statistics, SHAP and publication outputs later. `--resume` reuses
model checkpoints only if the input and scientific-core fingerprints match.
Final-delivery generation refuses to delete or replace an existing delivery:
choose a fresh output directory for a new complete run.

The full study-specific plan has 60 model configurations. It includes LOOCV
and nested grouped PLSR searches over 2,101 spectral features, so execution
can be expensive. `--n-jobs` controls the main model stage. The retained
capped-oasis PLSR function separately uses `n_jobs=-1`, exactly as in the final
analysis source.

## Code Map

| File | Purpose |
|---|---|
| `run_workflow.py` | Portable entry point and ordering of all stages |
| `code/workflow_core.py` | Data checks, fold assignments, models, pooled metrics, bootstrap |
| `code/pipeline_preprocessing.py` | Wavelength selection and per-spectrum continuum removal |
| `code/run_revised_workflow.py` | Model-stage checkpoints and diagnostics |
| `code/finalize_capped_oasis_blocking_table.py` | Final 1-4 component oasis-blocked PLSR |
| `code/build_sections_5_8.py` | Mixed effects, SHAP, table helpers, software version recording |
| `code/rebuild_submitted_style_figures.py` | Main Figures 1-5 and prediction diagnostics |
| `code/rebuild_submitted_supplementary_figures.py` | Submitted-style supplementary figures |
| `code/submitted_plot_style.py` | Bundled original supplementary plotting functions |
| `code/build_rf_blocking_comparison_figure.py` | Plot/site-blocked RF comparison |
| `code/build_validation_with_site_blocked_figure.py` | Validation comparison and both delta-R2 baselines |
| `code/improve_shap_figures.py` | Final SHAP styling |
| `code/build_final_for_manuscript.py` | Final PNG/PDF and CSV/Word delivery |

See [MODEL_SPECIFICATION.md](docs/MODEL_SPECIFICATION.md) for the exact model
settings and scientific definitions, and [SOURCE_MANIFEST.md](docs/SOURCE_MANIFEST.md)
for provenance and packaging-only adaptations.

## Generated Outputs Are Private

On an authorized local run, diagnostics include the cleaned working dataset,
sample identifiers, fold memberships and observed/predicted values. The final
delivery is generated under `OUTPUT/FINAL_FOR_MANUSCRIPT`. These files are
**not part of the public code release** and must not be uploaded inadvertently.
The supplied `.gitignore` excludes common data, result and binary formats, but
does not protect files uploaded manually through a website.

Released under the MIT licence (see `LICENSE`).

## Verification Scope

This code-only export is checked for syntax, imports, synthetic unit tests,
absence of data files and preservation of the scientific source functions.
The real-data workflow was not rerun to prepare this export. See
`docs/PACKAGE_VERIFICATION.md` for the completed packaging checks.

## Author

Wajdi Allagui — University of Vienna (CeMESS) and University of Tunis El Manar · [ORCID 0009-0002-4603-2473](https://orcid.org/0009-0002-4603-2473)

Code was developed with AI coding assistance. Study design, model choices and validation strategy are the author's own.
