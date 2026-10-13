# Code-Only Package Verification

Verification date: 2026-10-13. Reference runtime: Python 3.14.0 on Windows.

| Check | Result |
|---|---|
| `python run_workflow.py --help` | Passed; portable CLI available |
| `python run_workflow.py --check` | Passed; all 12 required distribution versions available |
| `python -m unittest discover -s tests -v` | 14 tests passed, synthetic data only |
| Syntax and statically referenced module dependencies | All 15 Python files passed |
| Scientific-core preservation | `workflow_core.py` and `pipeline_preprocessing.py` byte-identical to original sources |
| Extracted source definitions | 133 function/class definitions unchanged |
| Existing definitions adapted for packaging | 4, listed below |
| Original sources | All 12 source hashes still match the export manifest |
| Export contents | Python source, Markdown documentation, requirements and `.gitignore` only |

The four existing definitions adapted for portability/orchestration are
`run_model_stage`, supplementary `plot_mean_spectra`, supplementary `main`,
and final exporter `reset_output_directory`. Changes concern output paths,
bundled plotting helpers, historical comparison gates, safe resumption and
refusal to delete an existing delivery. No model-building or metric function
was altered.

Synthetic tests cover pooled metric definitions, LOOCV pooling, reproducible
paired bootstrap, CR hull/endpoints/interpolation/range/sample independence,
MSC training-reference isolation, RF settings/native multi-output construction,
small synthetic direct/multi RF fits, nested grouped PLSR, the 60-configuration
plan, the final 1-4 component cap, all-module imports, and rendering of two
retained supplementary designs. Seaborn produced pending-deprecation warnings
about a Matplotlib boxplot argument; those warnings did not fail the tests.

No real experimental data were loaded and no real-data model fitting or
manuscript regeneration was performed for this export. Successful synthetic
tests are not a claim of a complete end-to-end rerun on the study dataset.
The original analysis code and result folders were not modified.

The release archive excludes bytecode caches, all datasets, all generated
outputs and all fitted-model binaries. `.gitignore` reduces accidental Git
tracking, but manual website uploads still require checking the selected
files. The package does not upload anything by itself.
