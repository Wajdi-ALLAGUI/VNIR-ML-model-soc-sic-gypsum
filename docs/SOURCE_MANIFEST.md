# Source provenance

This package was prepared from the current revised workflow. No datasets or saved results were copied.
The two scientific core files are byte-identical copies. Other files have only the documented packaging changes.

| Package file | Original source | Original SHA-256 |
|---|---|---|
| `workflow_core.py` | `code/workflow_core.py` | `26f8b3c334602ac613fc79cda037455414f1a84e96a9df0d925118142ebd68aa` |
| `pipeline_preprocessing.py` | `code/pipeline_preprocessing.py` | `58e512ef79aad5b0429db77de305d2d25258b4c351de47ddd1eba3f6880031a4` |
| `run_revised_workflow.py` | `code/run_revised_workflow.py` | `d28cc21fc04b1957d926e84865286824131dddf8927cb8f043dcb28a66ae4e93` |
| `rebuild_submitted_style_figures.py` | `code/rebuild_submitted_style_figures.py` | `4f18e6ac0d5c78a3bcedb0e0a11574e900a505ea01af7f585149ba82e54758b2` |
| `rebuild_submitted_supplementary_figures.py` | `code/rebuild_submitted_supplementary_figures.py` | `d21160f492feed5612e9a266be427f2f1cf913f6cb4ed938279af2c1fda99b5f` |
| `build_rf_blocking_comparison_figure.py` | `code/build_rf_blocking_comparison_figure.py` | `61fe7741e3767dcaf1dc4c88669c24c228862d75c2af28f119fb9856cfb487c1` |
| `build_validation_with_site_blocked_figure.py` | `code/build_validation_with_site_blocked_figure.py` | `e456658fe8f548b6a3abf544d3a50581878eb1b4b84a46453785de5bb54c1865` |
| `improve_shap_figures.py` | `code/improve_shap_figures.py` | `eaa604d725ce8047b3114776febea3accab4fbba4bc522443ed9f409db1d945c` |
| `build_final_for_manuscript.py` | `code/build_final_for_manuscript.py` | `3c3009008525c377817a56a220e4dff741111e2926179384d0edb95b86cc7528` |
| `build_sections_5_8.py` | `code/build_sections_5_8.py (required functions only)` | `9ef923cea382070bdbc4d34db976d57399bc57d4e0850ff7c572803299909160` |
| `finalize_capped_oasis_blocking_table.py` | `code/finalize_capped_oasis_blocking_table.py (required functions only)` | `0dfa568181b7fcdb9a220974afcc1ba6d4b401b4d75ddd1639bac962fd723e33` |
| `submitted_plot_style.py` | `article_story_PLSR_fixed/code/build_article_story_outputs.py (plotting helpers only)` | `9d707ee62384e9326bf04df373fb9b9a87f449c01ecbfdb87a19eea38f7d0ab8` |

## Packaging changes

- Portable output-root and raw-input arguments replace local desktop paths.
- Required supplementary styling functions are bundled locally; historical loaders are excluded.
- Mixed-effects, SHAP and capped PLSR functions are retained; previous-manuscript comparison/report routines are excluded.
- The public runner has no dependency on a previous results file or submitted manuscript.
- Existing output directories are not recursively deleted by the package exporter.
- No estimator, preprocessing, split, metric or bootstrap formula was changed.
- The extracted supplementary style module includes its original ticker dependency (`matplotlib.ticker`).
- `run_model_stage` retains the original evaluation and pooled-summary calls, but replaces historical comparison gates with input/code fingerprint checks for safe checkpoint reuse.
- `run_workflow.py`, `package_paths.py`, documentation and synthetic tests are new distribution helpers, not new scientific methods.
- The CR matrix required by descriptive PCA is generated locally from the private raw input using the unchanged canonical CR transformer.
