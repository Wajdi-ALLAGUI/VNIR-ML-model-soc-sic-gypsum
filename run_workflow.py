"""Run the revised VNIR workflow with an explicitly supplied private input."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "code"))


def check_dependencies() -> None:
    from importlib.metadata import version

    print(f"Python {sys.version.split()[0]}")
    for name in (
        "pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "seaborn",
        "statsmodels", "shap", "openpyxl", "python-docx", "Pillow", "joblib",
    ):
        print(f"{name} {version(name)}")
    importlib.import_module("workflow_core")
    importlib.import_module("pipeline_preprocessing")
    print("DEPENDENCIES_OK (no private data loaded; no models run)")


def postprocess(raw_path: Path, output: Path) -> None:
    import pandas as pd
    from sklearn.impute import SimpleImputer

    from build_sections_5_8 import (
        compute_shap_top5, configure_style, fit_mixed_effects, load_sources,
        make_paths, package_versions, preprocess_scope_table,
    )
    from finalize_capped_oasis_blocking_table import fit_capped_oasis_plsr
    from pipeline_preprocessing import UpperHullContinuumRemoval
    from workflow_core import assert_summary_metrics, summarize_validation_predictions

    paths = make_paths(output)
    sources = load_sources(paths, raw_path)
    configure_style()
    preprocess_scope_table(paths)
    fit_mixed_effects(sources["raw"], paths)
    compute_shap_top5(sources, paths)
    package_versions(paths)

    data = sources["analysis"]
    assignments = pd.read_csv(paths.diagnostics / "outer_fold_assignments.csv")
    capped_predictions, capped_components = fit_capped_oasis_plsr(data, assignments)
    capped_summary = summarize_validation_predictions(capped_predictions)
    assert_summary_metrics(capped_summary)
    capped_folder = output / "revision_submission" / "diagnostics" / "capped_oasis_plsr"
    capped_folder.mkdir(parents=True, exist_ok=True)
    capped_predictions.to_csv(capped_folder / "oasis_blocked_plsr_capped4_predictions.csv", index=False)
    capped_components.to_csv(capped_folder / "oasis_blocked_plsr_capped4_components.csv", index=False)
    capped_summary.to_csv(capped_folder / "oasis_blocked_plsr_capped4_summary.csv", index=False)

    # PCA needs the same in-pipeline CR, not an external precomputed CR input.
    X = SimpleImputer(strategy="median").fit_transform(data.X)
    cr = UpperHullContinuumRemoval(wavelengths=data.wavelengths).fit_transform(X)
    spectral = pd.DataFrame(cr, columns=[f"X{int(w)}" for w in data.wavelengths])
    frame = pd.concat([data.working.reset_index(drop=True).drop(
        columns=[c for c in data.working if c.startswith("X") and c[1:].isdigit()]
    ), spectral], axis=1)
    verification = output / "verification"
    verification.mkdir(parents=True, exist_ok=True)
    frame.to_csv(verification / "computed_continuum_removal_400_2500.csv", index=False)

    for module_name in (
        "rebuild_submitted_style_figures",
        "rebuild_submitted_supplementary_figures",
        "build_rf_blocking_comparison_figure",
        "build_validation_with_site_blocked_figure",
        "improve_shap_figures",
        "build_final_for_manuscript",
    ):
        print(f"POSTPROCESS {module_name}", flush=True)
        importlib.import_module(module_name).main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, help="Private raw spectral CSV; never uploaded by this code.")
    parser.add_argument("--output", type=Path, default=PACKAGE / "outputs")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--resume", action="store_true", help="Resume matching model checkpoints.")
    parser.add_argument("--models-only", action="store_true", help="Skip statistics, SHAP, figures and final tables.")
    parser.add_argument("--postprocess-only", action="store_true", help="Use completed diagnostics in --output.")
    parser.add_argument("--check", action="store_true", help="Check dependencies without accessing data.")
    args = parser.parse_args()
    if args.check:
        check_dependencies()
        return
    if args.raw is None:
        parser.error("--raw must explicitly name your private CSV; no dataset is supplied.")
    args.raw = args.raw.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.raw.is_file():
        parser.error(f"Private input does not exist: {args.raw}")
    if args.models_only and args.postprocess_only:
        parser.error("--models-only and --postprocess-only cannot be combined.")
    if args.output == PACKAGE or args.output == PACKAGE / "code":
        parser.error("Use a separate output directory, not the source-code directory.")
    if not args.models_only and (args.output / "FINAL_FOR_MANUSCRIPT").exists():
        parser.error("Final delivery already exists; choose a new output directory.")
    os.environ["VNIR_OUTPUT_ROOT"] = str(args.output)
    os.environ["VNIR_PRIVATE_RAW_INPUT"] = str(args.raw)
    if not args.postprocess_only:
        from run_revised_workflow import run_model_stage
        run_model_stage(args)
    if not args.models_only:
        postprocess(args.raw, args.output)
    print(f"WORKFLOW_COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
