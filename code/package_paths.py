"""Portable paths shared by the code-only distribution."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def output_root() -> Path:
    return Path(os.environ.get("VNIR_OUTPUT_ROOT", PACKAGE_ROOT / "outputs")).expanduser().resolve()


def private_raw_path() -> Path:
    value = os.environ.get("VNIR_PRIVATE_RAW_INPUT")
    if not value:
        raise RuntimeError("Run run_workflow.py with --raw pointing to your private CSV.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Private raw input does not exist: {path}")
    return path
