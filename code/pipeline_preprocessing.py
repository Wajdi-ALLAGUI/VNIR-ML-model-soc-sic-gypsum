"""Canonical spectral preprocessing utilities for the revised VNIR workflow.

The modelling workflow has one input: the raw spectral table. All scenarios
first use the same inclusive 400-2500 nm range. Continuum removal is computed
from those raw bands at runtime; no precomputed continuum-removed table is an
input to this module.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


WAVELENGTH_MIN_NM = 400.0
WAVELENGTH_MAX_NM = 2500.0
EXPECTED_BAND_COUNT = 2101


def wavelength_from_column(column: object) -> float | None:
    """Return the numeric wavelength represented by an X<wavelength> column."""
    match = re.fullmatch(r"X(\d+(?:\.\d+)?)", str(column).strip())
    return float(match.group(1)) if match else None


def spectral_columns(frame: pd.DataFrame) -> list[str]:
    """Return spectral columns ordered by increasing wavelength."""
    pairs = [
        (column, wavelength_from_column(column))
        for column in frame.columns
        if wavelength_from_column(column) is not None
    ]
    return [str(column) for column, _ in sorted(pairs, key=lambda item: item[1])]


def common_range_columns(frame: pd.DataFrame) -> list[str]:
    """Select and validate the common inclusive 400-2500 nm spectral range."""
    selected = [
        column
        for column in spectral_columns(frame)
        if WAVELENGTH_MIN_NM
        <= float(wavelength_from_column(column))
        <= WAVELENGTH_MAX_NM
    ]
    wavelengths = np.asarray(
        [float(wavelength_from_column(column)) for column in selected], dtype=float
    )
    if selected == []:
        raise ValueError("No spectral bands were found in the 400-2500 nm range.")
    if len(selected) != EXPECTED_BAND_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_BAND_COUNT} bands from 400 through 2500 nm, "
            f"but found {len(selected)}."
        )
    expected = np.arange(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM + 1.0, 1.0)
    if not np.array_equal(wavelengths, expected):
        raise ValueError("Spectral bands must be contiguous one-nanometre values from 400 to 2500 nm.")
    return selected


def upper_hull_indices(wavelengths: np.ndarray, reflectance: np.ndarray) -> np.ndarray:
    """Find the upper convex-hull vertices with a monotone-chain algorithm.

    The first and last wavelengths are always retained, so the continuum is
    anchored at both endpoints of the fitted range. Interior collinear points
    are removed. The retained chain is the least concave majorant of the
    spectrum over the supplied wavelength interval.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    reflectance = np.asarray(reflectance, dtype=float)
    if wavelengths.ndim != 1 or reflectance.ndim != 1:
        raise ValueError("Wavelength and reflectance inputs must both be one-dimensional.")
    if wavelengths.size != reflectance.size:
        raise ValueError("Wavelength and reflectance vectors must have the same length.")
    if wavelengths.size < 2:
        raise ValueError("At least two wavelength points are required.")
    if np.any(~np.isfinite(wavelengths)) or np.any(~np.isfinite(reflectance)):
        raise ValueError(
            "Continuum removal requires finite values. Fold-fitted imputation must run first."
        )
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("Wavelengths must be strictly increasing with no duplicates.")

    hull: list[int] = []
    for current in range(wavelengths.size):
        while len(hull) >= 2:
            first = hull[-2]
            second = hull[-1]
            cross_product = (
                (wavelengths[second] - wavelengths[first])
                * (reflectance[current] - reflectance[first])
                - (reflectance[second] - reflectance[first])
                * (wavelengths[current] - wavelengths[first])
            )
            if cross_product < 0.0:
                break
            hull.pop()
        hull.append(current)

    indices = np.asarray(hull, dtype=int)
    if indices[0] != 0 or indices[-1] != wavelengths.size - 1:
        raise AssertionError("The upper hull is not anchored at both range endpoints.")
    return indices


def continuum_remove_spectrum(
    wavelengths: np.ndarray, reflectance: np.ndarray
) -> np.ndarray:
    """Divide one spectrum by its endpoint-anchored, piecewise-linear upper hull."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    reflectance = np.asarray(reflectance, dtype=float)
    hull_indices = upper_hull_indices(wavelengths, reflectance)
    continuum = np.interp(
        wavelengths,
        wavelengths[hull_indices],
        reflectance[hull_indices],
    )
    if np.any(~np.isfinite(continuum)) or np.any(continuum <= np.finfo(float).eps):
        raise ValueError("The fitted continuum contains a non-finite or non-positive value.")
    return reflectance / continuum


class UpperHullContinuumRemoval(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer for independent per-spectrum CR.

    ``fit`` only validates dimensions and stores no population estimate.
    ``transform`` computes a new upper hull independently for every spectrum.
    This makes the operation sample-wise and prevents information transfer
    between calibration and validation samples.
    """

    def __init__(self, wavelengths: Sequence[float]):
        self.wavelengths = wavelengths

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "UpperHullContinuumRemoval":
        matrix = check_array(X, dtype=float, ensure_all_finite=True)
        wavelengths = np.asarray(self.wavelengths, dtype=float)
        if matrix.shape[1] != wavelengths.size:
            raise ValueError(
                f"X has {matrix.shape[1]} features, but {wavelengths.size} wavelengths were supplied."
            )
        if wavelengths[0] != WAVELENGTH_MIN_NM or wavelengths[-1] != WAVELENGTH_MAX_NM:
            raise ValueError("Continuum removal must be fitted over 400-2500 nm inclusive.")
        if not np.array_equal(
            wavelengths,
            np.arange(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM + 1.0, 1.0),
        ):
            raise ValueError("Continuum-removal wavelengths must be contiguous at 1 nm spacing.")
        self.wavelengths_ = wavelengths
        self.n_features_in_ = matrix.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["wavelengths_", "n_features_in_"])
        matrix = check_array(X, dtype=float, ensure_all_finite=True)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features; fitted input had {self.n_features_in_}."
            )
        return np.vstack(
            [
                continuum_remove_spectrum(self.wavelengths_, spectrum)
                for spectrum in matrix
            ]
        )

