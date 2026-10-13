# Private Input Schema

This document describes column names and study design, not measured values.
No experimental records are included in this repository.

One raw reflectance CSV is supplied explicitly with `--raw`. The modelling
reader enforces the original study design: 216 unique samples, 18 plots,
6 oasis systems and 3 approved study areas. It is not a general-purpose
pipeline for arbitrary sampling designs without adapting those checks.

| Exact CSV header | Meaning / requirements |
|---|---|
| `identifiant` | Unique private sample identifier |
| `sites` | Recorded plot label, ending in a position number |
| `OasisType` | Recorded management code: 1 modern, 2 traditional |
| `Slope_Category` | Recorded topography code: 1 upslope, 2 midslope, 3 downslope |
| `Depth  ` | Depth in cm; original header has TWO trailing spaces |
| `SOC %` | Soil organic carbon, percent |
| `SIC %` | Soil inorganic carbon, percent |
| `Gypsum %` | Gypsum, percent |
| `X400` ... `X2500` | Raw reflectance, contiguous 1 nm wavelength columns |
| `Sand (%)`, `Silt (%)`, `Clay (%)` | Required for supplementary property plots |
| `pH ((1:5))`, `EC (dS/m)` | Required for supplementary correlation plots |

Bands outside 400-2500 nm may be present, but do not enter the models. The
core modelling reader detects a depth header ignoring surrounding spaces;
retained statistics/plotting routines use the original exact `Depth  ` header.
No target values may be missing/non-finite. Median feature imputation is
performed inside fitting; individual models may reject invalid spectra.

Plot labels are preserved, not translated or spelling-normalized. Oasis
stems are derived by removing trailing digits only. The approved design
mapping in the source is:

| Oasis stem | Added `Study_Area` |
|---|---|
| Bouhlel | Deguache |
| Degache | Deguache |
| Lahwar | Tozeur |
| Tozeur | Tozeur |
| Machtla | Nefta |
| Nefta | Nefta |

Management is read from `OasisType`, never inferred from these names.
Only `Study_Area` is added to the preserved raw working dataframe; other
labels used in diagnostics are derived metadata.

No precomputed continuum-removed file is an analytical input. Do not upload
the private CSV or the sample-level diagnostics generated from it.
