# Third-party data and software

This repository redistributes or depends on the following resources, each
governed by its own terms. They are **not** covered by this repository's
MIT or CC BY 4.0 licences.

## Data redistributed here

| Resource | File | Terms |
|---|---|---|
| GEBCO 2026 grid (regional subset) | `data/external/gebco_*.tif` | Freely available; attribution to GEBCO Compilation Group required |
| USGS / AFAD earthquake catalogues (FDSN) | `data/external/IEB_export.csv` | Public domain (USGS) / AFAD terms |
| GEM Global Active Faults | `data/external/naf_trace.txt` | CC BY-SA 4.0, GEM Foundation |

## Data referenced but not redistributed

Strong-motion waveforms are **not** included. Fetch them with
`scripts/download_records.py`, which resolves the entries in
`data/records_manifest.csv` against:

- Engineering Strong-Motion database (ORFEUS/INGV), DOI 10.13127/ESM.2
- AFAD–TADAS, https://tadas.afad.gov.tr
- SMD-TR flatfile, NHERI DesignSafe PRJ-3950 v3,
  DOI 10.17603/ds2-f21x-s189 (Open Data Commons Attribution licence)
- Global Human Settlement Layer, European Commission JRC

## Software

OpenSeesPy, OpenQuake hazardlib, scikit-learn, XGBoost, SHAP, PyMC, ArviZ,
ObsPy, NumPy, SciPy, pandas, GeoPandas, Matplotlib and GMT/PyGMT are used
under their respective open-source licences. See `requirements.txt`.
