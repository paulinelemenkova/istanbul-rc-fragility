# istanbul-rc-fragility

Code and simulation database for an explainable machine-learning surrogate of
nonlinear time-history analysis, predicting seismic damage and fragility for
reinforced-concrete frame archetypes representative of the Istanbul building
stock.

A parametric population of 60 two-dimensional fibre-section frames, spanning
code-conforming and deficient detailing, is analysed in OpenSeesPy under 30
recorded North Anatolian Fault ground motions scaled to six intensity levels
(10,800 analyses). A gradient-boosted surrogate trained with record-wise
cross-validation reproduces maximum inter-storey drift and EMS-98 damage state
at a fraction of the simulation cost; SHAP attribution, cross-checked against
pushover capacity analysis, identifies the design variables that govern
resistance; lognormal fragility parameters are estimated with PyMC.

## Citation

> Lemenkova, P., Zülfikar, A. C. (2026). Explainable machine-learning
> surrogates for the simulated seismic damage and fragility of reinforced
> concrete frame archetypes representative of the Istanbul building stock.
> *Discover Concrete and Cement*. [DOI to follow]

Software archive: [Zenodo DOI to follow]

## Installation

    git clone https://github.com/paulinelemenkova/istanbul-rc-fragility.git
    cd istanbul-rc-fragility
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Two non-Python dependencies:

**GMT 6** for the two shell scripts that build the maps
(https://www.generic-mapping-tools.org).

**Nimbus Sans** (URW base-35). The figure scripts refuse to export in
Matplotlib's DejaVu Sans fallback, so the font must be installed or every
`make_fig*.py` will stop with a clear error:

    brew install --cask font-urw-base35      # macOS
    sudo apt-get install fonts-urw-base35    # Debian / Ubuntu

## Third-party data you must download

Two tables are third-party and are **not** redistributed here. Download both
from the SMD-TR flatfile (NHERI DesignSafe PRJ-3950, v3,
https://doi.org/10.17603/ds2-f21x-s189) and place them in `data/external/`:

| File | Used by |
|---|---|
| `Metadata.csv` | `extract_smdtr_vs30.py`, `make_fig05.py` |
| `IM_RotD50.csv` | `make_fig05.py` |

Ground-motion waveforms are likewise not redistributed. `data/records_manifest.csv`
lists the 30 records of the suite with SHA-256 checksums; retrieve them from the
Engineering Strong-Motion database (https://esm-db.eu, DOI 10.13127/ESM.2) into
`data/records/` and verify against the checksums.

The GEBCO tile, GEM fault trace, and USGS/AFAD catalogue extract in
`data/external/` **are** included; see `NOTICE.md` for their licences.

## Reproduction order

`results/simulation_database.csv` is included, so the analysis and figure steps
can be run without repeating the 5.25 h campaign.

**1 — Simulation** (skip unless regenerating the database)

    python scripts/frame_population.py     # -> data/frame_population.csv (60 frames)
    python scripts/run_campaign.py         # -> results/simulation_database.csv
                                           #    10,800 analyses, ~5.25 h, 5 processes

`run_campaign.py` is resumable: kill it and rerun, and completed
`(fid, rid, level)` triples are skipped. It needs `data/records/` populated first.

**2 — Surrogate, interpretation and fragility**

    python scripts/oof_one.py xgb record    # -> results/derived/oof_xgb_record.npz
    python scripts/measure_cost.py          # computational-cost table
    python scripts/make_fig07.py            # ML performance  (reads results/derived/)
    python scripts/make_fig08.py            # SHAP attribution -> results/derived/shap_importance.csv
    python scripts/make_fig09.py            # fragility (PyMC) -> results/derived/fragility_fits.csv

`oof_one.py <model> <partition>` computes out-of-fold predictions for one
candidate model under one grouping; `model` is `linear | rf | xgb | mlp` and
`partition` is `record | frame`. Run all eight combinations to reproduce the
model-comparison table; `xgb record` is the configuration adopted in the paper.

**3 — Site data and figures**

    python scripts/extract_smdtr_vs30.py   # Metadata.csv -> data/AFAD_vs30.csv (51 stations)

    bash   scripts/GMT-01-Marmara-seis.sh   # Figure 1, GMT version
    bash   scripts/GMT-02-Istanbul-vs30.sh  # Figure 2, GMT version

    python scripts/make_fig01.py            # study area
    python scripts/make_fig02.py            # Vs30 and site class (needs AFAD_vs30.csv)
    python scripts/make_fig03.py            # workflow
    python scripts/make_fig04.py            # frame and fibre section
    python scripts/make_fig05.py            # ground-motion suite (needs the two SMD-TR tables)
    python scripts/make_fig06.py            # EDP clouds
    python scripts/make_fig10.py            # scenario response map
    python scripts/make_fig_ssi.py          # soil-structure interaction

All figures are written to `figures/` as vector PDF plus 600 dpi PNG.

## Repository layout

    scripts/            simulation, ML, fragility and figure code
      mpl_style.py      shared Matplotlib house style (imported by the figure scripts)
      check_overlap.py  label-collision audit used by figures 6-9
      paired_07.cpt     ColorBrewer Paired-7 palette for figure 3
    data/               frame population, ground-motion manifest, derived station table
    data/external/      third-party inputs (GEBCO, GEM, catalogue extract)
    data/records/       ground-motion waveforms (not tracked; fetched by the user)
    results/            simulation_database.csv
    results/derived/    out-of-fold predictions, SHAP importances, fragility fits
    figures/            generated figures (not tracked)
    docs/               column dictionaries

## Data

`results/simulation_database.csv` holds one row per frame–record–intensity
combination (10,800 rows). Column dictionary: `docs/columns.md`.

`data/frame_population.csv` holds the 60 archetypes with their eigenvalue-derived
fundamental periods. Reproducible from `frame_population.py` with the fixed seed
recorded in that file.

## Licence

Code: MIT (`LICENSE`). Data and figures: CC BY 4.0 (`LICENSE-DATA.md`).
Third-party inputs retain their own terms (`NOTICE.md`).

## Funding

Türkiye Bilimsel ve Teknolojik Araştırma Kurumu (TÜBİTAK), BIDEB 2221
Science Fellowships and Grant Programmes, reference number
B.14.2.TBT.0.06.01.02-220-859260.
