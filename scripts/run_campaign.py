#!/usr/bin/env python3
"""
run_campaign.py - execute the nonlinear time-history campaign.

Design (reduced, agreed):
    60 frames x 30 records x 6 intensity levels = 10,800 analyses

Scaling follows Section 4.3: each record is scaled by a single amplitude factor
applied uniformly to the whole time series, the scaling variable being the
5 %-damped Sa(T1) of the frame being analysed, evaluated from that frame's
eigenvalue analysis. Six levels are spaced logarithmically over
Sa(T1) = 0.05-2.0 g. Scale factors are recorded so that records requiring an
unusually large factor can be flagged, as Section 4.3 promises.

Outcome rules follow Section 4.2: a run exceeding 10 % inter-storey drift is
terminated and labelled Collapse; a run that fails to converge after
subdivision to dt/64 is labelled Collapse if the last converged drift already
exceeds the 6 % Heavy threshold, otherwise it is discarded as a numerical
failure and listed with its frame and record identifiers.

The runner is resumable: results are appended to the output CSV after every
task, and on restart any (fid, rid, level) already present is skipped. Kill it
and rerun at will.

Usage
-----
    python3 run_campaign.py                       # uses the repository defaults
    python3 run_campaign.py --nproc 16
    python3 run_campaign.py --limit 30            # smoke test

Defaults are resolved relative to the repository root, so the script runs from
any working directory: records from <repo>/data/records/, frame population from
<repo>/data/frame_population.csv, output to
<repo>/results/simulation_database.csv.
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import esm_io as E
import frame_population as FP
import rc_frame_model as M

# Repository layout: this file lives in <repo>/scripts/. Resolving from
# __file__ keeps every default path machine-independent.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# 6 levels, log-spaced but weighted toward the upper half of the range so more
# of them fall near the damage-state transitions (particularly collapse) that
# the PyMC fragility fit of Section 4.7 needs to constrain both theta and beta.
# Reduced from 10 uniformly log-spaced levels for tractable runtime on a
# single 8-core/8GB machine; see the design note in frame_population.py.
SA_LEVELS = np.array([0.05, 0.15, 0.35, 0.65, 1.10, 2.00])
DS_EDGES = [0.005, 0.012, 0.030, 0.060]
DS_NAMES = ["None", "Light", "Moderate", "Heavy", "Collapse"]
FIELDS = ["fid", "rid", "level", "sa_target_g", "sa_record_g", "scale",
          "T1_s", "midr", "ds_index", "damage_state", "status", "soft", "deficient",
          "n_storey", "fc_MPa", "rho_long", "r_k", "mw", "rjb_km", "vs30",
          "pga_g", "runtime_s"]

_FRAMES = _RECORDS = None


def _init(frames_csv, rec_folder):
    """Per-worker initialisation: load the population and the records once."""
    global _FRAMES, _RECORDS
    sys.stderr = open(os.devnull, "w")
    _FRAMES = pd.read_csv(frames_csv).set_index("fid")
    _RECORDS = {r.rid: E.trim_significant_duration(r) for r in
                E.select_horizontal_pairs(E.load_folder(rec_folder))}


def _task(job):
    fid, rid, lev = job
    row = _FRAMES.loc[fid]
    rec = _RECORDS[rid]
    f = FP.to_frame(row.to_frame().T.assign(fid=fid).iloc[0])
    T1 = float(row.T1_s)
    sa_t = float(SA_LEVELS[lev])
    sa_r = M.spectral_acceleration(rec.acc, rec.dt, T1)
    scale = sa_t / sa_r if sa_r > 0 else np.nan
    t0 = time.time()
    if not np.isfinite(scale):
        res = dict(status="badscale", midr=np.nan, T1=T1)
    else:
        res = M.run_nltha(f, rec.acc, rec.dt, scale=scale)
    dsi = -1 if not np.isfinite(res["midr"]) else int(np.digitize(res["midr"], DS_EDGES))
    ds = "" if dsi < 0 else DS_NAMES[dsi]
    return dict(fid=fid, rid=rid, level=lev, sa_target_g=sa_t,
                sa_record_g=round(sa_r, 5), scale=round(float(scale), 4),
                T1_s=T1, midr=res["midr"], ds_index=dsi, damage_state=ds,
                status=res["status"], soft=bool(row.soft),
                deficient=bool(row.deficient), n_storey=int(row.n_storey),
                fc_MPa=round(row.fc / 1e6, 2), rho_long=row.rho_long,
                r_k=row.r_k, mw=rec.mw, rjb_km=rec.rjb, vs30=rec.vs30,
                pga_g=round(rec.pga, 4), runtime_s=round(time.time() - t0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(DATA / "records"),
                    help="folder of ESM ASCII records "
                         "(default: data/records; populate it with "
                         "download_records.py)")
    ap.add_argument("--frames", default=str(DATA / "frame_population.csv"))
    ap.add_argument("--out", default=str(RESULTS / "simulation_database.csv"))
    ap.add_argument("--nproc", type=int, default=min(5, max(1, os.cpu_count() - 1)),
                help="worker processes; default is capped at 5 to leave headroom "
                     "on memory-constrained laptops (e.g. 8GB M1 Air)")
    ap.add_argument("--limit", type=int, default=0, help="run only N tasks")
    a = ap.parse_args()

    if not os.path.exists(a.frames):
        sys.exit(f"frame population not found: {a.frames}\n"
                 f"Run:  python3 frame_population.py")
    frames = pd.read_csv(a.frames)
    recs = E.select_horizontal_pairs(E.load_folder(a.records))
    if not recs:
        sys.exit(f"no readable ESM records in {a.records}")
    print(f"{len(frames)} frames x {len(recs)} records x {len(SA_LEVELS)} levels "
          f"= {len(frames)*len(recs)*len(SA_LEVELS):,} analyses")
    print(f"Sa(T1) levels (g): {', '.join(f'{s:g}' for s in SA_LEVELS)}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        d = pd.read_csv(a.out, keep_default_na=False)
        done = set(zip(d.fid, d.rid, d.level))
        print(f"resuming: {len(done):,} analyses already complete")
    else:
        with open(a.out, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    jobs = [(int(fid), r.rid, lev)
            for fid in frames.fid for r in recs for lev in range(len(SA_LEVELS))
            if (int(fid), r.rid, lev) not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs):,} analyses to run on {a.nproc} processes\n")

    t0, n = time.time(), 0
    with Pool(a.nproc, initializer=_init, initargs=(a.frames, a.records)) as pool:
        with open(a.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS)
            for res in pool.imap_unordered(_task, jobs, chunksize=4):
                w.writerow(res); fh.flush()
                n += 1
                if n % 50 == 0 or n == len(jobs):
                    el = time.time() - t0
                    rate = n / el
                    eta = (len(jobs) - n) / rate / 3600 if rate else 0
                    print(f"  {n:6,}/{len(jobs):,}  {rate:5.2f} an/s  "
                          f"ETA {eta:5.2f} h", flush=True)
    print(f"\ndone in {(time.time()-t0)/3600:.2f} h -> {a.out}")


if __name__ == "__main__":
    main()
