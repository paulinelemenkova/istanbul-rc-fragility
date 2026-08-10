#!/usr/bin/env python3
"""
frame_population.py - generate the parametric frame population.

Reduced design agreed for the campaign:
    60 frames x 30 records x 6 intensity levels = 10,800 analyses

The 60 frames are drawn by Latin-hypercube sampling within four archetype
classes, crossing two configurations with two detailing classes:

    regular  / code-conforming     15 frames
    regular  / deficient           15
    soft-storey / code-conforming  15
    soft-storey / deficient        15

Parameter ranges follow Table 2 of the manuscript (Istanbul low/mid-rise RC
stock) and the detailing separation of Table 3:

    storeys        2-8
    bays           3 (fixed)
    fc             25-35 MPa code-conforming | 16-25 MPa deficient
    fy             420 MPa   | 220 MPa (plain bars, pre-1997 stock)
    rho_long       0.010-0.020 | 0.008-0.014
    confinement    1.30 (135 deg hooks, close spacing) | 1.10 (90 deg hooks)
    r_k            1.0 regular | 0.30-0.60 soft-storey (Eq. 4)
    xi             0.05
    column size    scaled with storey count

Each frame's fundamental period T1 is then computed by a real eigenvalue
analysis, so the released table contains simulated, not assumed, periods.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import rc_frame_model as M

N_PER_CLASS = 15
SEED = 20260808


def _lhs(n, d, rng):
    """Latin-hypercube sample in the unit cube."""
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return u


def build_population(n_per_class=N_PER_CLASS, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    fid = 0
    for soft in (False, True):
        for deficient in (False, True):
            u = _lhs(n_per_class, 5, rng)
            for k in range(n_per_class):
                ns = int(2 + np.floor(u[k, 0] * 7))              # 2..8
                fc = ((25 + 10 * u[k, 1]) if not deficient else (16 + 9 * u[k, 1])) * 1e6
                rho = (0.010 + 0.010 * u[k, 2]) if not deficient else (0.008 + 0.006 * u[k, 2])
                rk = 1.0 if not soft else 0.30 + 0.30 * u[k, 3]
                col = 0.35 + 0.05 * np.floor(ns / 2) + 0.05 * u[k, 4]   # 0.35-0.60 m
                rows.append(dict(
                    fid=fid, soft=soft, deficient=deficient, n_storey=ns, n_bay=3,
                    fc=fc, fy=420e6 if not deficient else 220e6,
                    rho_long=rho, conf_ratio=1.30 if not deficient else 1.10,
                    r_k=rk, col_b=round(col, 3), col_h=round(col, 3),
                    beam_b=0.30, beam_h=0.50, xi=0.05))
                fid += 1
    return pd.DataFrame(rows)


def to_frame(row):
    """DataFrame row -> Frame object."""
    return M.Frame(n_storey=int(row.n_storey), n_bay=int(row.n_bay),
                   soft=bool(row.soft), r_k=float(row.r_k),
                   deficient=bool(row.deficient), fc=float(row.fc),
                   fy=float(row.fy), col_b=float(row.col_b), col_h=float(row.col_h),
                   beam_b=float(row.beam_b), beam_h=float(row.beam_h),
                   rho_long=float(row.rho_long), conf_ratio=float(row.conf_ratio),
                   xi=float(row.xi), fid=int(row.fid))


def add_periods(df, verbose=True):
    """Run a real eigenvalue analysis on every frame and record T1, T3."""
    T1, T3 = [], []
    for _, row in df.iterrows():
        f = to_frame(row)
        M.build(f)
        T = M.eigen_T(3)
        T1.append(float(T[0])); T3.append(float(T[2]))
    df = df.copy()
    df["T1_s"] = np.round(T1, 4)
    df["T3_s"] = np.round(T3, 4)
    if verbose:
        print(f"T1 range {df.T1_s.min():.3f}-{df.T1_s.max():.3f} s, "
              f"median {df.T1_s.median():.3f} s")
    return df


# Repository layout: this file lives in <repo>/scripts/, inputs and outputs
# live in <repo>/data/ and <repo>/results/. Resolving from __file__ keeps the
# campaign runnable from any working directory and on any machine.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

if __name__ == "__main__":
    import os
    import sys
    DATA.mkdir(parents=True, exist_ok=True)
    out_csv = DATA / "frame_population.csv"
    sys.stderr = open(os.devnull, "w")
    df = add_periods(build_population(), verbose=False)
    df.to_csv(out_csv, index=False)
    sys.stderr = sys.__stderr__
    print(f"{len(df)} frames written to {out_csv}")
    print(f"  storeys      {df.n_storey.min()}-{df.n_storey.max()}")
    print(f"  fc           {df.fc.min()/1e6:.1f}-{df.fc.max()/1e6:.1f} MPa")
    print(f"  rho_long     {df.rho_long.min():.4f}-{df.rho_long.max():.4f}")
    print(f"  T1           {df.T1_s.min():.3f}-{df.T1_s.max():.3f} s "
          f"(median {df.T1_s.median():.3f})")
    for (s, d), g in df.groupby(["soft", "deficient"]):
        lab = f"{'soft' if s else 'regular':8s}/{'deficient' if d else 'code-conf':10s}"
        print(f"  {lab}  n={len(g):3d}  T1 {g.T1_s.min():.2f}-{g.T1_s.max():.2f} s")
