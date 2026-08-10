#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig06_edp_clouds -- engineering demand parameters from the OpenSeesPy simulation
database: maximum inter-storey drift ratio (MIDR) against the scaled spectral
acceleration Sa(T1), for the regular and soft-storey archetype classes.

Built directly from simulation_database.csv (10 800 nonlinear time-history
analyses = 60 frames x 30 records x 6 intensity levels).  The database is a
stripe design at six discrete Sa(T1) levels rather than a continuous IDA, so the
"curves" are traces through those six levels:

  * shaded band  -- 16th-84th percentile of all analyses of that class at each
                    level (5th-95th as a fainter outer band);
  * thin traces  -- one per ground-motion record: the median over the 30 frames
                    of that class, i.e. 30 traces per class;
  * thick line   -- the class median over all 900 analyses at each level;
  * dashed rules -- EMS-98 damage-state drift thresholds;
  * dotted rule  -- the 10% drift cap at which an analysis is stopped and
                    labelled Collapse.

House style: scientific-plotting skill (Nimbus Sans, major+minor ticks on both
axes, thin grid under the data, Okabe-Ito categorical colours with distinct
marker shapes, vector PDF + 600 dpi PNG).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402
from matplotlib.patches import Patch                     # noqa: E402

from mpl_style import set_rc, assert_not_dejavu          # noqa: E402
from check_overlap import covers_data, check_figure      # noqa: E402

# ---------------------------------------------------------------- paths ----
# This file lives in <repo>/scripts/. Every path below is resolved from
# __file__, so the script runs from any working directory and on any machine.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXTERNAL = DATA / "external"
RESULTS = ROOT / "results"
DERIVED = RESULTS / "derived"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

CSV = str(RESULTS / "simulation_database.csv")
NAME = "fig06_edp_clouds"
DCAP = 0.10                                   # drift cap = Collapse
DS = [("Light", 0.005), ("Moderate", 0.012), ("Heavy", 0.030), ("Collapse", 0.060)]

# Okabe-Ito: blue = regular (code-conforming), vermillion = soft-storey.
C_REG, C_SOFT = "#0072B2", "#D55E00"
M_REG, M_SOFT = "o", "s"

# ------------------------------------------------------------------ data ----
d = pd.read_csv(CSV, keep_default_na=False, na_values=[""])
LEVELS = np.sort(d["sa_target_g"].unique())


def stats(sub):
    """Class median and percentile envelopes over the six intensity levels."""
    g = sub.groupby("sa_target_g")["midr"]
    q = g.quantile([0.05, 0.16, 0.50, 0.84, 0.95]).unstack()
    return q.reindex(LEVELS)


def record_traces(sub):
    """One trace per record: the median over the frames of that class."""
    p = sub.pivot_table(index="sa_target_g", columns="rid", values="midr",
                        aggfunc="median").reindex(LEVELS)
    return p


reg, soft = d[~d["soft"]], d[d["soft"]]
q_reg, q_soft = stats(reg), stats(soft)
t_reg, t_soft = record_traces(reg), record_traces(soft)
print(f"n = {len(d)}   regular {len(reg)}   soft {len(soft)}   "
      f"records/class {t_reg.shape[1]}")

# ----------------------------------------------------------------- style ----
set_rc()
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8, "savefig.dpi": 600,
})

W, H = 6.90, 4.30
LM, RM, BM, TM = 0.66, 0.10, 0.62, 0.22
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([LM / W, BM / H, (W - LM - RM) / W, (H - BM - TM) / H])

XLO, XHI = 0.042, 2.95          # head-room on the right for the threshold labels
YLO, YHI = 2.2e-4, 0.145

# ---- percentile envelopes (context, drawn first) ---------------------------
for q, c in ((q_reg, C_REG), (q_soft, C_SOFT)):
    ax.fill_between(LEVELS, q[0.05], q[0.95], color=c, alpha=0.09, lw=0, zorder=2)
    ax.fill_between(LEVELS, q[0.16], q[0.84], color=c, alpha=0.22, lw=0, zorder=3)

# ---- per-record median traces (context ink, below the class medians) -------
for t, c in ((t_reg, C_REG), (t_soft, C_SOFT)):
    for rid in t.columns:
        ax.plot(LEVELS, t[rid].to_numpy(), color=c, lw=0.6, alpha=0.35, zorder=4)

# ---- class medians (the result: 2.0 pt, with marker shapes) ----------------
ax.plot(LEVELS, q_reg[0.50], color=C_REG, lw=2.0, marker=M_REG, ms=5.0,
        mec="white", mew=0.6, zorder=7)
ax.plot(LEVELS, q_soft[0.50], color=C_SOFT, lw=2.0, marker=M_SOFT, ms=4.6,
        mec="white", mew=0.6, zorder=7)

# ---- damage-state thresholds and the drift cap ----------------------------
for name, y in DS:
    ax.axhline(y, color="0.55", lw=0.7, ls=(0, (5, 3)), zorder=5)
    ax.text(XHI / 1.03, y * 1.13, name, fontsize=8.5, color="0.25",
            va="bottom", ha="right", zorder=8)
ax.axhline(DCAP, color="0.35", lw=0.7, ls=(0, (1, 2)), zorder=5)
ax.text(XHI / 1.03, DCAP * 1.13, "drift cap", fontsize=8.5, color="0.25",
        va="bottom", ha="right", zorder=8)
ax.axvline(2.0, color="0.85", lw=0.5, zorder=1)   # edge of the sampled range

# ---- axes ------------------------------------------------------------------
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(XLO, XHI); ax.set_ylim(YLO, YHI)
ax.set_xlabel(r"Scaled spectral acceleration $S_a(T_1)$ (g)")
ax.set_ylabel("Maximum inter-storey drift ratio, MIDR (–)")
for axis in (ax.xaxis, ax.yaxis):
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1,
                                      numticks=14))
    axis.set_minor_formatter(NullFormatter())
ax.tick_params(which="both", top=True, right=True, direction="in")
ax.tick_params(which="major", length=4.5)
ax.tick_params(which="minor", length=2.5)
ax.grid(which="major", color="0.85", lw=0.5)
ax.set_axisbelow(True)

# ---- legend, in the empty upper-left (low intensity, high drift) ----------
handles = [
    Line2D([0], [0], color=C_REG, lw=2.0, marker=M_REG, ms=5.0,
           label="Regular, class median"),
    Line2D([0], [0], color=C_SOFT, lw=2.0, marker=M_SOFT, ms=4.6,
           label="Soft-storey, class median"),
    Line2D([0], [0], color="0.45", lw=0.8, label="per-record median (30 per class)"),
    Patch(facecolor="0.45", alpha=0.22, label="16th–84th percentile"),
    Patch(facecolor="0.45", alpha=0.09, label="5th–95th percentile"),
]
leg = ax.legend(handles=handles, loc="upper left", fontsize=8.5,
                handlelength=1.8, borderpad=0.35, labelspacing=0.30,
                framealpha=0.95, edgecolor="0.7", fancybox=False)
leg.get_frame().set_linewidth(0.5)

# ------------------------------------------------------ checks + export -----
fig.canvas.draw()
r = fig.canvas.get_renderer()
print("legend covers_data hits:", covers_data(leg.get_window_extent(r), ax))
check_figure(fig)
assert_not_dejavu(fig)
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"{NAME}.{ext}", bbox_inches="tight")

# ---- the numbers the manuscript quotes ------------------------------------
ratio = (q_soft[0.50] / q_reg[0.50])
print("\nclass median MIDR (%):")
print(pd.DataFrame({"Sa_g": LEVELS, "regular": q_reg[0.50].values * 100,
                    "soft": q_soft[0.50].values * 100,
                    "ratio": ratio.values}).round(3).to_string(index=False))
for lbl, q in (("regular", q_reg), ("soft", q_soft)):
    x, y = np.log10(LEVELS), np.log10(q[0.50].to_numpy())
    for thr in (0.030, 0.060):
        if y.max() < np.log10(thr):
            print(f"{lbl:8s} median never reaches {thr:.3f} within the sampled range")
        else:
            i = int(np.argmax(y >= np.log10(thr)))
            sa = 10 ** np.interp(np.log10(thr), [y[i - 1], y[i]], [x[i - 1], x[i]])
            print(f"{lbl:8s} median reaches {thr:.3f} at Sa(T1) = {sa:.2f} g")
coll = d.assign(c=d["ds_index"] == 4).pivot_table(
    index="sa_target_g", columns="soft", values="c")
print("\ncollapse fraction:\n", coll.round(3).to_string())
