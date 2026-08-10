#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig07_ml_performance -- surrogate-model performance for the interpretable-ML
RC-frame fragility study (Marmara / Istanbul).

Both panels are built from the SAME out-of-fold predictions produced by
oof_one.py (XGBoost, GroupKFold(5) grouped on the ground-motion record, so each
of the 10 800 nonlinear time-history analyses contributes exactly one held-out
prediction):

  (a) predicted vs. simulated maximum inter-storey drift ratio (MIDR) on
      logarithmic axes, drawn as a two-dimensional density with a random
      subsample overplotted as points, with the 1:1 line and a factor-of-two
      band; R^2, RMSE and MAE are computed on log10(MIDR).
  (b) row-normalised confusion matrix over the five EMS-98 damage states, with
      per-cell counts and row percentages; accuracy, macro-F1 and one-vs-rest
      ROC-AUC are reported beneath the panel.

House style: scientific-plotting skill -- Nimbus Sans, major+minor ticks on
both axes, thin grid under the data, one colour-blind- and greyscale-safe
sequential palette, double-column width, vector PDF + 600 dpi PNG.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import LogLocator, NullFormatter

from mpl_style import set_rc, assert_not_dejavu          # noqa: E402
from check_overlap import covers_data, check_figure       # noqa: E402

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

OOF = str(DERIVED / "oof_xgboost.csv")
META = str(DERIVED / "fig07_meta.json")
NAME = "fig07_ml_performance"
DS_NAMES = ["None", "Light", "Moderate", "Heavy", "Collapse"]
K = len(DS_NAMES)
SUBSAMPLE, SEED = 600, 7

# Perceptually uniform sequential palettes, one per panel: magma for the
# density in (a), plasma for the classification rates in (b).  Both are
# REVERSED, so that high values are dark on the white page rather than pale
# yellow, and both are truncated at the extreme light end so that the lowest
# non-zero level still prints.  Reversal also keeps the ink proportional to the
# quantity, and both maps remain monotonic in lightness, hence greyscale- and
# colour-vision-deficiency-safe.
def _trunc(name, lo=0.06, hi=0.97, n=256):
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_{lo}_{hi}", base(np.linspace(lo, hi, n)))


CMAP_A = plt.get_cmap("jet")             # (a): analyses per hexagon
CMAP_B = _trunc("plasma_r")              # (b): row-normalised fraction
CMAP_B.set_bad("white")           # empty confusion-matrix cells stay white


def _ink(cmap, value, vmin=0.0, vmax=1.0):
    """Black or white label, whichever contrasts with the cell colour."""
    r, g, b, _ = cmap((value - vmin) / (vmax - vmin))
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "0.10"

# ------------------------------------------------------------------ data ----
d = pd.read_csv(OOF)
meta = json.load(open(META))
y_true = d["midr_sim"].to_numpy(float)
y_pred = d["midr_pred"].to_numpy(float)
ti, pi = d["ds_true"].to_numpy(int), d["ds_pred"].to_numpy(int)

lt, lp = np.log10(y_true), np.log10(y_pred)
R2 = 1.0 - np.sum((lt - lp) ** 2) / np.sum((lt - lt.mean()) ** 2)
RMSE = float(np.sqrt(np.mean((lt - lp) ** 2)))            # dex
MAE = float(np.mean(np.abs(lt - lp)))                     # dex
within2 = float(np.mean(np.abs(lt - lp) <= np.log10(2.0)))

CM = np.zeros((K, K), int)
for a, b in zip(ti, pi):
    CM[a, b] += 1
acc = np.trace(CM) / CM.sum()
f1s = []
for k in range(K):
    tp = CM[k, k]
    prec = tp / CM[:, k].sum() if CM[:, k].sum() else 0.0
    rec = tp / CM[k, :].sum() if CM[k, :].sum() else 0.0
    f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
macroF1 = float(np.mean(f1s))
AUC = meta["record_wise"]["AUC"]

# ----------------------------------------------------------------- style ----
set_rc()
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8, "savefig.dpi": 600,
})

# Explicit geometry, in inches, converted to figure fractions: the two panels
# are exactly the same square size, each colour bar hugs its own panel, and
# every gap is only as wide as the labels that have to fit into it.
# The colour bars run HORIZONTALLY beneath their panels rather than vertically
# beside them.  A vertical bar costs ~0.6 in of width (bar + tick labels +
# rotated label) on each side, all of it taken out of the panels; moving the
# bars into the vertical direction, which is not width-constrained, buys that
# width back and lets the panels grow while the gap between them shrinks to
# just the room panel (b)'s class names need.
W = 6.90            # true double-column width (17.5 cm): no rescaling in LaTeX
LM_A, LM_B = 0.62, 0.88   # left margins (tick labels + rotated axis label)
GAP, RM = 0.10, 0.08      # clear space between the panel blocks; right margin
S = 0.5 * (W - LM_A - GAP - LM_B - RM)          # panel side, both panels
XA, XB = LM_A, LM_A + S + GAP + LM_B

MET, CBTL, CBH, CBLB = 0.32, 0.15, 0.12, 0.15   # metrics, bar ticks, bar, label
XLB, XTL = 0.18, 0.34                            # axis label, tick labels
Y_CB = MET + 0.05 + CBTL                         # bottom of the colour bars
BOT = Y_CB + CBH + CBLB + 0.12 + XLB + XTL       # bottom of the panels
TOP = 0.32
H = BOT + S + TOP


def _fr(x0, y0, w, h):
    return [x0 / W, y0 / H, w / W, h / H]


Y0 = BOT
fig = plt.figure(figsize=(W, H))
axA = fig.add_axes(_fr(XA, Y0, S, S))
cax_a = fig.add_axes(_fr(XA, Y_CB, S, CBH))
axB = fig.add_axes(_fr(XB, Y0, S, S))
cax_b = fig.add_axes(_fr(XB, Y_CB, S, CBH))

# ------------------------------------------------ panel (a): regression -----
lo, hi = 1.2e-4, 0.13
hb = axA.hexbin(y_true, y_pred, xscale="log", yscale="log", gridsize=40,
                extent=(np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi)),
                cmap=CMAP_A, norm=LogNorm(vmin=1, vmax=300), mincnt=1,
                linewidths=0.0, zorder=2)
rng = np.random.default_rng(SEED)
sub = rng.choice(y_true.size, size=SUBSAMPLE, replace=False)
axA.scatter(y_true[sub], y_pred[sub], s=3.5, c="0.05", alpha=0.55,
            edgecolors="none", zorder=4,
            label=f"subsample ($n={SUBSAMPLE}$)")
axA.plot([lo, hi], [lo, hi], color="0.10", lw=0.7, zorder=5, label="1:1")
axA.plot([lo, hi], [2 * lo, 2 * hi], color="0.10", lw=0.7, ls=(0, (4, 2)),
         zorder=5, label="factor of 2")
axA.plot([lo, hi], [0.5 * lo, 0.5 * hi], color="0.10", lw=0.7, ls=(0, (4, 2)),
         zorder=5)

axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
axA.set_xlabel("Simulated MIDR (OpenSeesPy), –")
axA.set_ylabel("Predicted MIDR (surrogate), –")
axA.set_title("(a) Drift regression", loc="left", fontweight="bold", pad=5)
for axis in (axA.xaxis, axA.yaxis):
    axis.set_major_locator(LogLocator(base=10, numticks=6))
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1,
                                      numticks=12))
    axis.set_minor_formatter(NullFormatter())
axA.tick_params(which="both", top=True, right=True, direction="in")
axA.tick_params(which="major", length=4.5)
axA.tick_params(which="minor", length=2.5)
axA.grid(which="major", lw=0.5, color="0.85")
axA.set_axisbelow(True)

cbA = fig.colorbar(hb, cax=cax_a, orientation="horizontal", extend="max")
cbA.ax.xaxis.set_label_position("top")
cbA.set_label("Analyses per hexagon, –", fontsize=9, labelpad=3)
cbA.ax.tick_params(which="both", direction="in", length=3, labelsize=8.5)

legA = axA.legend(loc="lower right", fontsize=8.5, handlelength=1.6,
                  borderpad=0.30, labelspacing=0.28, handletextpad=0.5,
                  framealpha=0.95, edgecolor="0.7", fancybox=False)
legA.get_frame().set_linewidth(0.5)

# ------------------------------------------------ panel (b): confusion ------
CMn = CM / CM.sum(axis=1, keepdims=True).clip(min=1)
CMm = np.ma.masked_where(CM == 0, CMn)          # empty cells drawn white
im = axB.imshow(CMm, cmap=CMAP_B, vmin=0, vmax=1, aspect="auto")
axB.set_xticks(range(K)); axB.set_yticks(range(K))
axB.set_xticklabels(DS_NAMES, rotation=30, ha="right")
axB.set_yticklabels(DS_NAMES)
axB.set_xlabel("Predicted damage state")
axB.set_ylabel("True damage state")
axB.set_title("(b) EMS-98 damage-state classification", loc="left",
              fontweight="bold", pad=5)
axB.set_xticks(np.arange(-.5, K, 1), minor=True)
axB.set_yticks(np.arange(-.5, K, 1), minor=True)
axB.grid(which="minor", color="0.75", lw=0.5)
axB.tick_params(which="both", length=0)
for i in range(K):
    for j in range(K):
        if CM[i, j] == 0:
            continue
        axB.text(j, i, f"{CM[i, j]}\n{CMn[i, j]*100:.0f}%", ha="center",
                 va="center", fontsize=8.5,
                 color=_ink(CMAP_B, CMn[i, j]))

cbB = fig.colorbar(im, cax=cax_b, orientation="horizontal")
cbB.ax.xaxis.set_label_position("top")
cbB.set_label("Row-normalised fraction, –", fontsize=9, labelpad=3)
cbB.ax.tick_params(which="both", direction="in", length=3, labelsize=8.5)

# --------------- metric lines, outside the axes, one under each panel -------
# placed from the measured tight bounding box of each panel, so they clear the
# rotated tick labels and the axis title whatever the final layout does.
fig.canvas.draw()
_r = fig.canvas.get_renderer()
for ax, txt in ((cax_a, f"$R^2 = {R2:.3f}$   RMSE $= {RMSE:.3f}$ dex   "
                      f"MAE $= {MAE:.3f}$ dex\nwithin a factor of 2: "
                      f"{within2*100:.1f}%"),
                (cax_b, f"accuracy $= {acc:.3f}$   macro-$F_1 = {macroF1:.3f}$\n"
                      f"one-vs-rest ROC-AUC $= {AUC:.3f}$")):
    bb = ax.get_tightbbox(_r).transformed(fig.transFigure.inverted())
    fig.text(0.5 * (bb.x0 + bb.x1), bb.y0 - 0.022, txt, ha="center", va="top",
             fontsize=9)

# ------------------------------------------------------ checks + export -----
fig.canvas.draw()
r = _r
print("legend (a) covers_data hits:", covers_data(legA.get_window_extent(r), axA))
print("max hexagon count:", int(hb.get_array().max()))
check_figure(fig)
assert_not_dejavu(fig)

for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"{NAME}.{ext}", bbox_inches="tight")

print(f"R2={R2:.4f}  RMSE={RMSE:.4f} dex  MAE={MAE:.4f} dex  within2={within2:.4f}")
print(f"acc={acc:.4f}  macroF1={macroF1:.4f}  ROC-AUC={AUC:.4f}  n={len(d)}")
print("class support:", CM.sum(axis=1).tolist())
print("per-class F1:", [round(f, 3) for f in f1s])
