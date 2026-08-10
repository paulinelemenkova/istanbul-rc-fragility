#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig08_shap_importance -- SHAP feature-importance summary for the interpretable-ML
RC-frame fragility study (Marmara / Istanbul).

Two panels, shared feature ordering:
  (a) mean |SHAP| global importance (ranked horizontal bars);
  (b) SHAP summary "beeswarm": one point per sample per feature, positioned by
      its SHAP value (impact on predicted drift) and coloured by the feature
      value (low -> high), revealing the DIRECTION of each effect.

PROVENANCE
----------
The attribution is EXACT TreeSHAP, read from the fitted gradient-boosted
regressor with XGBoost's `pred_contribs=True` -- no sampling, no kernel
approximation, no `shap` dependency.  The model is the one of Section 4.5,
refitted on the whole released simulation database; the target is
log10(MIDR), so SHAP values are in dex.  The feature set is the one the
database actually contains: the earlier draft of this figure also listed Arias
intensity, significant duration, axial-load ratio and transverse confinement,
none of which is a column of simulation_database.csv.

House style: Set3 bars, haxby feature-value colour scale, minor ticks + thin
grid on the value axes, compact layout (shared y, anchored panels, minimal
gaps), Nimbus Sans, vector PDF + 600 dpi PNG.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.ticker import AutoMinorLocator           # noqa: E402
from matplotlib.gridspec import GridSpec                 # noqa: E402
from matplotlib.cm import ScalarMappable                 # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap   # noqa: E402
import xgboost as xgb                                    # noqa: E402
from xgboost import XGBRegressor                         # noqa: E402

from mpl_style import set_rc, assert_not_dejavu          # noqa: E402
from check_overlap import check_figure                   # noqa: E402
from oof_one import load, FEATURES                       # noqa: E402

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

set_rc()
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8, "axes.labelpad": 2, "axes.titlepad": 4,
    "savefig.dpi": 600,
})

SEED, NSUB = 7, 2000
# Panel (a): Set3, one colour per feature. Set3 is qualitative and carries no
# ordering, so the ranking is read from bar length rather than from hue.
SET3 = plt.get_cmap("Set3")(np.linspace(0.0, 1.0, len(FEATURES)))
# Panel (b): haxby (GMT), rebuilt from its control points -- matplotlib does
# not ship it. Sampled short of the white end so high values stay visible.
HAXBY = LinearSegmentedColormap.from_list("haxby", np.array([
    (37, 57, 175), (40, 127, 251), (50, 190, 255), (106, 235, 255),
    (138, 236, 174), (205, 255, 162), (240, 236, 121), (255, 189, 87),
    (255, 161, 68), (255, 186, 133), (255, 255, 255)], float) / 255.0)
HAXBY = LinearSegmentedColormap.from_list("haxby_t", HAXBY(np.linspace(0, 0.82, 256)))

DISPLAY = {
    "sa_t1_g": r"$S_a(T_1)$, scaled", "pga_scaled_g": "PGA, scaled",
    "sa_ratio": "Sa$_{ratio}$", "mw": r"$M_w$", "rjb_km": r"$R_{jb}$",
    "vs30": r"$V_{s30}$", "n_storey": "No. of storeys", "T1_s": r"$T_1$",
    "fc_MPa": r"$f'_c$", "rho_long": "Long. steel ratio",
    "r_k": r"Stiffness ratio $r_k$", "soft_i": "Soft-storey flag",
    "deficient_i": "Deficient detailing flag",
}

# ----------------------------------------------------------------------------
# === Real SHAP results ======================================================
#   feat_names : feature labels
#   shap_vals  : (n_samples, n_features) exact TreeSHAP values, dex
#   feat_disp  : (n_samples, n_features) feature values as 0-1 percentiles,
#                for the colour scale (binary flags stay 0/1)
# ----------------------------------------------------------------------------
d, X = load()
y = np.log10(d["midr"].to_numpy(float))
reg = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                   objective="reg:squarederror", n_jobs=-1,
                   random_state=SEED).fit(X, y)
SH_ALL = reg.get_booster().predict(
    xgb.DMatrix(X, feature_names=FEATURES), pred_contribs=True)[:, :-1]

feat_names = [DISPLAY[f] for f in FEATURES]
mean_abs = np.abs(SH_ALL).mean(axis=0)          # panel (a): ALL 10 800 analyses

# panel (b): subsample stratified by EMS-98 damage state, for legibility
rng = np.random.default_rng(SEED)
ds = d["ds_index"].to_numpy(int)
per = NSUB // 5
sub = np.concatenate([rng.choice(np.flatnonzero(ds == k),
                                 size=min(per, int((ds == k).sum())),
                                 replace=False) for k in range(5)])
shap_vals = SH_ALL[sub]
nsub, nf = shap_vals.shape
binary = np.array([f in ("soft_i", "deficient_i") for f in FEATURES])
feat_disp = np.zeros_like(shap_vals)
for j in range(nf):
    v = X[sub, j]
    feat_disp[:, j] = v if binary[j] else np.argsort(np.argsort(v)) / (nsub - 1)
# ============================================================================

order = np.argsort(mean_abs)[::-1]                # most important first
ypos = np.arange(nf)                              # 0 = top after invert


def beeswarm_offsets(x, width=0.42, nbins=90):
    """Density-aware vertical offsets (classic beeswarm look)."""
    x = np.asarray(x)
    edges = np.linspace(np.nanmin(x), np.nanmax(x) + 1e-12, nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    off = np.zeros_like(x, dtype=float)
    counts = np.bincount(idx, minlength=nbins)
    for b in range(nbins):
        m = np.where(idx == b)[0]
        k = len(m)
        if k == 0:
            continue
        pos = np.arange(k) - (k - 1) / 2.0
        if k > 1:
            pos = pos / np.max(np.abs(pos))
        # scale by local density so sparse bins do not spread to full width
        off[m] = pos * width * np.sqrt(k / max(counts.max(), 1))
    return off


# ----------------------------------------------------------------------------
# Figure (compact: shared y, anchored panels, minimal gaps)
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(6.9, 4.3), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
gs = GridSpec(1, 2, figure=fig, width_ratios=[0.62, 1.0])
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1], sharey=axA)
axA.set_anchor("E")
axB.set_anchor("W")

# ---- panel (a): mean |SHAP| bars ------------------------------------------
axA.barh(ypos, mean_abs[order], color=SET3[order], edgecolor="0.25", lw=0.5,
         height=0.7)
axA.set_yticks(ypos)
axA.set_yticklabels([feat_names[k] for k in order])
axA.invert_yaxis()                                # most important on top
axA.set_xlabel("mean |SHAP| (dex)")
axA.set_title("(a) Global importance")
axA.set_xlim(0, mean_abs.max() * 1.12)
axA.xaxis.set_minor_locator(AutoMinorLocator())
axA.tick_params(which="both", top=True, direction="in")
axA.tick_params(which="major", length=4.5)
axA.tick_params(which="minor", length=2.5)
axA.grid(which="major", axis="x", lw=0.5, color="0.85")
axA.grid(which="minor", axis="x", lw=0.3, color="0.92")
axA.set_axisbelow(True)

# ---- panel (b): SHAP beeswarm ---------------------------------------------
axB.axvline(0.0, color="0.45", lw=1.0, zorder=2)
norm = Normalize(0, 1)
for r, j in enumerate(order):
    sv = shap_vals[:, j]
    yy = r + beeswarm_offsets(sv)
    axB.scatter(sv, yy, c=feat_disp[:, j], cmap=HAXBY, norm=norm,
                s=2.5, alpha=0.7, edgecolors="none", zorder=3, rasterized=True)
axB.set_xlabel("SHAP value (impact on predicted log$_{10}$ MIDR, dex)")
axB.set_title("(b) Direction of effect")
axB.tick_params(labelleft=False)                  # shared y; labels on (a) only
axB.xaxis.set_minor_locator(AutoMinorLocator())
axB.tick_params(which="both", top=True, direction="in")
axB.tick_params(which="major", length=4.5)
axB.tick_params(which="minor", length=2.5)
axB.grid(which="major", axis="x", lw=0.5, color="0.85")
axB.grid(which="minor", axis="x", lw=0.3, color="0.92")
axB.set_axisbelow(True)
axB.set_ylim(nf - 0.5, -0.5)                       # match inverted shared axis

sm = ScalarMappable(norm=norm, cmap=HAXBY); sm.set_array([])
cb = fig.colorbar(sm, ax=axB, fraction=0.046, pad=0.02)
cb.set_label("Feature value (low to high)", fontsize=9)
cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
cb.ax.tick_params(which="both", direction="in", length=3, labelsize=8.5)

fig.canvas.draw()
check_figure(fig)
assert_not_dejavu(fig)
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"fig08_shap_importance.{ext}", bbox_inches="tight",
                pad_inches=0.02)

tab = pd.DataFrame({
    "feature": [FEATURES[k] for k in order],
    "mean_abs_shap_dex": mean_abs[order],
    "corr_feature_vs_shap": [float(np.corrcoef(X[:, k], SH_ALL[:, k])[0, 1])
                             for k in order]})
DERIVED.mkdir(parents=True, exist_ok=True)
tab.to_csv(DERIVED / "shap_importance.csv", index=False)
print(tab.round(4).to_string(index=False))
GM = ["sa_t1_g", "pga_scaled_g", "sa_ratio", "mw", "rjb_km", "vs30"]
g = mean_abs[[FEATURES.index(f) for f in GM]].sum()
print(f"\nground-motion + site {100*g/mean_abs.sum():.1f}% | structural "
      f"{100*(1-g/mean_abs.sum()):.1f}% | panel (a) n = {len(d)} | "
      f"panel (b) n = {nsub}")
