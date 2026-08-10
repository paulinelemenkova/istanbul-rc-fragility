#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig10_response_map -- predicted seismic damage-state distribution across the
Istanbul metropolitan area for a Marmara M 7.3 scenario rupture.

Pipeline
--------
1. Shaded-relief basemap from the GEBCO 2026 grid (cartographic-processing skill).
2. V_s30 site proxy from the DEM slope (Wald & Allen, 2007): flat basins -> soft
   (low V_s30), steep ground -> stiff (high V_s30).
3. Scenario S_a(T1) and PGA at every land cell from a distance attenuation off
   the Main Marmara Fault rupture, amplified by the site term (V_ref/V_s30)^p.
4. Expected EMS-98 damage state E[DS] = sum_k k * P(DS = k) obtained by
   evaluating the TRAINED GRADIENT-BOOSTED CLASSIFIER at every land cell, once
   for each of the 60 archetype frames, and averaging over the archetype
   population.  This is the surrogate doing the work the paper claims for it:
   roughly 2.5 million predictions, a calculation that would be prohibitive by
   direct nonlinear time-history analysis.

The classifier is the one described in Section 4.5 and scored in Section 5.2:
XGBoost, 500 trees, depth 5, learning rate 0.05, inverse-frequency class
weights, refitted here on the whole of simulation_database.csv.

SCOPE
-----
The hazard field is a scenario demonstration, not a calibrated hazard study: the
attenuation constants and the slope-to-V_s30 proxy stand in for a GMPE
(e.g. OpenQuake hazardlib) conditioned on a measured V_s30 model and
district-level GHSL exposure.  The damage field, by contrast, is now produced by
the released surrogate rather than by hard-coded fragility constants.  Two
extrapolation limits are reported at run time and should be quoted with the map:
cells whose S_a(T1) exceeds the highest sampled intensity (2.0 g) and cells
whose V_s30 falls below the lowest value in the training set (278 m/s).
Relief data: GEBCO 2026 (gebco.net).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LightSource, BoundaryNorm
from matplotlib.patches import FancyArrow
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_style import set_rc, assert_not_dejavu          # noqa: E402

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


def _gebco():
    """Locate the GEBCO tile in data/external/ without hard-coding its name."""
    hits = sorted(EXTERNAL.glob("gebco_*geotiff.tif"))
    if not hits:
        raise SystemExit(
            f"GEBCO grid not found in {EXTERNAL}. Download the regional tile "
            "from https://www.gebco.net and place it there.")
    return str(hits[0])

set_rc()          # Nimbus Sans, house rcParams (DejaVu Sans is forbidden)
DEM = os.environ.get("FIG10_DEM") or _gebco()
DB = str(RESULTS / "simulation_database.csv")   # the released simulation database

MW_SCENARIO = 7.3                              # scenario magnitude
SA_TRAIN_MAX = 2.00                            # highest sampled intensity (g)
VS30_TRAIN_MIN = 278.0                         # lowest V_s30 in the training set

# --- Istanbul metropolitan window -----------------------------------------
W, E, S, N = 28.40, 29.60, 40.70, 41.30

with rasterio.open(DEM) as ds:
    full = ds.read(1).astype("float64")
    b = ds.bounds
    flon = np.linspace(b.left, b.right, full.shape[1])
    flat = np.linspace(b.top, b.bottom, full.shape[0])
ci = np.where((flon >= W) & (flon <= E))[0]
ri = np.where((flat >= S) & (flat <= N))[0]
elev = full[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]
lon = flon[ci[0]:ci[-1] + 1]
lat = flat[ri[0]:ri[-1] + 1]
LON, LAT = np.meshgrid(lon, lat)
mlat = 0.5 * (S + N)

# --- topographic colormap + hillshade (reuse fig01 'geo' palette) ----------
vmin, vmax = -1200.0, 1200.0
anchors = [(-1200, "#08306b"), (-600, "#2b7bba"), (-150, "#7fb8df"), (-1, "#d8eef9"),
           (0, "#2e7d32"), (150, "#5fa052"), (350, "#9cb558"), (600, "#cbb56b"),
           (900, "#b0895a"), (1200, "#e9e2d8")]
topo_cmap = LinearSegmentedColormap.from_list("geo_ist", [((v - vmin) / (vmax - vmin), c) for v, c in anchors])
tnorm = Normalize(vmin, vmax)
dx = (lon[1] - lon[0]) * 111320.0 * np.cos(np.radians(mlat))
dy = (lat[0] - lat[1]) * 110570.0
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(elev, cmap=topo_cmap, norm=tnorm, blend_mode="soft", vert_exag=2.5, dx=dx, dy=dy)

# ==========================================================================
# === Scenario hazard field ================================================
# V_s30 proxy from slope (Wald & Allen 2007, active-tectonic relation, proxy)
gy, gx = np.gradient(elev, dy, dx)
slope = np.sqrt(gx**2 + gy**2)                       # m/m
Vs30 = np.clip(300.0 + 250.0 * np.log10((slope + 1e-4) / 3e-3), 180.0, 600.0)

# Main Marmara Fault rupture trace through the window (indicative)
fx = np.linspace(W - 0.1, E + 0.1, 120)
fy = 40.808 + 0.012 * np.sin((fx - W) * 3.0)         # gentle E-W trace ~40.81N
epi = (28.95, 40.81)                                  # scenario epicentre

# R_rup: min distance (km) from each cell to the fault polyline
Rrup = np.full(elev.shape, 1e9)
for xf, yf in zip(fx, fy):
    ddx = (LON - xf) * 111.320 * np.cos(np.radians(LAT))
    ddy = (LAT - yf) * 110.570
    Rrup = np.minimum(Rrup, np.sqrt(ddx**2 + ddy**2))

# scenario S_a(T1) and PGA: rock attenuation * site amplification.  PGA decays
# faster and is amplified less than the longer-period spectral ordinate, so the
# spectral-shape ratio Sa(T1)/PGA -- a feature the surrogate uses -- varies
# across the map rather than being constant.
A0, kdec, Vref, pamp = 0.95, 0.045, 600.0, 0.55
Sa = np.clip(A0 * np.exp(-kdec * Rrup) * (Vref / Vs30) ** pamp, 1e-3, 1.6)
PGA = np.clip(0.62 * A0 * np.exp(-0.060 * Rrup) * (Vref / Vs30) ** 0.35, 1e-3, 1.6)
# ==========================================================================

# --------------------------------------------------------------------------
# Expected EMS-98 damage state from the TRAINED SURROGATE
# --------------------------------------------------------------------------
FEATURES = ["sa_t1_g", "pga_scaled_g", "sa_ratio", "mw", "rjb_km", "vs30",
            "n_storey", "T1_s", "fc_MPa", "rho_long", "r_k",
            "soft_i", "deficient_i"]
STRUCT = ["n_storey", "T1_s", "fc_MPa", "rho_long", "r_k", "soft_i", "deficient_i"]


def train_surrogate():
    """Refit the Section 4.5 classifier on the whole released database."""
    from xgboost import XGBClassifier
    db = pd.read_csv(DB, keep_default_na=False, na_values=[""])
    db["sa_t1_g"] = db["sa_target_g"]
    db["pga_scaled_g"] = db["pga_g"] * db["scale"]
    db["sa_ratio"] = db["sa_record_g"] / db["pga_g"]
    db["soft_i"] = db["soft"].astype(int)
    db["deficient_i"] = db["deficient"].astype(int)
    X = db[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    nan = np.isnan(X)                                   # one record lacks V_s30
    if nan.any():
        X[nan] = np.take(np.nanmedian(X, axis=0), np.where(nan)[1])
    y = db["ds_index"].to_numpy(int)
    cnt = np.bincount(y, minlength=5).astype(float)
    w = (len(y) / (5 * cnt))[y]                         # inverse-frequency weights
    clf = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        objective="multi:softprob", num_class=5, n_jobs=-1,
                        random_state=7).fit(X, y, sample_weight=w)
    archetypes = (db.drop_duplicates("fid")[STRUCT]
                    .to_numpy(float))                   # 60 frames x 7 parameters
    return clf, archetypes


land = elev >= 0
clf, ARCH = train_surrogate()

sa_f = Sa[land]
pga_f = PGA[land]
site = np.column_stack([sa_f, pga_f, sa_f / pga_f,
                        np.full(sa_f.size, MW_SCENARIO), Rrup[land], Vs30[land]])

# One pass per archetype; average the damage-state probabilities over the
# population, then take the expectation of the EMS-98 grade.
acc = np.zeros((sa_f.size, 5))
for row in ARCH:
    F = np.column_stack([site, np.tile(row, (sa_f.size, 1))])
    acc += clf.predict_proba(F)
acc /= len(ARCH)
eds_flat = acc @ np.arange(5.0)

EDS = np.zeros_like(Sa)
EDS[land] = eds_flat

n_pred = sa_f.size * len(ARCH)
hi_sa = float(np.mean(sa_f > SA_TRAIN_MAX))
lo_vs = float(np.mean(Vs30[land] < VS30_TRAIN_MIN))
print(f"surrogate evaluations: {n_pred:,} ({sa_f.size:,} land cells x "
      f"{len(ARCH)} archetypes)")
print(f"extrapolation: {hi_sa*100:.1f}% of cells above Sa = {SA_TRAIN_MAX} g, "
      f"{lo_vs*100:.1f}% below Vs30 = {VS30_TRAIN_MIN:.0f} m/s")

EDS_m = np.ma.masked_where(~land, EDS)

# magma, reversed, for the damage field: pale yellow at the safe end (None)
# darkening monotonically to near-black at Collapse.  Monotonic lightness means
# ink is proportional to expected damage, the shaded relief stays visible under
# the large undamaged area, and the field survives greyscale conversion and
# colour-vision deficiency.
dmg_cmap = plt.get_cmap("magma_r")
dnorm = Normalize(0.0, 4.0)

# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(11.6, 7.0))
ax = fig.add_axes([0.06, 0.05, 0.86, 0.86])
ax.set_aspect(1.0 / np.cos(np.radians(mlat)))
div = make_axes_locatable(ax)
cax = div.append_axes("right", size="2.6%", pad=0.12)

ax.imshow(rgb, extent=[W, E, S, N], origin="upper", interpolation="bilinear", zorder=1)
ax.contour(LON, LAT, elev, levels=[0], colors="k", linewidths=0.7, zorder=4)
# GEBCO isolines at a 200 m interval: azure below sea level, brown above
ax.contour(LON, LAT, elev, levels=np.arange(-1400, 0, 200), colors="#4fb3d9",
           linewidths=0.6, alpha=0.75, zorder=2.5)
ax.contour(LON, LAT, elev, levels=np.arange(200, 1401, 200), colors="#8c5a2b",
           linewidths=0.6, alpha=0.75, zorder=2.5)
dm = ax.pcolormesh(LON, LAT, EDS_m, cmap=dmg_cmap, norm=dnorm, alpha=0.72,
                   shading="auto", zorder=3)
# Grade boundaries as isolines: most of the land falls inside the first EMS-98
# grade, where a continuous 0-4 colour ramp resolves poorly, so the transitions
# are drawn explicitly.
from scipy.ndimage import uniform_filter
EDS_s = np.ma.masked_where(~land, uniform_filter(EDS, size=3))
cs = ax.contour(LON, LAT, EDS_s, levels=[1.0, 2.0, 3.0], colors="0.05",
                linewidths=1.2, zorder=3.6)
for coll in cs.get_paths():
    pass
cs.set(path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
lbl = ax.clabel(cs, fmt={1.0: "$E[DS]$=1", 2.0: "$E[DS]$=2", 3.0: "$E[DS]$=3"},
                fontsize=10, inline=True, inline_spacing=5)
for t in lbl:
    t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    t.set_zorder(8.5)

# fault + epicentre
ax.plot(fx, fy, color="#c8102e", lw=3.0, zorder=5)
ax.scatter([epi[0]], [epi[1]], marker="*", s=260, c="#ffe000", edgecolors="k",
           linewidths=0.8, zorder=7)
ax.text(epi[0] + 0.012, epi[1] - 0.016, "Scenario epicentre\n(Main Marmara Fault, $M\\,7.3$)",
        fontsize=10, color="white", style="italic", ha="left", va="top", zorder=8,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="#3a0a14")])

# --- transport corridors ---------------------------------------------------
# SCHEMATIC alignments: polylines through the approximate coordinates of the
# corridors' well-known waypoints, digitised by hand, NOT an authoritative GIS
# dataset.  They run to both edges of the window so that no corridor appears to
# terminate inside the map: the O-3/TEM and D-100 continue west toward Silivri
# and east toward Izmit, and the rail corridor continues beyond the Marmaray
# termini (Halkali in the west, Gebze in the east) as the main line.
tem = np.array([(28.30, 41.105), (28.48, 41.098), (28.62, 41.092), (28.80, 41.085),
                (28.95, 41.092), (29.06, 41.090), (29.15, 41.048), (29.28, 40.972),
                (29.38, 40.900), (29.50, 40.832), (29.70, 40.800)])
d100 = np.array([(28.28, 41.072), (28.42, 41.068), (28.52, 41.040), (28.62, 41.005),
                 (28.72, 40.982), (28.87, 40.978),
                 (28.92, 41.002), (28.97, 41.022), (29.02, 41.040), (29.06, 41.020),
                 (29.10, 40.985), (29.13, 40.938), (29.19, 40.902), (29.24, 40.878),
                 (29.27, 40.862), (29.30, 40.828),
                 (29.43, 40.802), (29.70, 40.792)])
rail = np.array([(28.28, 41.086), (28.45, 41.080), (28.58, 41.055), (28.70, 41.040),
                 (28.79, 41.032), (28.87, 40.982),
                 (28.92, 41.000), (28.96, 41.012), (28.98, 41.017), (29.01, 41.026), (29.03, 41.022),
                 (29.06, 40.992), (29.10, 40.968), (29.13, 40.935), (29.19, 40.902),
                 (29.24, 40.876), (29.27, 40.860), (29.30, 40.826), (29.43, 40.800), (29.70, 40.790)])


def keep_on_land(line, n=500, reach=0.05, step=0.002, smooth=9):
    """Densify a corridor and pull any sample that falls in the sea inland.

    The GEBCO coastline is 15 arc-seconds (~460 m), so a hand-digitised corridor
    following a narrow coastal strip can fall on the seaward side of the mask
    even where the real road is on land.  Each wet sample is walked north (then
    south) until it reaches land, up to `reach` degrees; samples that find no
    land within that distance are genuine water crossings -- the Bosphorus
    bridge, the Marmaray tunnel, the Buyukcekmece bay causeway -- and are left
    alone.  A short moving average removes the sawtooth the snapping leaves.
    """
    seg = np.asarray(line, float)
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1])))]
    d /= d[-1]
    t = np.linspace(0, 1, n)
    xs, ys = np.interp(t, d, seg[:, 0]), np.interp(t, d, seg[:, 1])

    def wet(x, y):
        if not (W <= x <= E and S <= y <= N):
            return False
        j = int(np.argmin(np.abs(lon - x)))
        i = int(np.argmin(np.abs(lat - y)))
        return elev[i, j] < 0

    for m in range(n):
        if not wet(xs[m], ys[m]):
            continue
        for off in np.arange(step, reach + 1e-9, step):
            for cand in (ys[m] + off, ys[m] - off):
                if not wet(xs[m], cand):
                    ys[m] = cand
                    break
            else:
                continue
            break
    if smooth > 1:
        k = np.ones(smooth) / smooth
        ys = np.convolve(np.r_[[ys[0]] * (smooth // 2), ys,
                               [ys[-1]] * (smooth // 2)], k, mode="valid")
    return np.column_stack([xs, ys])


tem, d100, rail = (keep_on_land(tem), keep_on_land(d100), keep_on_land(rail))

# motorway + highway: thin coloured line over a dark casing
for line, lwid in [(tem, 2.2), (d100, 2.0)]:
    ax.plot(line[:, 0], line[:, 1], color="#5a2a00", lw=lwid, solid_capstyle="round", zorder=5.5)
ax.plot(tem[:, 0], tem[:, 1], color="#ff7f0e", lw=1.2, solid_capstyle="round", zorder=5.6)
ax.plot(d100[:, 0], d100[:, 1], color="#ffd92f", lw=1.2, solid_capstyle="round", zorder=5.6)
# railway: black line with white "ties"
ax.plot(rail[:, 0], rail[:, 1], color="k", lw=1.7, solid_capstyle="butt", zorder=5.7)
ax.plot(rail[:, 0], rail[:, 1], color="white", lw=0.8, dashes=(1, 3), zorder=5.8)

# place labels
def lab(lo, la, txt, fs=10, col="#111", style="normal", weight="normal", rot=0,
        ha="center", marker=True, sea=False):
    if sea:
        t = ax.text(lo, la, txt, fontsize=fs, color="white", style="italic",
                    weight="bold", rotation=rot, ha=ha, va="center", zorder=8)
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="#08306b")])
        return
    if marker:
        ax.scatter([lo], [la], s=20, marker="s", c="#111", zorder=8)
    t = ax.text(lo, la + 0.018, txt, fontsize=fs, color=col, style=style,
                weight=weight, rotation=rot, ha=ha, va="bottom", zorder=8)
    t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

# --- 20 cities/districts (Istanbul = capital marker; 19 districts/towns) -----
ax.scatter([28.979], [41.008], s=190, marker="o", c="#d40000", edgecolors="k",
           linewidths=1.1, zorder=9)
_ist = ax.text(28.972, 41.023, "\u0130stanbul", fontsize=15.5, color="white",
               weight="bold", ha="center", va="bottom", zorder=10)
_ist.set_path_effects([pe.withStroke(linewidth=2.8, foreground="#7a0000")])

def city(lo, la, name, dx=0.0, dy=0.013, ha="center", fs=10.0):
    ax.plot([lo], [la], marker="o", ms=8, mfc="#ff8c00", mec="white", mew=0.6,
            ls="none", zorder=8)
    # plain black type, no casing: legible over the pale end of the damage
    # ramp, weak where the field darkens toward Collapse.
    ax.text(lo + dx, la + dy, name, fontsize=fs, color="black", ha=ha,
            va="bottom", zorder=9)

CITIES = [
    ("Kad\u0131k\u00f6y", 29.030, 40.990, 0.012, -0.004, "left"),
    ("Bak\u0131rk\u00f6y", 28.872, 40.980, 0.0, -0.030, "center"),
    ("Pendik", 29.235, 40.876, 0.012, 0.0, "left"),
    ("B\u00fcy\u00fck\u00e7ekmece", 28.585, 41.022, 0.0, 0.012, "center"),
    ("Gebze", 29.430, 40.802, 0.012, 0.0, "left"),
    ("Beylikd\u00fcz\u00fc", 28.642, 40.985, 0.0, -0.030, "center"),
    ("Avc\u0131lar", 28.722, 40.979, -0.010, 0.010, "right"),
    ("K\u00fc\u00e7\u00fck\u00e7ekmece", 28.781, 41.010, 0.0, 0.012, "center"),
    ("Ba\u011fc\u0131lar", 28.857, 41.045, 0.0, 0.012, "center"),
    ("\u00c7atalca", 28.461, 41.143, 0.012, 0.0, "left"),
    ("Fatih", 28.949, 41.019, 0.0, -0.030, "center"),
    ("Be\u015fikta\u015f", 29.008, 41.047, 0.040, 0.010, "left"),
    ("\u00dcsk\u00fcdar", 29.018, 41.022, 0.012, -0.006, "left"),
    ("\u00dcmraniye", 29.124, 41.024, 0.0, 0.012, "center"),
    ("Sar\u0131yer", 29.052, 41.167, 0.012, 0.0, "left"),
    ("Maltepe", 29.131, 40.935, 0.012, 0.0, "left"),
    ("Kartal", 29.190, 40.905, 0.012, 0.0, "left"),
    ("Tuzla", 29.302, 40.817, 0.0, -0.030, "center"),
    ("Dar\u0131ca", 29.378, 40.772, 0.0, 0.012, "center"),
]
for nm, lo, la, dx, dy, ha in CITIES:
    city(lo, la, nm, dx, dy, ha)

# transport legend (upper-left, over low-damage area)
leg_h = [
    Line2D([0], [0], color="#ff7f0e", lw=2.2, label="Motorway (TEM, O-3)"),
    Line2D([0], [0], color="#ffd92f", lw=2.0, label="Highway (D-100)"),
    Line2D([0], [0], color="k", lw=2.0, label="Rail (Marmaray + main line)"),
    Line2D([0], [0], color="#c8102e", lw=2.0, label="Main Marmara Fault"),
]
leg = ax.legend(handles=leg_h, loc="upper left", fontsize=10, framealpha=0.9,
                edgecolor="0.6", borderpad=0.5, handlelength=1.8,
                title="Transport \u00b7 fault (schematic)")
leg.get_title().set_fontsize(7.6)
leg.set_zorder(11)
lab(28.90, 40.74, "Sea of Marmara", 11, rot=0, ha="center", sea=True)
lab(29.085, 41.15, "Bosphorus", 8.5, rot=79, sea=True)
lab(29.52, 40.76, "Gulf of \u0130zmit", 8.5, sea=True)
lab(29.05, 41.275, "Black Sea", 11, rot=0, sea=True)

# frame, graticule (DMS minor ticks)
ax.set_xlim(W, E); ax.set_ylim(S, N)
xt = np.arange(28.5, 29.61, 0.25); yt = np.arange(40.75, 41.31, 0.25)
ax.set_xticks(xt); ax.set_yticks(yt)
def _xl(v):
    d = int(v); m = int(round((v - d) * 60))
    return f"{d}\u00b0E" if m == 0 else f"{d}\u00b0{m:02d}\u2032E"
def _yl(v):
    d = int(v); m = int(round((v - d) * 60))
    return f"{d}\u00b0N" if m == 0 else f"{d}\u00b0{m:02d}\u2032N"
ax.set_xticklabels([_xl(v) for v in xt], fontsize=10)
ax.set_yticklabels([_yl(v) for v in yt], fontsize=10)
ax.xaxis.set_minor_locator(MultipleLocator(1 / 60.0))   # 1'
ax.yaxis.set_minor_locator(MultipleLocator(1 / 60.0))   # 1'
ax.tick_params(which="major", top=True, right=True, labeltop=False, labelright=False,
               direction="out", length=5, labelsize=8.3)
ax.tick_params(which="minor", top=True, right=True, direction="out", length=2.4)
ax.grid(True, which="major", color="white", lw=1.2, alpha=0.85)
for sp in ax.spines.values(): sp.set_edgecolor("#333"); sp.set_linewidth(1.1)

# damage colourbar: 5 EMS-98 states as labelled major ticks; the 10 colour steps
# (numeric E[DS] in 0.4 increments) shown as minor ticks.
sm = plt.cm.ScalarMappable(cmap=dmg_cmap, norm=dnorm); sm.set_array([])
cb = fig.colorbar(sm, cax=cax, ticks=[0, 1, 2, 3, 4])
cb.set_label("Expected EMS-98 damage state  $E[DS]$ (surrogate)", fontsize=10)
cb.ax.set_yticklabels(["None", "Light", "Moderate", "Heavy", "Collapse"], fontsize=10)
cb.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
cb.ax.tick_params(which="minor", length=2.4, color="0.25")
cb.ax.tick_params(which="major", length=5)

# scale bar (km)
km = 20.0; dlon = km / (111.320 * np.cos(np.radians(mlat)))
x0, y0 = W + 0.06, S + 0.05
ax.plot([x0, x0 + dlon], [y0, y0], color="k", lw=3.0, solid_capstyle="butt", zorder=10)
for fr in (0, 0.5, 1.0):
    ax.plot([x0 + fr * dlon] * 2, [y0, y0 + 0.012], color="k", lw=1.0, zorder=10)
    _ts = ax.text(x0 + fr * dlon, y0 + 0.02, f"{int(fr*km)}", ha="center", va="bottom", fontsize=10, color="white", zorder=10)
    _ts.set_path_effects([pe.withStroke(linewidth=1.8, foreground="#1a1a1a")])
_tk = ax.text(x0 + dlon / 2, y0 - 0.028, "km \u00b7 Mercator", ha="center", va="top", fontsize=10, color="white", zorder=10)
_tk.set_path_effects([pe.withStroke(linewidth=1.8, foreground="#1a1a1a")])

# north arrow (over Black Sea water, top of map)
nx, ny = 29.545, 41.185
ax.add_patch(FancyArrow(nx, ny, 0, 0.050, width=0.003, head_width=0.020, head_length=0.022,
             length_includes_head=True, color="k", zorder=10))
nt = ax.text(nx, ny + 0.058, "N", ha="center", va="bottom", fontsize=10, fontweight="bold", color="white", zorder=10)
nt.set_path_effects([pe.withStroke(linewidth=2.4, foreground="#1a1a1a")])

ax.text(0.5, 1.045, "Predicted damage-state distribution across Istanbul",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold")
ax.text(0.5, 1.010, "Marmara $M\\,7.3$ scenario: gradient-boosted surrogate evaluated at every land cell",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#333")

assert_not_dejavu(fig)
fig.savefig(FIGDIR / "fig10_response_map.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.06)
fig.savefig(FIGDIR / "fig10_response_map.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.06)
print("E[DS] land range: %.2f..%.2f | Sa land range: %.2f..%.2f g | Vs30 %.0f..%.0f" %
      (EDS[land].min(), EDS[land].max(), Sa[land].min(), Sa[land].max(),
       Vs30[land].min(), Vs30[land].max()))
print("E[DS] mean %.2f | fraction of land above 'Moderate' (E[DS] >= 2): %.1f%%" %
      (EDS[land].mean(), 100.0 * np.mean(EDS[land] >= 2.0)))
