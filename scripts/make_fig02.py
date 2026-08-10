#!/usr/bin/env python3
"""
fig02_vs30_site_class - Istanbul metropolitan area site characterisation.

  (a) Vs30 from the Wald & Allen (2007) active-tectonic slope->Vs30 relation
      applied to the GEBCO 2026 grid, with the SMD-TR strong-motion stations
      of the map window overlaid on the same colour scale.
  (b) the corresponding EC8/NEHRP site classes (180/360/760 m/s thresholds).
  (c) station Vs30 against the proxy value sampled at the same coordinates -
      a direct test of whether the proxy field is defensible for this window.

Station data: SMD-TR flatfile, NHERI DesignSafe PRJ-3950 v3,
DOI 10.17603/ds2-f21x-s189, table Metadata.csv, reduced to unique in-window
sites by extract_smdtr_vs30.py.

Vs30_Flag (from ColumnInfo.csv):
    1 = calculated from available reports
    2 = calculated from scaled limited soil profile
    3 = AFAD reported value
    4 = unavailable
Sentinels -1 (Vs30) and -999 (Z1, f0) mark unavailable entries and are dropped.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import rasterio
import scipy
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter

from mpl_style import ensure_nimbus, assert_not_dejavu                  # noqa: E402

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

ensure_nimbus()
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Helvetica"],
    "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold",
    "mathtext.cal": "Nimbus Sans:italic", "mathtext.sf": "Nimbus Sans",
    "mathtext.tt": "Nimbus Sans", "mathtext.default": "it",
    "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white",
})
HALO = [matplotlib.patheffects.withStroke(linewidth=1.6, foreground="white")]

FS_TAG, FS_LAB, FS_TICK = 10.0, 9.5, 8.5

DEM = _gebco()
STA = str(DATA / "AFAD_vs30.csv")   # produced by extract_smdtr_vs30.py
OUT = str(FIGDIR / "fig02_vs30_site_class")
W, E, S, N = 28.0, 30.0, 40.5, 41.5

# ---- GEBCO window ---------------------------------------------------------
src = rasterio.open(DEM)
elev = src.read(1, window=from_bounds(W, S, E, N, src.transform)).astype(float)
ny, nx = elev.shape
meanlat = 0.5 * (S + N)
dx = (E - W) / (nx - 1) * 111320.0 * np.cos(np.deg2rad(meanlat))
dy = (N - S) / (ny - 1) * 110540.0
gy, gx = np.gradient(gaussian_filter(elev, 1.2), dy, dx)
slope = np.hypot(gx, gy)

# ---- Wald & Allen (2007) active-tectonic slope -> Vs30 --------------------
SL = np.array([1e-5, 2.2e-3, 6.3e-3, 1.8e-2, 5.0e-2, 1.0e-1, 1.38e-1, 3.0e-1])
VS = np.array([180., 240., 300., 360., 490., 620., 760., 760.])
vs30 = np.interp(np.log10(np.clip(slope, SL[0], SL[-1])), np.log10(SL), VS)
sea = elev <= 0

# ---- stations -------------------------------------------------------------
FLAG_TXT = {"1": "from available reports",
            "2": "from scaled limited profile",
            "3": "AFAD reported value"}
FLAG_MK = {"1": "o", "2": "s", "3": "^"}
FLAG_COL = {"1": "#8968CD",     # mediumpurple3
            "2": "#FF82AB",     # palevioletred1
            "3": "#97FFFF"}     # darkslategray1

sta = []
for r in csv.DictReader(open(STA, encoding="utf-8-sig")):
    try:
        lo, la, v = float(r["lon"]), float(r["lat"]), float(r["vs30"])
    except (KeyError, ValueError):
        continue
    if v <= 0 or not (W <= lo <= E and S <= la <= N):
        continue
    j = min(max(int(round((lo - W) / (E - W) * (nx - 1))), 0), nx - 1)
    i = min(max(int(round((N - la) / (N - S) * (ny - 1))), 0), ny - 1)
    sta.append(dict(lon=lo, lat=la, vs30=v, flag=r.get("vs30_source", "").strip(),
                    proxy=float(vs30[i, j]), slope=float(slope[i, j])))
print(f"{len(sta)} stations plotted")

lr = np.log10(np.array([s["vs30"] for s in sta]) / np.array([s["proxy"] for s in sta]))
ratio, rmse = 10 ** np.median(lr), float(np.sqrt((lr ** 2).mean()))
within = int(np.sum(np.abs(lr) < np.log10(1.5)))
print(f"median station/proxy {ratio:.2f} | log10 RMSE {rmse:.3f} "
      f"(factor {10 ** rmse:.2f}) | within 1.5x: {within}/{len(sta)}")

# ---- colour scales --------------------------------------------------------
# (a) gnuplot2, truncated at 0.90 so the high (safest) end is yellow rather
#     than white, which would be indistinguishable from the page.
# (b) three saturated classes sampled from jet at its pure blue / yellow / red
#     nodes.
# Note: neither ramp is perceptually uniform or colour-blind safe; cividis is
# kept below as a one-line alternative.
_g2 = plt.get_cmap("gnuplot2")
vs_cmap = ListedColormap(_g2(np.linspace(0.06, 0.90, 256)), name="gnuplot2_trunc")
# vs_cmap = plt.get_cmap("cividis")        # CVD-safe alternative
vs_norm = Normalize(180, 760)

# X11/R colour names, given as hex because Matplotlib knows only the
# unnumbered CSS variants (e.g. "royalblue" = #4169E1, not royalblue1).
# ordered by hazard: red = softest and most damaging, blue = stiffest ground
CLS_COL = ["#EE6363",      # indianred2  - EC8 C / NEHRP D, 180-360 (worst)
           "#FFD700",      # gold        - EC8 B / NEHRP C, 360-760
           "#4876FF"]      # royalblue1  - >=760, ceiling of the relation
cls_cmap = ListedColormap(CLS_COL)
cls_norm = BoundaryNorm([180, 360, 760, 3000], cls_cmap.N)


def ec8_colour(v):
    """Class colour of a Vs30 value, matching panel (b)."""
    return CLS_COL[0] if v < 360 else (CLS_COL[1] if v < 760 else CLS_COL[2])

ls = LightSource(azdeg=315, altdeg=45)
hs = np.dstack([ls.hillshade(elev, vert_exag=2.0, dx=dx, dy=dy)] * 3)


def drape(data, cmap, norm):
    rgb = np.clip(cmap(norm(data))[..., :3] * (0.55 + 0.45 * hs), 0, 1)
    rgb[sea] = np.array([0.83, 0.89, 0.94])
    return rgb


rgb_vs = drape(vs30, vs_cmap, vs_norm)
rgb_cls = drape(np.clip(vs30, 180, 2999), cls_cmap, cls_norm)

# name, lon, lat, ha, va, (dx, dy) in points
CITIES = [("\u0130stanbul", 28.98, 41.02, "right", "bottom", (-5, 7)),
          ("Kad\u0131k\u00f6y", 29.03, 40.99, "left", "top", (6, -5)),
          ("Silivri", 28.25, 41.07, "center", "bottom", (0, 6)),
          ("B\u00fcy\u00fck\u00e7ekmece", 28.55, 41.02, "right", "center", (-5, 0)),
          ("Gebze", 29.43, 40.80, "right", "top", (-6, -4)),
          ("\u0130zmit", 29.92, 40.77, "right", "bottom", (-6, 6))]
SEAS = [("Sea of Marmara", 28.62, 40.60), ("Black Sea", 28.55, 41.42)]


def dm(v, kind):
    d = int(abs(v)); m = int(round((abs(v) - d) * 60))
    if m == 60:
        d, m = d + 1, 0
    h = ("E" if v >= 0 else "W") if kind == "lon" else ("N" if v >= 0 else "S")
    return f"{d}\u00b0{m:02d}\u2032{h}" if m else f"{d}\u00b0{h}"


# ===========================================================================
FIGW, FIGH = 6.89, 9.40
aspect = 1.0 / np.cos(np.deg2rad(meanlat))
MAPX, MAPW = 0.093, 0.580
# height that exactly matches the data aspect, so add_axes is not shrunk
MAPH = (MAPW * FIGW) * ((N - S) * aspect / (E - W)) / FIGH
fig = plt.figure(figsize=(FIGW, FIGH))
axA = fig.add_axes([MAPX, 0.672, MAPW, MAPH])
axB = fig.add_axes([MAPX, 0.360, MAPW, MAPH])
axC = fig.add_axes([MAPX, 0.125, 0.400, 0.165])
ext = [W, E, S, N]

for ax in (axA, axB):
    ax.set_xlim(W, E); ax.set_ylim(S, N); ax.set_aspect(aspect)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1 / 12))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1 / 12))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: dm(v, "lon")))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: dm(v, "lat")))
    ax.tick_params(which="both", top=True, right=True, direction="out", labelsize=FS_TICK)
    ax.grid(True, which="major", color="white", lw=0.4, alpha=0.75, zorder=5)
    ax.set_axisbelow(False)
axA.tick_params(labelbottom=False)


def annotate(ax):
    ax.contour(np.linspace(W, E, nx), np.linspace(N, S, ny), elev,
               levels=[0], colors="k", linewidths=0.5)
    for name, lo, la, ha, va, off in CITIES:
        ax.plot(lo, la, "s", ms=3.2, mfc="black", mec="white", mew=0.6, zorder=10)
        ax.annotate(name, (lo, la), xytext=off, textcoords="offset points",
                    fontsize=FS_TICK, ha=ha, va=va, zorder=11,
                    path_effects=[matplotlib.patheffects.withStroke(
                        linewidth=2.4, foreground="white")])
    for name, lo, la in SEAS:
        ax.annotate(name, (lo, la), fontsize=FS_TICK, style="italic",
                    color="#1f3b57", ha="center", zorder=7)
    for fl in ("1", "2", "3"):
        g = [s for s in sta if s["flag"] == fl]
        if g:
            ax.scatter([s["lon"] for s in g], [s["lat"] for s in g],
                       c=[s["vs30"] for s in g], cmap=vs_cmap, norm=vs_norm,
                       s=36, marker=FLAG_MK[fl], edgecolors="k", linewidths=0.7,
                       zorder=8,
                       path_effects=[matplotlib.patheffects.withStroke(
                           linewidth=2.2, foreground="white")])


# ---- (a) Vs30 -------------------------------------------------------------
axA.imshow(rgb_vs, extent=ext, origin="upper", aspect=aspect, interpolation="nearest")
annotate(axA)
cax = fig.add_axes([MAPX + MAPW + 0.020, 0.672, 0.019, MAPH])
cb = fig.colorbar(ScalarMappable(norm=vs_norm, cmap=vs_cmap), cax=cax, extend="max")
cb.set_label("$V_{S30}$ (m s$^{-1}$)", fontsize=FS_LAB, labelpad=2)
cb.ax.yaxis.set_major_locator(mticker.MultipleLocator(100))
cb.ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
cb.ax.tick_params(which="major", length=3.4, labelsize=FS_TICK)
cb.ax.tick_params(which="minor", length=1.9)
axA.annotate("N", xy=(E - 0.07, N - 0.08), xytext=(E - 0.07, N - 0.25),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="k"), ha="center",
             fontsize=FS_LAB, fontweight="bold", zorder=9, path_effects=HALO)
fig.text(MAPX, 0.958, "(a) $V_{S30}$ from topographic slope (Wald & Allen, 2007)",
         fontsize=FS_TAG, fontweight="bold", ha="left", va="bottom")

# ---- (b) site classes -----------------------------------------------------
axB.imshow(rgb_cls, extent=ext, origin="upper", aspect=aspect, interpolation="nearest")
annotate(axB)
km = 50.0
dlon = km * 1000 / (111320 * np.cos(np.deg2rad(meanlat)))
x0, y0 = E - 0.09 - dlon, S + 0.06
axB.plot([x0, x0 + dlon], [y0, y0], "k-", lw=2.2, solid_capstyle="butt", zorder=9)
axB.text(x0 + dlon / 2, y0 + 0.022, "50 km", ha="center", fontsize=FS_TICK,
         zorder=9, path_effects=HALO)
cls_leg = [Patch(facecolor=CLS_COL[0], edgecolor="k", lw=0.4,
                 label="EC8 C / NEHRP D  (180\u2013360)"),
           Patch(facecolor=CLS_COL[1], edgecolor="k", lw=0.4,
                 label="EC8 B / NEHRP C  (360\u2013760)"),
           Patch(facecolor=CLS_COL[2], edgecolor="k", lw=0.4,
                 label="NEHRP B ($\\geq$760): relation ceiling;\nEC8 A ($\\geq$800) is not reachable")]
lg = fig.legend(handles=cls_leg, loc="upper left",
                bbox_to_anchor=(MAPX + MAPW + 0.014, 0.360 + MAPH), fontsize=FS_TICK,
                title="Site class ($V_{S30}$, m s$^{-1}$)", title_fontsize=FS_TICK,
                frameon=False, borderpad=0.3, handlelength=1.3,
                handletextpad=0.5, labelspacing=0.4)
lg._legend_box.align = "left"
fig.text(MAPX, 0.650, "(b) EC8 / NEHRP site classes", fontsize=FS_TAG,
         fontweight="bold", ha="left", va="bottom")

# ---- (c) proxy vs station Vs30 -------------------------------------------
axC.set_axisbelow(True)
axC.grid(True, which="major", color="0.88", lw=0.5)
lim = (140, 2100)
axC.plot(lim, lim, color="0.25", lw=1.0, zorder=2)
axC.plot(lim, [v * 1.5 for v in lim], color="0.55", lw=0.8, ls=(0, (4, 2)), zorder=2)
axC.plot(lim, [v / 1.5 for v in lim], color="0.55", lw=0.8, ls=(0, (4, 2)), zorder=2)
for fl in ("1", "2", "3"):
    g = [s for s in sta if s["flag"] == fl]
    if g:
        axC.scatter([s["proxy"] for s in g], [s["vs30"] for s in g], s=32,
                    marker=FLAG_MK[fl], color=FLAG_COL[fl],
                    edgecolors="k", linewidths=0.7, zorder=3,
                    label=f"{fl} \u2013 {FLAG_TXT[fl]} ($n$={len(g)})")
axC.set_xscale("log"); axC.set_yscale("log")
axC.set_xlim(*lim); axC.set_ylim(*lim)
axC.set_xlabel("Slope-proxy $V_{S30}$ (m s$^{-1}$)", fontsize=FS_LAB, labelpad=2)
axC.set_ylabel("Station $V_{S30}$ (m s$^{-1}$)", fontsize=FS_LAB, labelpad=2)
for a in (axC.xaxis, axC.yaxis):
    a.set_major_locator(mticker.FixedLocator([200, 400, 800, 1600]))
    a.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:.0f}"))
    a.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1,
                                           numticks=20))
    a.set_minor_formatter(mticker.NullFormatter())
axC.tick_params(which="major", direction="in", length=3.4, width=0.8,
                labelsize=FS_TICK, top=True, right=True)
axC.tick_params(which="minor", direction="in", length=1.9, width=0.6,
                top=True, right=True)
shape_leg = [Line2D([0], [0], color="none", marker=FLAG_MK[fl], ms=6,
                    mfc=FLAG_COL[fl], mec="k", mew=0.7,
                    label=f"{fl} \u2013 {FLAG_TXT[fl]} "
                          f"($n$={sum(1 for s in sta if s['flag'] == fl)})")
             for fl in ("1", "2", "3") if any(s["flag"] == fl for s in sta)]
lg2 = fig.legend(handles=shape_leg, loc="upper left",
                 bbox_to_anchor=(0.520, 0.285), fontsize=FS_TICK,
                 title="$V_{S30}$ reference flag", title_fontsize=FS_TICK,
                 frameon=False, borderpad=0.3, handlelength=1.0,
                 handletextpad=0.5, labelspacing=0.45)
lg2._legend_box.align = "left"
fig.text(0.520, 0.163,
         f"median station/proxy   {ratio:.2f}\n"
         f"$\\log_{{10}}$ RMSE   {rmse:.2f}  (factor {10 ** rmse:.2f})\n"
         f"within factor 1.5   {within}/{len(sta)}",
         fontsize=FS_TICK, va="top", ha="left", color="0.25", linespacing=1.6)
fig.text(MAPX, 0.297, "(c) Station $V_{S30}$ against the slope proxy",
         fontsize=FS_TAG, fontweight="bold", ha="left", va="bottom")

fig.suptitle("Site characterisation of the Istanbul metropolitan area",
             fontsize=11.5, fontweight="bold", y=0.991)

cap = ("Depiction: (a) $V_{S30}$ from the Wald & Allen (2007) active-tectonic slope relation applied to the GEBCO 2026 grid "
       f"(15 arc-sec; sea masked and shown flat), with the {len(sta)} SMD-TR strong-motion stations of the window drawn on the "
       "same colour scale and marker shape giving the $V_{S30}$ reference flag; (b) the corresponding EC8/NEHRP site classes; "
       "(c) station $V_{S30}$ against the proxy value sampled at the same coordinates, with the 1:1 line and factor-1.5 bounds. "
       "Values above 760 m s$^{-1}$ saturate the colour scale. Station data: SMD-TR flatfile, NHERI DesignSafe PRJ-3950 v3, "
       "DOI 10.17603/ds2-f21x-s189, table Metadata.csv. Software used for producing graph: Python 3.12.3 with Matplotlib "
       f"{matplotlib.__version__} (pyplot, colors.LightSource, ticker, patheffects), NumPy {np.__version__}, SciPy "
       f"{scipy.__version__} and rasterio {rasterio.__version__}. Data: GEBCO 2026 grid. Source: authors.")
fig.text(0.5, 0.006, cap, ha="center", va="bottom", fontsize=8.0,
         color="#333333", wrap=True)

assert_not_dejavu(fig)
fig.savefig(OUT + ".pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT + ".png", dpi=600, bbox_inches="tight", facecolor="white")

from PIL import Image                                                    # noqa: E402
im = Image.open(OUT + ".png")
if im.mode in ("RGBA", "LA"):
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[-1])
    bg.save(OUT + ".png", dpi=(600, 600))
print("wrote", OUT + ".pdf/.png")
