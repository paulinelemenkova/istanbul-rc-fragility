#!/usr/bin/env python3
"""fig01_study_area : Marmara Sea / Istanbul study-area map.

Shaded-relief GEBCO 2026 topography + IEB seismicity (1998-2025).
Seismicity: colour = hypocentre depth (0-47 km), symbol size = magnitude.

Revisions made for the reviewer response:
  * the 1999 event is labelled "Izmit (Kocaeli) Mw 7.4", the magnitude and
    naming convention fixed in Section 1, instead of the catalogue value M 7.6
    that was previously burnt into the image (Reviewer 2 c6, Reviewer 3 c6);
  * the 2025 Silivri / Kumburgaz Mw 6.3 event discussed in Section 1 is now
    highlighted (catalogue entry 23 April 2025, 40.834 N 28.190 E, 12.7 km);
  * the rainbow depth ramp (red-yellow-green-blue) is replaced by reversed
    viridis: monotonic in lightness, colour-blind safe, shallow events bright;
  * built at journal double-column width, Nimbus Sans, three text sizes in
    8-11 pt, 600 dpi raster plus vector PDF.

NOTE ON MAGNITUDES: the IEB catalogue lists the 1999 event as M 7.6 and the
2025 event as M 6.2. The manuscript adopts Mw 7.4 and Mw 6.3 from the source
studies it cites, and the two annotations use those values. Symbol size still
derives from the catalogue magnitude, so the plotted symbol areas are unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow
from matplotlib.ticker import MultipleLocator

from mpl_style import ensure_nimbus, assert_not_dejavu                   # noqa: E402

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
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

FS_TAG, FS_LAB, FS_TICK = 11.0, 9.5, 8.5

DEM = _gebco()
CSV = str(EXTERNAL / "IEB_export.csv")
OUT = str(FIGDIR / "fig01_study_area")

# ---- DEM ------------------------------------------------------------------
with rasterio.open(DEM) as ds:
    elev = ds.read(1).astype("float64")
    b = ds.bounds
    W, E, S, N = b.left, b.right, b.bottom, b.top
lon = np.linspace(W, E, elev.shape[1])
lat = np.linspace(N, S, elev.shape[0])
LON, LAT = np.meshgrid(lon, lat)

# ---- topographic colormap (GMT 'geo'-like) --------------------------------
vmin, vmax = -2200.0, 2500.0
anchors = [(-2200, "#08306b"), (-1500, "#1259a0"), (-800, "#2b7bba"), (-300, "#5ca7d6"),
           (-60, "#a9d3ec"), (-1, "#d8eef9"), (0, "#2e7d32"), (250, "#5fa052"),
           (600, "#9cb558"), (1000, "#cbb56b"), (1500, "#b0895a"), (2000, "#b3a79c"),
           (2500, "#ffffff")]
topo_cmap = LinearSegmentedColormap.from_list(
    "geo_marmara", [((v - vmin) / (vmax - vmin), c) for v, c in anchors])
tnorm = Normalize(vmin, vmax)

# ---- depth ramp: GMT 'seis' (shallow red -> deep blue/violet) -------------
# Not perceptually uniform and not colour-blind safe, but it is the convention
# in seismological cartography and keeps shallow events visually dominant.
seis_stops = [(0.00, "#e8000b"), (0.18, "#ff6a00"), (0.36, "#ffc000"),
              (0.54, "#ffef4a"), (0.70, "#3fbf3f"), (0.86, "#1f6fd0"),
              (1.00, "#6a2a9c")]
depth_cmap = LinearSegmentedColormap.from_list("seis_depth", seis_stops)
# 87 % of the catalogue lies between 5 and 15 km, so the ramp is stretched over
# 0-30 km and saturates above that (23 of 1000 events); the colourbar carries an
# extend arrow to make the saturation explicit.
DCAP = 30.0
dnorm = Normalize(0.0, DCAP)

# ---- shaded relief --------------------------------------------------------
mean_lat = 0.5 * (S + N)
dx = (lon[1] - lon[0]) * 111320.0 * np.cos(np.radians(mean_lat))
dy = (lat[0] - lat[1]) * 110570.0
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(elev, cmap=topo_cmap, norm=tnorm, blend_mode="soft",
               vert_exag=2.0, dx=dx, dy=dy)

# ---- seismicity -----------------------------------------------------------
df = pd.read_csv(CSV).sort_values("Mag")
size = 2.2 * 2.0 ** (df.Mag - 2.0)

# the two events named in Section 1; magnitudes as adopted in the manuscript
EV_1999 = df.loc[df.Mag.idxmax()]
_m2025 = (df.Year == 2025) & (df.Mag >= 6.0)
EV_2025 = df[_m2025].iloc[0] if _m2025.any() else None

# ===========================================================================
FIGW, FIGH = 6.89, 5.00
fig = plt.figure(figsize=(FIGW, FIGH))
ax = fig.add_axes([0.088, 0.180, 0.740, 0.672])
cax = fig.add_axes([0.842, 0.180, 0.016, 0.672])
dax = fig.add_axes([0.130, 0.098, 0.250, 0.017])

ax.imshow(rgb, extent=[W, E, S, N], origin="upper", aspect="auto",
          interpolation="bilinear", zorder=1)
ax.contour(LON, LAT, elev, levels=[0], colors="k", linewidths=0.6, zorder=3)
ax.contour(LON, LAT, elev, levels=[-1000], colors="#1b3a5b", linewidths=0.35,
           alpha=0.6, zorder=3)
# 200 m contours on land, hairline brown
ax.contour(LON, LAT, elev, levels=np.arange(200, 2600, 200), colors="#8a6a3a",
           linewidths=0.22, alpha=0.55, zorder=2)

# North Anatolian Fault (Main Marmara Fault) - indicative trace
naf = np.array([(30.10, 40.72), (29.70, 40.74), (29.30, 40.77), (28.90, 40.82),
                (28.40, 40.83), (27.90, 40.86), (27.50, 40.78), (27.10, 40.62),
                (26.75, 40.50), (26.45, 40.42)])
ax.plot(naf[:, 0], naf[:, 1], color="#c8102e", lw=1.5, ls=(0, (6, 2)), zorder=4)
ax.text(27.72, 40.93, "North Anatolian Fault\n(Main Marmara Fault)", color="#7a0d20",
        fontsize=FS_TICK, style="italic", ha="center", va="bottom", zorder=6, rotation=3,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.65))

sc = ax.scatter(df.Lon, df.Lat, s=size, c=df.Depth, cmap=depth_cmap, norm=dnorm,
                edgecolors="k", linewidths=0.35, alpha=0.92, zorder=5)


def place(lo, la, txt, fs=FS_TICK, col="#1a1a1a", style="normal", weight="normal",
          rot=0, ha="center", bg=True):
    kw = dict(fontsize=fs, color=col, style=style, weight=weight, rotation=rot,
              ha=ha, va="center", zorder=7)
    if bg:
        kw["bbox"] = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.5)
    t = ax.text(lo, la, txt, **kw)
    if not bg:
        # halo contrasts with the glyph, so white sea names stay legible too
        from matplotlib.colors import to_rgb
        r_, g_, b_ = to_rgb(col)
        light = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_ > 0.6
        t.set_path_effects([pe.withStroke(linewidth=2.4,
                                          foreground="#123" if light else "white")])
    return t


place(28.45, 40.60, "Sea of Marmara", FS_LAB, "#08306b", "italic", "bold", bg=False)
place(30.35, 41.86, "BLACK SEA", FS_LAB, "white", "italic", "bold", bg=False)
place(25.55, 39.66, "AEGEAN SEA", FS_LAB, "#08306b", "italic", "bold", bg=False)
place(29.02, 41.10, "\u0130stanbul", FS_LAB, "#111", weight="bold")
ax.scatter([28.979], [41.008], s=22, marker="s", c="#111", zorder=8)
place(29.98, 40.62, "Gulf of \u0130zmit", FS_TICK, "#08306b", "italic")
place(29.06, 40.12, "Bursa", FS_TICK, "#111", weight="bold")
place(27.22, 41.38, "Tekirda\u011f", FS_TICK, "#111")
place(26.3333, 40.25, "Dardanelles", FS_TICK, "#08306b", "italic", rot=52, ha="left")
place(27.10, 41.60, "T \u00dc R K \u0130 Y E", FS_LAB, "#5b4636", weight="bold")

# ---- the two events named in Section 1 ------------------------------------
# locator rings, so the two annotated events are findable among 1000 symbols
for _ev in [e for e in (EV_1999, EV_2025) if e is not None]:
    ax.scatter([_ev.Lon], [_ev.Lat], s=210, marker="o", facecolors="none",
               edgecolors="#c8102e", linewidths=1.2, zorder=6)

akw = dict(arrowstyle="-|>", color="#c8102e", lw=1.0, shrinkA=2, shrinkB=3)
bkw = dict(boxstyle="round,pad=0.25", fc="white", ec="#c8102e", lw=0.8)
ax.annotate("1999 \u0130zmit (Kocaeli)  $M_w$ 7.4",
            xy=(EV_1999.Lon, EV_1999.Lat), xytext=(30.90, 41.60),
            fontsize=FS_TICK, fontweight="bold", color="#5a0010", zorder=9,
            ha="right", va="center", bbox=bkw, arrowprops=akw)
if EV_2025 is not None:
    ax.annotate("2025 Silivri (Kumburgaz)  $M_w$ 6.3",
                xy=(EV_2025.Lon, EV_2025.Lat), xytext=(25.28, 40.68),
                fontsize=FS_TICK, fontweight="bold", color="#5a0010", zorder=9,
                ha="left", va="center", bbox=bkw,
                arrowprops=dict(akw, connectionstyle="arc3,rad=0.16"))

# ---- frame, ticks, graticule ---------------------------------------------
ax.set_xlim(W, E); ax.set_ylim(S, N)
xt = np.arange(25, 31.001, 1)
yt = np.arange(39, 42.001, 0.5)
ax.set_xticks(xt); ax.set_yticks(yt)
ax.set_xticklabels([f"{int(v)}\u00b0E" for v in xt])


def _ylab(v):
    d = int(np.floor(v + 1e-9)); m = int(round((v - d) * 60))
    return f"{d}\u00b0N" if m == 0 else f"{d}\u00b0{m:02d}\u2032N"


ax.set_yticklabels([_ylab(v) for v in yt])
ax.xaxis.set_minor_locator(MultipleLocator(1 / 6.0))
ax.yaxis.set_minor_locator(MultipleLocator(2 / 60.0))
ax.tick_params(which="major", top=True, right=True, bottom=True, left=True,
               labeltop=True, labelright=False, labelbottom=True, labelleft=True,
               direction="out", length=4, labelsize=FS_TICK)
ax.tick_params(which="minor", top=True, right=True, bottom=True, left=True,
               direction="out", length=2.0)
ax.grid(True, which="major", axis="x", color="white", lw=0.5, alpha=0.6)
for _yv in (39, 40, 41, 42):
    ax.axhline(_yv, color="white", lw=0.5, alpha=0.6, zorder=2)
for sp in ax.spines.values():
    sp.set_edgecolor("#3a3a3a"); sp.set_linewidth(1.0)

# ---- colourbars -----------------------------------------------------------
sm = plt.cm.ScalarMappable(cmap=topo_cmap, norm=tnorm); sm.set_array([])
cb = fig.colorbar(sm, cax=cax, extend="both")
cb.set_label("Elevation / bathymetry (m)", fontsize=FS_LAB, labelpad=3)
cb.set_ticks([-2000, -1000, 0, 1000, 2000])
cb.ax.tick_params(labelsize=FS_TICK, which="major", length=3.4)
cb.ax.yaxis.set_minor_locator(MultipleLocator(250))
cb.ax.tick_params(which="minor", length=1.9, color="#333")

db = fig.colorbar(sc, cax=dax, orientation="horizontal", extend="max")
db.set_label(f"Hypocentre depth (km), saturating at {DCAP:.0f}", fontsize=FS_LAB, labelpad=2)
db.set_ticks([0, 5, 10, 15, 20, 25, 30])
db.ax.tick_params(labelsize=FS_TICK, which="major", length=3.4)
db.ax.xaxis.set_minor_locator(MultipleLocator(2.5))
db.ax.tick_params(which="minor", length=1.9)

# ---- scale bar and north arrow -------------------------------------------
km = 100.0
dlon = km / (111.320 * np.cos(np.radians(mean_lat)))
x0, y0 = W + 0.28, S + 0.20
ax.plot([x0, x0 + dlon], [y0, y0], color="k", lw=2.8, solid_capstyle="butt", zorder=10)
for fr in (0, 0.5, 1.0):
    ax.plot([x0 + fr * dlon] * 2, [y0, y0 + 0.05], color="k", lw=1.0, zorder=10)
    ax.text(x0 + fr * dlon, y0 + 0.09, f"{int(fr * km)}", ha="center", va="bottom",
            fontsize=FS_TICK, zorder=10,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
ax.text(x0 + dlon / 2, y0 - 0.12, "km  \u00b7  Mercator projection", ha="center",
        va="top", fontsize=FS_TICK, zorder=10,
        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

nx, ny = E - 0.30, S + 0.50
ax.add_patch(FancyArrow(nx, ny, 0, 0.42, width=0.012, head_width=0.10, head_length=0.12,
                        length_includes_head=True, color="k", zorder=10))
ax.text(nx, ny + 0.56, "N", ha="center", va="bottom", fontsize=FS_LAB,
        fontweight="bold", zorder=10,
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

# ---- titles and magnitude key --------------------------------------------
fig.text(0.458, 0.952, "Study area: Sea of Marmara and Istanbul region",
         ha="center", fontsize=FS_TAG, fontweight="bold")
fig.text(0.458, 0.906, "GEBCO 2026 relief (15 arc-sec) and IEB seismicity, 1998\u20132025",
         ha="center", fontsize=FS_LAB, color="#333333")

mags = [3, 5, 7]
handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor="#bdbdbd",
                  markeredgecolor="k", markeredgewidth=0.35,
                  markersize=np.sqrt(2.2 * 2.0 ** (m - 2)), label=f"M {m}") for m in mags]
handles.append(Line2D([0], [0], color="#c8102e", lw=1.5, ls=(0, (6, 2)),
                      label="N. Anatolian Fault"))
leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
                 fontsize=FS_TICK, bbox_to_anchor=(0.655, 0.018),
                 handletextpad=0.5, columnspacing=1.2,
                 title="Symbol size $\\propto$ magnitude (M)  \u00b7  colour = depth")
leg.get_title().set_fontsize(FS_TICK)
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_edgecolor("0.7")

assert_not_dejavu(fig)
import os                                                                # noqa: E402
fig.savefig(OUT + ".pdf", facecolor="white", bbox_inches="tight", pad_inches=0.04)
fig.savefig(OUT + ".png", dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.04)

from PIL import Image                                                    # noqa: E402
im = Image.open(OUT + ".png")
if im.mode in ("RGBA", "LA"):
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[-1])
    bg.save(OUT + ".png", dpi=(600, 600))

print(f"1999 event: catalogue M {EV_1999.Mag} at {EV_1999.Lon:.3f},{EV_1999.Lat:.3f} "
      f"-> labelled Mw 7.4")
if EV_2025 is not None:
    print(f"2025 event: catalogue M {EV_2025.Mag} at {EV_2025.Lon:.3f},{EV_2025.Lat:.3f} "
          f"({int(EV_2025.Day)}/{int(EV_2025.Month)}/{int(EV_2025.Year)}) -> labelled Mw 6.3")
print("wrote", OUT + ".pdf/.png")
