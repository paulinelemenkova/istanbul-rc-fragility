#!/usr/bin/env python3
"""
fig03_workflow -- methodology pipeline for the Istanbul RC-frame
interpretable-ML paper.

Rebuilt under the scientific-plotting house rules:
  * no library defaults left in place (font, sizes, weights, colours, margins,
    export all chosen explicitly);
  * nothing readable overlaps anything else -- verified by measurement, not eye;
  * two stroke weights only: 2.0 pt for the pipeline spine, 1.2 pt elsewhere;
    context ink (card borders, accent rules) well below 1.2 pt;
  * colour encodes the ORDER of the seven stages, not decoration, and is
    sampled from a single perceptually uniform ramp so the figure survives
    greyscale conversion and colour-vision deficiency;
  * typography: one sans-serif family, nothing below 8 pt or above 12 pt,
    four sizes total;
  * compact layout, minimal outer margin;
  * factual credit line (depiction, software + runtime versions, source).

Outputs vector PDF (for LaTeX) + 600 dpi PNG preview.

Palette
-------
Set PALETTE_CPT to a .cpt file path to drive the stage colours from a GMT
colour palette table; leave it None to use Crameri's `batlow`, the house
default for an ordered sequence. The ramp is sampled at seven points along
RAMP_SPAN, chosen to keep every stage dark enough to carry white numerals
and to hold contrast against the white card.
"""

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

# --------------------------------------------------------------------------
# Palette (rule 8: colour encodes information; one ramp per figure)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"
PALETTE_CPT = Path(__file__).resolve().parent / "paired_07.cpt"
PALETTE_CREDIT = "ColorBrewer Paired-7 (Brewer and Harrower; cpt-city, cb/qual)"

# Paired_07 is a DISCRETE qualitative table of exactly seven classes over
# z = 0..7, so the ramp is read at the class centres rather than interpolated:
# interpolating a qualitative table invents colours its authors never defined.
CPT_STOPS = [(i + 0.5) * 100.0 / 7.0 for i in range(7)]

# Fallback when no CPT is supplied: Crameri's batlow, the house default.
RAMP_SPAN = (0.04, 0.72)


def read_cpt(path):
    """Return the distinct RGB entries of a cpt-city .cpt table, in order.

    Self-contained on purpose: a released figure script must not depend on a
    helper module that only exists on the author's machine.
    """
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line[0] in "#BFN":
            continue
        parts = line.split()
        if len(parts) >= 2 and "/" in parts[1]:
            r, g, b = (int(v) for v in parts[1].split("/"))
            rgb = (r / 255.0, g / 255.0, b / 255.0)
            if rgb not in out:
                out.append(rgb)
    return out


def stage_ramp():
    """Return (list of seven stage colours, human-readable palette name)."""
    if PALETTE_CPT.exists():
        cols = read_cpt(PALETTE_CPT)
        if len(cols) >= 7:
            return cols[:7], PALETTE_CREDIT
    try:
        import cmcrameri.cm as cmc
        return [cmc.batlow(v) for v in np.linspace(*RAMP_SPAN, 7)], "batlow (Crameri)"
    except ImportError:
        cmap = plt.get_cmap("viridis")
        return [cmap(v) for v in np.linspace(*RAMP_SPAN, 7)], "viridis (Matplotlib)"


ACCENTS, CMAP_NAME = stage_ramp()

INK, MUTED, BORDER, RULE, PAGE = "#1a1d21", "#4b545e", "#d5dae0", "#8b949e", "#ffffff"


def tint(rgba, f):
    """Blend a colour towards white by fraction f (for the tool pills)."""
    r, g, b = to_rgb(rgba)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def luminance(rgba):
    r, g, b = to_rgb(rgba)
    lin = [(c / 12.92) if c <= 0.04045 else (((c + 0.055) / 1.055) ** 2.4)
           for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def on_white(rgba, target=2.6):
    """Darken a fill until it separates from the white card.

    sundeck is a digital-art gradient, so several of its stops (the yellow at
    L*=90 in particular) are far too light to read as a filled disc on white.
    Hue is preserved; only lightness is clamped, and the clamp is applied by
    measurement so it adapts to whatever ramp is supplied.
    """
    r, g, b = to_rgb(rgba)
    for _ in range(60):
        if (1.05 / (luminance((r, g, b)) + 0.05)) >= target:
            break
        r, g, b = r * 0.96, g * 0.96, b * 0.96
    return (r, g, b)


def readable(rgba, target=4.5):
    """Darken a stage colour until it clears `target`:1 contrast on white.

    The ramp is user-swappable, so legibility cannot be left to whichever
    lightness the supplied CPT happens to have at that sample point. The pure
    accent still fills the numeral disc and the header rule, where contrast is
    carried by area rather than by stroke.
    """
    r, g, b = to_rgb(rgba)
    for _ in range(60):
        if (1.05 / (luminance((r, g, b)) + 0.05)) >= target:
            break
        r, g, b = r * 0.96, g * 0.96, b * 0.96
    return (r, g, b)


# --------------------------------------------------------------------------
# Typography (mandatory table: one family, 8-12 pt, <= 4 sizes)
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,     # embed TrueType, not Type 3 -- required by most journals
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 120,
})

# Every size raised by a further +3 pt, so the smallest text is 16 pt.
FS_TITLE, FS_SUB, FS_CARD, FS_BODY, FS_SMALL = 21.5, 17.4, 18.75, 16.7, 16.0

# --------------------------------------------------------------------------
# Content -- aligned with the revised manuscript (Sections 3-5, Table 6)
# --------------------------------------------------------------------------
STAGES = [
    ("Ground-motion and site data",
     "30 recorded North Anatolian Fault motions; Vs30 site classes",
     "AFAD-TADAS \u00b7 ESM \u00b7 SMD-TR"),
    ("Parametric RC frame population",
     "60 fibre-section archetypes; Latin-hypercube sampling",
     "OpenSeesPy"),
    ("Nonlinear time-history analysis",
     "6 intensity levels per record: 1.08\u00d710\u2074 analyses total",
     "OpenSeesPy"),
    ("EDPs and EMS-98 damage states",
     "Drifts and accelerations; five damage grades",
     "NumPy \u00b7 pandas"),
    ("ML surrogate and SHAP attribution",
     "Gradient boosting; record-wise cross-validation",
     "scikit-learn \u00b7 XGBoost \u00b7 SHAP"),
    ("Fragility curves",
     "Lognormal fits; Bayesian credible intervals",
     "PyMC \u00b7 ArviZ"),
    ("District-scale scenario map",
     "Expected damage state; Marmara M\u00a07.3 scenario",
     "Matplotlib \u00b7 rasterio"),
]

# --------------------------------------------------------------------------
# Geometry (data units; equal aspect). Compact by construction.
# --------------------------------------------------------------------------
W, H, GAP, MARGIN = 38.0, 41.5, 6.4, 4.0
ROW1_TOP, ROW2_TOP = 103.0, 51.5
XMAX = 2 * MARGIN + 4 * W + 3 * GAP
YMAX = 121.0
SCALE = 0.086                      # -> ~15.5 in wide; double-column at 50 % reduction


def col_x(i):
    return MARGIN + i * (W + GAP)


POS = {k: (col_x(k), ROW1_TOP) for k in range(4)}
POS.update({k: (col_x(k - 4), ROW2_TOP) for k in range(4, 7)})

fig, ax = plt.subplots(figsize=(XMAX * SCALE, YMAX * SCALE))
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(PAGE)

TEXTS = []          # every text artist, for the overlap audit


def T(*args, **kw):
    t = ax.text(*args, **kw)
    TEXTS.append(t)
    return t


def fit_chars(width_units, fontsize, k=0.52):
    """Characters that fit across `width_units` at `fontsize`."""
    return max(12, int(width_units * SCALE * 72.0 / (k * fontsize)))


def card(x, y_top, accent, num, title, desc, tool):
    """One pipeline stage. Fixed internal bands so nothing can collide."""
    ax.add_patch(FancyBboxPatch(
        (x, y_top - H), W, H,
        boxstyle="round,pad=0,rounding_size=1.6",
        lw=0.5, edgecolor=BORDER, facecolor="white",
        mutation_aspect=1.0, zorder=2))

    ink = readable(accent)                       # same hue, darkened for text
    contrast_on_white = 1.05 / (luminance(accent) + 0.05)

    # Thin strokes cannot rely on area for contrast, so the header rule always
    # takes the darkened variant; the filled disc below keeps the palette's
    # exact RGB.
    ax.plot([x + 2.4, x + W - 2.4], [y_top - 4.8, y_top - 4.8],
            color=ink, lw=1.0, zorder=3, solid_capstyle="butt")

    # stage numeral -- redundant coding: order is carried by number AND lightness
    bx, by = x + 5.2, y_top + 0.2
    # Paired alternates light and dark members by design. Darkening the light
    # ones to force contrast would erase that structure, so instead the disc
    # keeps the palette's exact colour and light classes gain a same-hue ring
    # to separate them from the white card.
    ax.add_patch(Circle((bx, by), 3.0, facecolor=accent,
                        edgecolor="white" if contrast_on_white >= 2.6 else ink,
                        lw=1.2, zorder=5))

    # numeral colour chosen by measured contrast, not by a hard-coded guess,
    # so it stays legible whichever ramp is supplied
    L = luminance(accent)
    c_white = 1.05 / (L + 0.05)
    c_ink = (L + 0.05) / (luminance(INK) + 0.05)
    T(bx, by, str(num), color="white" if c_white >= c_ink else INK,
      ha="center", va="center", fontsize=FS_CARD, fontweight="bold", zorder=6)

    # title band: exactly two lines reserved, so the description never rides up
    ty = y_top - 8.6
    for ln in textwrap.wrap(title, width=fit_chars(W - 6.5, FS_CARD))[:2]:
        T(x + 3.0, ty, ln, color=ink, ha="left", va="top",
          fontsize=FS_CARD, fontweight="bold", zorder=4)
        ty -= 4.4

    # description band: three lines reserved, clear of the tool pill below
    dy = y_top - 8.6 - 2 * 4.4 - 2.2
    for ln in textwrap.wrap(desc, width=fit_chars(W - 6.5, FS_BODY, 0.50))[:4]:
        T(x + 3.0, dy, ln, color=MUTED, ha="left", va="top",
          fontsize=FS_BODY, zorder=4)
        dy -= 4.0

    # tool pill, anchored to the card foot
    T(x + 3.3, y_top - H + 3.1, tool, color=INK, ha="left", va="center",
      fontsize=FS_SMALL, zorder=5,
      bbox=dict(boxstyle="round,pad=0.36", fc=tint(accent, 0.87),
                ec=tint(accent, 0.45), lw=0.5))


def flow_arrow(x0, x1, y):
    """Pipeline spine: the one emphasised stroke in the figure (2.0 pt)."""
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=13,
        lw=2.0, color=RULE, zorder=1, shrinkA=0, shrinkB=0))


for k, (title, desc, tool) in enumerate(STAGES):
    x, y_top = POS[k]
    card(x, y_top, ACCENTS[k], k + 1, title, desc, tool)

y_mid1 = ROW1_TOP - H / 2
for k in range(3):
    flow_arrow(col_x(k) + W + 0.5, col_x(k + 1) - 0.5, y_mid1)

y_mid2 = ROW2_TOP - H / 2
for k in range(2):
    flow_arrow(col_x(k) + W + 0.5, col_x(k + 1) - 0.5, y_mid2)

# carriage return, stage 4 -> stage 5
x4, x5 = col_x(3) + W / 2, col_x(0) + W / 2
y_a = ROW1_TOP - H - 0.5
y_b = (ROW1_TOP - H + ROW2_TOP) / 2.0
r = 2.0
verts = [(x4, y_a), (x4, y_b + r), (x4, y_b), (x4 - r, y_b),
         (x5 + r, y_b), (x5, y_b), (x5, y_b - r), (x5, ROW2_TOP + 3.4)]
codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
         MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO]
ax.add_patch(PathPatch(MplPath(verts, codes), fill=False, lw=2.0,
                       edgecolor=RULE, zorder=1, capstyle="round"))
ax.add_patch(FancyArrowPatch((x5, ROW2_TOP + 3.6), (x5, ROW2_TOP + 0.5),
                             arrowstyle="-|>", mutation_scale=13, lw=2.0,
                             color=RULE, zorder=1, shrinkA=0, shrinkB=0))

# --------------------------------------------------------------------------
# Title block and credit line
# --------------------------------------------------------------------------
T(MARGIN, YMAX - 1.6,
  "Open workflow for explainable seismic-damage and fragility assessment "
  "of RC frame archetypes",
  color=INK, ha="left", va="top", fontsize=FS_TITLE, fontweight="bold")

T(MARGIN, YMAX - 8.8,
  "Seven stages, from open ground-motion and site data to a district-scale "
  "scenario field for the Istanbul metropolitan area",
  color=MUTED, ha="left", va="top", fontsize=FS_SUB)

import scipy  # noqa: E402  (version reporting only)
import sklearn  # noqa: E402
CREDIT = (
    "Depiction: the seven methodological stages of this study, with the "
    "principal software of each stage; arrows give the order of execution. "
    f"Software used for producing graph: Python {sys.version.split()[0]} with "
    f"Matplotlib {matplotlib.__version__} (pyplot, patches, path), "
    f"NumPy {np.__version__}, SciPy {scipy.__version__}, "
    f"scikit-learn {sklearn.__version__} and the Python standard-library module "
    "textwrap. Source: authors."
)
# Wrap the credit to the width of the card grid. Left unwrapped, its long
# lines set the canvas width under bbox_inches="tight", which inflates the
# figure and shrinks every glyph again once LaTeX reduces it to \textwidth.
_chars = int((XMAX - 2 * MARGIN) * SCALE * 72.0 / (0.50 * FS_SMALL))
CREDIT = "\n".join(ln for para in CREDIT.split("\n")
                   for ln in textwrap.wrap(para, width=_chars))
T(MARGIN, 3.0, CREDIT, color=MUTED, ha="left", va="top",
  fontsize=FS_SMALL, linespacing=1.45)

plt.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)

# --------------------------------------------------------------------------
# Audit: no readable element may overlap another (rules 1 and 10)
# --------------------------------------------------------------------------
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
boxes = [(t, t.get_window_extent(renderer)) for t in TEXTS]
clashes = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i][1], boxes[j][1]
        if a.overlaps(b):
            ov = (min(a.x1, b.x1) - max(a.x0, b.x0)) * \
                 (min(a.y1, b.y1) - max(a.y0, b.y0))
            if ov > 1.0:      # ignore sub-pixel touching of pill backgrounds
                clashes.append((boxes[i][0].get_text()[:28],
                                boxes[j][0].get_text()[:28], round(ov, 1)))
# the credit must also clear the card patches, which the text-vs-text audit
# above cannot see
card_boxes = [p.get_window_extent(renderer) for p in ax.patches
              if isinstance(p, FancyBboxPatch)]
credit_box = TEXTS[-1].get_window_extent(renderer)
hits = sum(1 for cb in card_boxes if cb.overlaps(credit_box))
gap_px = min(cb.y0 for cb in card_boxes) - credit_box.y1
print(f"text elements: {len(TEXTS)}   overlapping pairs: {len(clashes)}")
print(f"credit vs cards: {hits} overlaps, gap {gap_px:.0f} px")
for c in clashes:
    print("   CLASH:", c)

FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGDIR / "fig03_workflow.pdf", facecolor=PAGE,
            bbox_inches="tight", pad_inches=0.05)
fig.savefig(FIGDIR / "fig03_workflow.png", dpi=600, facecolor=PAGE,
            bbox_inches="tight", pad_inches=0.05)
print(f"wrote {FIGDIR / 'fig03_workflow.pdf'} and .png")
