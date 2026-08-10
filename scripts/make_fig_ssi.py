#!/usr/bin/env python3
"""
fig_ssi_site - Soil-structure interaction and site response.

  (a) substructure model: SDOF superstructure on a rigid footing supported by
      vertical, horizontal (sway) and rocking base impedances drawn as
      spring-dashpot pairs, distinguishing kinematic from inertial interaction.
  (b) strain-dependent modulus-reduction G/Gmax and damping-ratio curves from a
      hyperbolic (Hardin-Drnevich / Darendeli-type) model for two reference
      soils, which parameterise the equivalent-linear site response and the
      strain-compatible soil-spring coefficients.

Design retained from the original; this version fixes the house-style defects:
Nimbus Sans throughout, three text sizes in 8-12 pt, panel tags top-left,
major+minor ticks on both axes of (b), light grid under the data, legend moved
into verified-empty space, callout texts moved off the model and off each other,
Okabe-Ito greyscale-safe colours with redundant marker coding, no hedging text,
compact layout, vector PDF + 600 dpi PNG.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator
from matplotlib.patheffects import withStroke

from mpl_style import ensure_nimbus, assert_not_dejavu           # noqa: E402

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

ensure_nimbus()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold", "mathtext.cal": "Nimbus Sans:italic",
    "mathtext.sf": "Nimbus Sans", "mathtext.tt": "Nimbus Sans", "mathtext.default": "it",
    "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white",
})

HALO = [withStroke(linewidth=1.8, foreground="white")]

# three text sizes only: 10.5 tags/titles, 9.5 labels, 8.5 ticks/annotations
FS_TAG, FS_LAB, FS_TICK, FS_CRED = 9.5, 9.5, 8.5, 8.0

# Okabe-Ito derived palette, one visual language across both panels
STRUCT = "#0072B2"    # blue       - superstructure
SOIL   = "#E4D5AE"    # ochre      - soil half-space fill
SOILE  = "#B39B6A"
SPR    = "#222222"    # near-black - springs, dashpots, anchors
KIN    = "#009E73"    # green      - kinematic interaction
INE    = "#D55E00"    # vermillion - inertial interaction
SAND   = "#1A1A1A"    # near-black - sand curves      (grey L ~ 0.05)
CLAY   = "#D55E00"    # vermillion - clay curves      (grey L ~ 0.44)

LW_MAIN, LW_SER, LW_ARROW, LW_CTX = 1.5, 1.0, 1.0, 0.8


# ---------------------------------------------------------------------------
# spring / dashpot primitives (orientation-independent)
# ---------------------------------------------------------------------------
def spring(ax, p0, p1, coils=6, amp=0.09, lw=LW_ARROW, color=SPR):
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    d = p1 - p0; L = np.hypot(*d); u = d / L; pr = np.array([-u[1], u[0]])
    a = p0 + u * 0.18 * L; b = p1 - u * 0.18 * L; seg = b - a
    pts = [p0, a]
    for i in range(coils):
        t = (i + 0.5) / coils
        pts.append(a + seg * t + pr * amp * (1 if i % 2 == 0 else -1))
    pts += [b, p1]
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=4)


def dashpot(ax, p0, p1, w=0.11, lw=LW_ARROW, color=SPR):
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    d = p1 - p0; L = np.hypot(*d); u = d / L; pr = np.array([-u[1], u[0]])
    A = p0 + u * 0.42 * L; Bc = p0 + u * 0.32 * L; Cc = p1 - u * 0.10 * L
    seg = lambda q, r: ax.plot([q[0], r[0]], [q[1], r[1]], color=color, lw=lw, zorder=4)
    seg(p0, A); seg(A - pr * w * 0.72, A + pr * w * 0.72)
    seg(Bc + pr * w, Cc + pr * w); seg(Bc - pr * w, Cc - pr * w); seg(Cc + pr * w, Cc - pr * w)
    seg(Cc, p1)


def anchor(ax, x0, x1, y, n=7, h=0.13, color=SPR):
    ax.plot([x0, x1], [y, y], color=color, lw=LW_ARROW, zorder=4)
    for xx in np.linspace(x0, x1, n):
        ax.plot([xx, xx - h * 0.8], [y, y - h], color=color, lw=0.7, zorder=4)


def spiral(ax, c, r0, r1, turns=2.6, color=SPR, lw=LW_ARROW):
    th = np.linspace(0, turns * 2 * np.pi, 240); r = np.linspace(r0, r1, 240)
    ax.plot(c[0] + r * np.cos(th), c[1] + r * np.sin(th), color=color, lw=lw, zorder=4)


# ===========================================================================
# built at journal double-column width (17.5 cm = 6.89 in)
fig = plt.figure(figsize=(6.89, 4.60))
axA = fig.add_axes([0.005, 0.205, 0.545, 0.685]); axA.set_aspect("equal"); axA.axis("off")
axB = fig.add_axes([0.665, 0.295, 0.295, 0.595])

# ----------------------- (a) SSI substructure ------------------------------
axA.add_patch(Rectangle((0.13, 0.0), 6.33, 1.9, facecolor=SOIL, edgecolor="none", zorder=0))
axA.add_patch(Rectangle((0.13, 0.0), 6.33, 1.9, facecolor="none", edgecolor=SOILE,
                        hatch="....", lw=0.0, zorder=0))
axA.plot([0.13, 6.46], [1.9, 1.9], color="#7d6f4d", lw=LW_CTX, zorder=1)

# rigid footing
fx0, fx1, fy0, fy1 = 3.15, 4.85, 1.78, 2.08
axA.add_patch(Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, facecolor="#9aa6b2",
                        edgecolor="k", lw=LW_ARROW, zorder=5))
fxc = 0.5 * (fx0 + fx1)

# superstructure: column (stiffness k) + lumped mass m
axA.plot([fxc, fxc], [fy1, 4.35], color=STRUCT, lw=3.0, solid_capstyle="round", zorder=4)
axA.add_patch(Rectangle((fxc - 0.55, 4.35), 1.1, 0.5, facecolor=STRUCT,
                        edgecolor="k", lw=0.7, zorder=5))
axA.text(fxc, 4.60, "$m$", color="white", ha="center", va="center", fontsize=FS_LAB, zorder=6)
axA.text(fxc + 0.30, 3.21, "$k$", color=STRUCT, ha="left", va="center",
         fontsize=FS_LAB, path_effects=HALO, zorder=6)
axA.text(fxc, 5.08, "superstructure (SDOF)", ha="center", fontsize=FS_TICK,
         color=STRUCT, path_effects=HALO)

# lateral DOF u
axA.annotate("", (fxc + 1.02, 4.60), (fxc + 0.55, 4.60),
             arrowprops=dict(arrowstyle="-|>", lw=LW_ARROW, color=INE), zorder=6)
axA.text(fxc + 1.08, 4.60, "$u$", color=INE, ha="left", va="center", fontsize=FS_LAB)

# height h
axA.annotate("", (fxc - 0.40, fy1), (fxc - 0.40, 4.35),
             arrowprops=dict(arrowstyle="<->", lw=0.7, color="k"))
axA.text(fxc - 0.52, 0.5 * (fy1 + 4.35), "$h$", ha="right", va="center",
         fontsize=FS_LAB, path_effects=HALO)

# vertical impedance
spring(axA, (fxc - 0.28, fy0), (fxc - 0.28, 1.18))
dashpot(axA, (fxc + 0.28, fy0), (fxc + 0.28, 1.18))
anchor(axA, fxc - 0.55, fxc + 0.55, 1.18)
axA.text(fxc + 0.05, 0.84, r"Vertical:  $k_v,\,c_v$", ha="center",
         fontsize=FS_TICK, path_effects=HALO)

# horizontal (sway) impedance - to a fixed wall on the right
spring(axA, (fx1, 1.99), (5.90, 1.99), coils=6, amp=0.08)
dashpot(axA, (fx1, 1.74), (5.90, 1.74))
axA.plot([5.90, 5.90], [1.58, 2.13], color=SPR, lw=LW_ARROW)
for yy in np.linspace(1.60, 2.10, 6):
    axA.plot([5.90, 6.05], [yy, yy + 0.12], color=SPR, lw=0.7)
axA.text(4.74, 2.52, "Horizontal (sway):\n" + r"$k_h,\,c_h$", ha="left", va="bottom", linespacing=1.25,
         fontsize=FS_TICK, path_effects=HALO)

# rocking impedance
spiral(axA, (2.82, 1.50), 0.04, 0.18)
axA.add_patch(FancyArrowPatch((3.05, 1.66), (2.60, 1.66), connectionstyle="arc3,rad=0.5",
                              arrowstyle="-|>", lw=LW_ARROW, color=SPR, zorder=4))
axA.text(2.20, 0.84, r"Rocking: $k_\theta,\,c_\theta$", ha="center",
         fontsize=FS_TICK, path_effects=HALO)

# free-field column and input motion
axA.plot([1.05, 1.05], [0.10, 1.90], color=KIN, lw=LW_SER, ls=(0, (4, 2)), zorder=2)
axA.annotate("", (1.52, 0.50), (1.05, 0.50),
             arrowprops=dict(arrowstyle="-|>", lw=LW_ARROW, color=KIN))
axA.text(1.58, 0.50, "$u_g$", color=KIN, fontsize=FS_LAB, va="center", path_effects=HALO)
axA.annotate("", (1.52, 1.74), (1.05, 1.74),
             arrowprops=dict(arrowstyle="-|>", lw=LW_ARROW, color=KIN))
axA.text(1.58, 1.74, "$u_{ff}$", color=KIN, fontsize=FS_LAB, va="center", path_effects=HALO)

# the two callouts: text in blank space, one annotate call each so the arrow
# is tied to its own text (house rule 3)
axA.annotate("Kinematic interaction\n(foundation input motion)",
             xy=(3.16, 2.12), xytext=(0.16, 2.98),
             color=KIN, fontsize=FS_TICK, ha="left", va="center",
             path_effects=HALO,
             arrowprops=dict(arrowstyle="-|>", lw=LW_ARROW, color=KIN,
                             connectionstyle="arc3,rad=-0.22",
                             shrinkA=4, shrinkB=2))
axA.annotate("Inertial interaction\n($m\\mathit{\u00fc}$ through the springs)",
             xy=(3.42, 4.58), xytext=(0.16, 4.24),
             color=INE, fontsize=FS_TICK, ha="left", va="center",
             path_effects=HALO,
             arrowprops=dict(arrowstyle="-|>", lw=LW_ARROW, color=INE,
                             connectionstyle="arc3,rad=-0.12",
                             shrinkA=4, shrinkB=2))
# inertial load path down the column into the base impedances
axA.add_patch(FancyArrowPatch((fxc + 0.62, 4.20), (fxc + 0.62, 2.20),
                              arrowstyle="-|>", lw=LW_ARROW, color=INE,
                              ls=(0, (4, 2)), zorder=3, mutation_scale=11))

axA.set_xlim(0.13, 6.46); axA.set_ylim(0.00, 5.32)
axA.text(0.015, 1.002, "(a) Substructure model: base impedances",
         transform=axA.transAxes, ha="left", va="bottom",
         fontsize=FS_TAG, fontweight="bold")

# ----------------------- (b) G/Gmax and damping ----------------------------
g = np.logspace(-4, 1, 500)                                   # shear strain (%)


def GG(gamma, gref, a=0.92):
    """Hyperbolic modulus-reduction curve (Hardin-Drnevich form)."""
    return 1.0 / (1.0 + (gamma / gref) ** a)


def XI(gg, xmin, Dmax=21.0, k=1.1):
    """Damping ratio rising as the secant modulus degrades (Darendeli-type)."""
    return xmin + Dmax * (1.0 - gg) ** k


soils = [("Sand", 0.035, 1.0, SAND, "o"),
         ("Clay", 0.13, 2.6, CLAY, "s")]

axB.set_axisbelow(True)
axB.grid(True, which="major", color="0.86", lw=0.5, zorder=0)
axB2 = axB.twinx()
axB2.set_zorder(axB.get_zorder() - 1)
axB.patch.set_visible(False)

handles = []
for name, gref, xmin, c, mk in soils:
    gg = GG(g, gref); xi = XI(gg, xmin)
    axB.semilogx(g, gg, color=c, lw=LW_MAIN, zorder=3,
                 marker=mk, markevery=95, ms=3.6, mfc="white", mew=1.0)
    axB2.semilogx(g, xi, color=c, lw=LW_SER, ls=(0, (4, 2)), zorder=3)
    handles.append(Line2D([0], [0], color=c, lw=LW_MAIN, marker=mk, ms=3.6,
                          mfc="white", mew=1.0, label=name))
handles += [Line2D([0], [0], color="0.30", lw=LW_MAIN, label=r"$G/G_{\max}$"),
            Line2D([0], [0], color="0.30", lw=LW_SER, ls=(0, (4, 2)), label=r"$\xi$")]

axB.set_xlim(1e-4, 1e1); axB.set_ylim(0, 1.02); axB2.set_ylim(0, 26)
axB.set_xlabel(r"Shear strain $\gamma$ (%)", fontsize=FS_LAB, labelpad=2)
axB.set_ylabel(r"$G/G_{\max}$  (solid lines)", fontsize=FS_LAB, labelpad=2)
axB2.set_ylabel(r"Damping ratio $\xi$ (%)  (dashed)", fontsize=FS_LAB, labelpad=3)

for a_ in (axB, axB2):
    a_.tick_params(which="major", direction="in", length=3.4, width=0.8, labelsize=FS_TICK)
    a_.tick_params(which="minor", direction="in", length=1.9, width=0.6)
axB.tick_params(which="both", top=True)
axB.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
axB.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
axB.set_yticks(np.arange(0, 1.01, 0.2))
axB.yaxis.set_minor_locator(AutoMinorLocator(2))
axB2.set_yticks(np.arange(0, 26, 5))
axB2.yaxis.set_minor_locator(AutoMinorLocator(2))

# legend in the mid-left band, which both quantities leave empty
# direct-labelling legend, placed in the largest verified-empty region of the
# panel (axes fraction x 0.00-0.34, y 0.16-0.82); solid/dashed is carried by the
# two y-axis labels, so only the two soils need a key
leg = axB.legend(handles=handles, loc="center left", bbox_to_anchor=(0.015, 0.50),
                 fontsize=FS_TICK, frameon=False, borderpad=0.2,
                 handlelength=1.4, handletextpad=0.45, labelspacing=0.45)
leg.set_zorder(6)
axB.text(0.015, 1.002, "(b) Modulus reduction and damping", transform=axB.transAxes,
         ha="left", va="bottom", fontsize=FS_TAG, fontweight="bold")

# ----------------------- title and credit line -----------------------------
fig.suptitle("Soil\u2013structure interaction and site response",
             fontsize=11.5, fontweight="bold", y=0.988)

cap = ("Depiction: (a) substructure model of a rigid shallow footing on vertical, horizontal (sway) and rocking base "
       "impedances (spring\u2013dashpot pairs), separating kinematic interaction (foundation input motion) from inertial "
       "interaction ($m\\mathit{\u00fc}$ transmitted through the springs); (b) modulus reduction $G/G_{\\max}$ (solid, left axis) and "
       "damping ratio $\\xi$ (dashed, right axis) from a hyperbolic Hardin\u2013Drnevich/Darendeli-type model, computed for a "
       "sand ($\\gamma_{\\mathrm{r}}=0.035$ %, $\\xi_{\\min}=1.0$ %) and a clay of PI $\\approx$ 30 "
       "($\\gamma_{\\mathrm{r}}=0.13$ %, $\\xi_{\\min}=2.6$ %) with $\\xi_{\\max}=21$ %; these curves set the strain-compatible "
       "soil-spring coefficients and the equivalent-linear site response. Software used for producing graph: Python 3.12.3 "
       f"with Matplotlib {matplotlib.__version__} (pyplot, patches, lines, ticker, patheffects), NumPy {np.__version__} and "
       f"SciPy {scipy.__version__}. Source: authors.")
fig.text(0.5, 0.006, cap, ha="center", va="bottom", fontsize=FS_CRED, color="#333333", wrap=True)

assert_not_dejavu(fig)
fig.savefig(FIGDIR / "fig_ssi_site.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR / "fig_ssi_site.png", dpi=600, bbox_inches="tight", facecolor="white")

# flatten the PNG to opaque RGB: a transparent background renders black in some
# PDF/print workflows
from PIL import Image                                            # noqa: E402
_p = str(FIGDIR / "fig_ssi_site.png")
_im = Image.open(_p)
if _im.mode in ("RGBA", "LA"):
    _bg = Image.new("RGB", _im.size, "white")
    _bg.paste(_im, mask=_im.split()[-1])
    _bg.save(_p, dpi=(600, 600))
print("wrote fig_ssi_site (pdf + png @600 dpi, RGB)")
