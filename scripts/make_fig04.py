#!/usr/bin/env python3
"""
fig04_frame_fibre - three linked views of the RC-frame fibre model.

  (a) frame configurations: regular vs soft-storey. The masonry infill is drawn
      in a flat light tone and explicitly keyed as NOT represented in the
      numerical model (Reviewer 1 comment 5; Reviewer 3 comment 2). Only the
      beam-column elements are modelled; infill enters the analysis solely
      through the ground-to-upper-storey stiffness ratio r_k of Eq. (4).
  (b) fibre discretisation of the 400 x 600 mm beam-column section.
  (c) cyclic moment-curvature obtained by integrating the section fibres.

Panel (c) is a genuine fibre computation. The constitutive laws are the ones
the manuscript declares in Section 4.2:
  concrete  - Concrete04 (Mander/Popovics envelope, separate confined and
              unconfined branches, Karsan-Jirsa unload-reload, linear tensile
              softening to zero residual stress);
  steel     - Steel02 (Giuffre-Menegotto-Pinto with isotropic hardening).
The previous version integrated a bilinear-kinematic steel law, which
contradicted Section 4.2 and Reviewer 3 comment 1.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrow
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patheffects import withStroke

from mpl_style import ensure_nimbus, assert_not_dejavu                 # noqa: E402

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
    "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Helvetica"],
    "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold",
    "mathtext.cal": "Nimbus Sans:italic", "mathtext.sf": "Nimbus Sans",
    "mathtext.tt": "Nimbus Sans", "mathtext.default": "it",
    "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white",
})
HALO = [withStroke(linewidth=1.6, foreground="white")]

FS_TAG, FS_LAB, FS_TICK, FS_CRED = 10.0, 9.5, 8.5, 8.0

# Okabe-Ito derived palette
COL_FRAME = "#0072B2"   # blue       - modelled beam-column elements
COL_SOFT  = "#D55E00"   # vermillion - soft-storey ground columns
COL_LOAD  = "#D55E00"   # vermillion - lateral load pattern
COL_HINGE = "#E69F00"   # amber      - ground-storey hinge locations
COL_INF   = "#EDEAE4"   # flat light tone - infill, NOT modelled
COL_INFE  = "#B9B3A8"
COL_COVER = "#D9DCE0"
COL_CORE  = "#F2CE95"
COL_MESH  = "#B07D3A"
LW_MAIN, LW_SER, LW_CTX = 1.5, 1.2, 0.8

# ===========================================================================
# Section geometry (m) - shared by panel (b) drawing and panel (c) fibre model
# ===========================================================================
B, H, COV = 0.40, 0.60, 0.04
fc_u, e0u, ecuu = 30e6, 0.002, 0.004                    # unconfined concrete
Kconf = 1.30
fc_c, e0c, ecuc = Kconf * fc_u, 0.005, 0.020            # confined core
res_c = 0.20 * fc_c                                     # confined residual
Ec = 5000.0 * np.sqrt(fc_u / 1e6) * 1e6                 # Concrete04 Ec (Pa)
ft = 0.33 * np.sqrt(fc_u / 1e6) * 1e6                   # tensile strength (Pa)
et0 = ft / Ec
etu = 8.0 * et0                                         # zero residual tension
Es, fy, bH = 200e9, 420e6, 0.01                         # Steel02
R0, cR1, cR2 = 18.0, 0.925, 0.15                        # GMP curvature params
a1, a2, a3, a4 = 0.02, 1.0, 0.02, 1.0                   # isotropic hardening
Ag = B * H
Naxial = 0.08 * fc_u * Ag                               # axial compression (N)

# ----- concrete fibres (layered through depth) -----------------------------
Ny = 60
dy = H / Ny
yc = (np.arange(Ny) + 0.5) * dy
inside = (yc >= COV) & (yc <= H - COV)
A_conf = np.where(inside, B - 2 * COV, 0.0) * dy
A_unc = B * dy - A_conf
cy = np.concatenate([yc, yc[inside]])
cA = np.concatenate([A_unc, A_conf[inside]])
cis = np.concatenate([np.zeros(Ny, bool), np.ones(inside.sum(), bool)])
cfc = np.where(cis, fc_c, fc_u)
ce0 = np.where(cis, e0c, e0u)
cecu = np.where(cis, ecuc, ecuu)
cres = np.where(cis, res_c, 0.0)
Esec = cfc / ce0
crr = Ec / (Ec - Esec)                                  # Popovics exponent

# ----- steel fibres (top, mid, bottom bar layers) --------------------------
Abar = np.pi / 4 * 0.020 ** 2
sy = np.array([COV, H / 2, H - COV])
sA = np.array([3 * Abar, 2 * Abar, 3 * Abar])


# ===========================================================================
# Concrete04: Mander/Popovics envelope + Karsan-Jirsa unload-reload
# (compression negative)
# ===========================================================================
def conc_env(e):
    """Compressive envelope stress (negative) at strain e (<= 0)."""
    x = np.clip(-e, 0.0, None) / ce0
    pop = cfc * x * crr / (crr - 1.0 + x ** crr)
    xu = cecu / ce0
    su = cfc * xu * crr / (crr - 1.0 + xu ** crr)
    tail = su + (cres - su) * np.clip((-e - cecu) / cecu, 0.0, 1.0)
    return -np.where(-e <= cecu, pop, tail)


def conc_epl(emin):
    """Karsan-Jirsa plastic (residual) strain for a peak compressive emin."""
    X = np.clip(-emin, 0.0, None) / ce0
    lo = 0.145 * X ** 2 + 0.13 * X
    hi = 0.707 * (X - 2.0) + 0.834
    return -ce0 * np.where(X <= 2.0, lo, hi)


def conc_sigma(e, emin, etmax):
    """Stress at strain e given the committed compressive and tensile peaks."""
    epl = conc_epl(emin)
    on_env = e <= emin
    unld = conc_env(emin) * (e - epl) / np.where(emin - epl < -1e-12, emin - epl, -1e-12)
    comp = np.where(on_env, conc_env(e), np.where(e < epl, unld, 0.0))

    def tenv(x):
        rise = Ec * x
        soft = ft * np.clip((etu - x) / (etu - et0), 0.0, 1.0)
        return np.where(x <= et0, rise, soft)

    tens = np.where(e >= etmax, tenv(e),
                    tenv(etmax) * e / np.where(etmax > 1e-12, etmax, 1e-12))
    return np.where(e < 0.0, comp, np.clip(tens, 0.0, None))


# ===========================================================================
# Steel02: Giuffre-Menegotto-Pinto with isotropic hardening
# ===========================================================================
EY = fy / Es


def mp_sigma(eps, st):
    """Stress and updated state for the Menegotto-Pinto law (Filippou 1983)."""
    epsP, sigP, kon, epsr, sigr, e0a, s0a, epsmax, epsmin, epspl = st
    deps = eps - epsP
    if kon == 0:
        if deps >= 0.0:
            kon, epsmax, epsmin = 1, EY, -EY
            e0a, s0a, epspl = EY, fy, 0.0
        else:
            kon, epsmax, epsmin = 2, EY, -EY
            e0a, s0a, epspl = -EY, -fy, 0.0
        epsr = sigr = 0.0
    elif kon == 2 and deps > 0.0:                      # reversal to tension
        kon = 1
        epsr, sigr = epsP, sigP
        epsmin = min(epsmin, epsP)
        d1 = (epsmax - epsmin) / (2.0 * a4 * EY)
        shft = 1.0 + a3 * max(d1, 0.0) ** 0.8
        e0a = (fy * shft - Es * EY * shft * bH - sigr + Es * epsr) / (Es - bH * Es)
        s0a = fy * shft + bH * Es * (e0a - EY * shft)
        epspl = epsmax
    elif kon == 1 and deps < 0.0:                      # reversal to compression
        kon = 2
        epsr, sigr = epsP, sigP
        epsmax = max(epsmax, epsP)
        d1 = (epsmax - epsmin) / (2.0 * a2 * EY)
        shft = 1.0 + a1 * max(d1, 0.0) ** 0.8
        e0a = (-fy * shft + Es * EY * shft * bH - sigr + Es * epsr) / (Es - bH * Es)
        s0a = -fy * shft + bH * Es * (e0a + EY * shft)
        epspl = epsmin

    xi = abs((epspl - e0a) / EY)
    R = R0 * (1.0 - cR1 * xi / (cR2 + xi))
    den = e0a - epsr
    rat = (eps - epsr) / den if abs(den) > 1e-14 else 0.0
    sig = bH * rat + (1.0 - bH) * rat / (1.0 + abs(rat) ** R) ** (1.0 / R)
    sig = sig * (s0a - sigr) + sigr
    return sig, (epsP, sigP, kon, epsr, sigr, e0a, s0a, epsmax, epsmin, epspl)


# ===========================================================================
# Cyclic moment-curvature by fibre integration at constant axial load
# ===========================================================================
def section_forces(phi, eps0, cemin, cetmax, sst):
    ec = eps0 + phi * (H / 2 - cy)
    es = eps0 + phi * (H / 2 - sy)
    sc = conc_sigma(ec, cemin, cetmax)
    ss = np.array([mp_sigma(es[i], sst[i])[0] for i in range(3)])
    N = (sc * cA).sum() + (ss * sA).sum()
    M = (sc * cA * (H / 2 - cy)).sum() + (ss * sA * (H / 2 - sy)).sum()
    return N, M


amps = [0.004, 0.008, 0.014, 0.022, 0.032]
hist = [0.0]
for a in amps:
    for _ in range(2):
        hist += list(a * np.sin(np.linspace(0, 2 * np.pi, 60))[1:])
phis = np.array(hist)

cemin = np.zeros_like(cy)
cetmax = np.zeros_like(cy)
sst = [(0.0, 0.0, 0, 0.0, 0.0, EY, fy, EY, -EY, 0.0) for _ in range(3)]
eps0 = 0.0
Mrec = []
for phi in phis:
    for _ in range(60):                                # Newton on axial strain
        N0, _ = section_forces(phi, eps0, cemin, cetmax, sst)
        r = N0 + Naxial
        if abs(r) < 50.0:
            break
        N1, _ = section_forces(phi, eps0 + 1e-7, cemin, cetmax, sst)
        d = (N1 - N0) / 1e-7
        if abs(d) < 1e3:
            break
        eps0 -= np.clip(r / d, -5e-4, 5e-4)
    ec = eps0 + phi * (H / 2 - cy)
    es = eps0 + phi * (H / 2 - sy)
    cemin = np.minimum(cemin, ec)
    cetmax = np.maximum(cetmax, ec)
    new = []
    for i in range(3):
        s, stt = mp_sigma(es[i], sst[i])
        new.append((es[i], s, stt[2], stt[3], stt[4], stt[5], stt[6], stt[7], stt[8], stt[9]))
    sst = new
    _, M = section_forces(phi, eps0, cemin, cetmax, sst)
    Mrec.append(M)
Mrec = np.array(Mrec) / 1e3                            # kN.m

# ===========================================================================
# FIGURE - journal double-column width (17.5 cm)
# ===========================================================================
# FIGURE - journal double-column width (17.5 cm)
# ---------------------------------------------------------------------------
# The three drawings of the top row share one vertical band: both frames use a
# common y-range, so every storey box renders at exactly the same height, and
# the section box is scaled so that its top and bottom align with the frames.
# Panel (c) spans the same width as (a) + (b) together.
# ===========================================================================
FIGW, FIGH = 6.89, 6.60
LEFT, RIGHT = 0.115, 0.965
ROW_Y, ROW_H = 0.585, 0.305

R_REG = [LEFT, ROW_Y, 0.240, ROW_H]
R_SFT = [0.360, ROW_Y, 0.240, ROW_H]
R_SEC = [0.640, ROW_Y, RIGHT - 0.640, ROW_H]

fig = plt.figure(figsize=(FIGW, FIGH))
ax_reg = fig.add_axes(R_REG)
ax_soft = fig.add_axes(R_SFT)
ax_sec = fig.add_axes(R_SEC)
ax_mphi = fig.add_axes([LEFT, 0.200, RIGHT - LEFT, 0.232])


def equal_box(ax, rect, ylim, xcentre):
    """Set limits so the data aspect equals the axes-box aspect (no shrinking)."""
    yr = ylim[1] - ylim[0]
    xr = yr * (rect[2] * FIGW) / (rect[3] * FIGH)
    ax.set_xlim(xcentre - xr / 2, xcentre + xr / 2)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return xr


# common vertical frame for the two frames; H_FRAME is the taller (soft-storey)
YLIM_FR = (-0.95, 4.75)
H_FRAME = 4.6
LABEL_Y = -0.46                       # both orange captions sit on this level


# ---- (a) frame configurations ---------------------------------------------
def draw_frame(ax, soft, rect):
    nb, ns, bw = 3, 4, 1.0
    heights = [1.6, 1.0, 1.0, 1.0] if soft else [1.0] * 4
    ylev = np.concatenate([[0], np.cumsum(heights)])
    xs = np.arange(nb + 1) * bw
    # infill: flat light tone, dotted outline - schematic only, NOT modelled
    for s in range(ns):
        if soft and s == 0:
            continue
        for bx in range(nb):
            ax.add_patch(Rectangle((xs[bx] + 0.07, ylev[s] + 0.05), bw - 0.14,
                                   heights[s] - 0.1, facecolor=COL_INF,
                                   edgecolor=COL_INFE, ls=(0, (1.5, 1.5)),
                                   lw=0.6, zorder=1))
    for x in xs:                                        # columns
        for s in range(ns):
            c = COL_SOFT if (soft and s == 0) else COL_FRAME
            lw = 3.0 if (soft and s == 0) else 2.4
            ax.plot([x, x], [ylev[s], ylev[s + 1]], color=c, lw=lw,
                    solid_capstyle="round", zorder=3)
    for s in range(1, ns + 1):                          # beams
        ax.plot([xs[0], xs[-1]], [ylev[s], ylev[s]], color=COL_FRAME, lw=2.4,
                solid_capstyle="round", zorder=3)
    ax.plot([xs[0] - 0.2, xs[-1] + 0.2], [0, 0], "k", lw=1.2)
    for x in xs:
        for d in np.linspace(-0.12, 0.12, 4):
            ax.plot([x + d - 0.07, x + d + 0.03], [-0.12, 0], "k", lw=0.7)
    for s in range(1, ns + 1):                          # lateral load pattern
        L = 0.18 + 0.22 * (ylev[s] / H_FRAME)
        ax.add_patch(FancyArrow(xs[0] - 0.18 - L, ylev[s], L, 0, width=0.014,
                                head_width=0.09, head_length=0.09,
                                length_includes_head=True, color=COL_LOAD, zorder=4))
    if soft:
        for x in xs:
            for yy in (ylev[0], ylev[1]):
                ax.add_patch(Circle((x, yy), 0.075, facecolor=COL_HINGE,
                                    edgecolor="k", lw=0.6, zorder=5))
        ax.plot([0, 0.55, 0.62, 0.68, 0.72], ylev, "--", color="0.35",
                lw=LW_CTX, zorder=2)
    txt = "open soft storey" if soft else "infilled ground storey"
    ax.text(np.mean(xs), LABEL_Y, txt, ha="center", va="center",
            fontsize=FS_TICK, color=COL_SOFT, path_effects=HALO)
    equal_box(ax, rect, YLIM_FR, 1.35)


draw_frame(ax_reg, False, R_REG)
draw_frame(ax_soft, True, R_SFT)
ax_reg.set_title("Regular", fontsize=FS_LAB, pad=3)
ax_soft.set_title("Soft-storey", fontsize=FS_LAB, pad=3)
fig.text(LEFT, 0.916, "(a) Frame configurations", fontsize=FS_TAG,
         fontweight="bold", ha="left", va="bottom")

# key: what is modelled and what is not (Reviewer 1 c5 / Reviewer 3 c2)
key = [Line2D([0], [0], color=COL_FRAME, lw=2.4,
              label="beam\u2013column fibre elements (modelled)"),
       Rectangle((0, 0), 1, 1, facecolor=COL_INF, edgecolor=COL_INFE,
                 ls=(0, (1.5, 1.5)), lw=0.6,
                 label="masonry infill: schematic, not modelled"),
       Line2D([0], [0], color=COL_SOFT, lw=3.0,
              label="open ground-storey columns"),
       Line2D([0], [0], color="none", marker="o", ms=6, mfc=COL_HINGE, mec="k",
              mew=0.6, label="ground-storey hinge locations"),
       Line2D([0], [0], color=COL_LOAD, lw=1.6, marker=">", markevery=[-1], ms=5,
              label="inverted-triangular lateral load pattern")]
leg = fig.legend(handles=key, loc="upper left", bbox_to_anchor=(LEFT - 0.005, 0.578),
                 ncol=2, fontsize=FS_TICK, frameon=True, framealpha=1.0,
                 edgecolor="0.75", fancybox=False, borderpad=0.45,
                 handlelength=1.9, handletextpad=0.55, columnspacing=1.4,
                 labelspacing=0.35)
leg.get_frame().set_linewidth(0.5)

# ---- (b) fibre section ----------------------------------------------------
# scale so the section box spans the same vertical band as the frames
YR_SEC = H / (H_FRAME / (YLIM_FR[1] - YLIM_FR[0]))
YLO_SEC = H - (H_FRAME - YLIM_FR[0]) / (YLIM_FR[1] - YLIM_FR[0]) * YR_SEC
XR_SEC = equal_box(ax_sec, R_SEC, (YLO_SEC, YLO_SEC + YR_SEC), 0.227)

ax_sec.add_patch(Rectangle((0, 0), B, H, facecolor=COL_COVER, edgecolor="k",
                           lw=1.0, zorder=1))
cx0, cy0, cw, ch = COV, COV, B - 2 * COV, H - 2 * COV
ax_sec.add_patch(Rectangle((cx0, cy0), cw, ch, facecolor=COL_CORE,
                           edgecolor="none", zorder=2))
ncx, ncy = 6, 9
for i in range(ncx + 1):
    ax_sec.plot([cx0 + i * cw / ncx] * 2, [cy0, cy0 + ch], color=COL_MESH, lw=0.4, zorder=3)
for j in range(ncy + 1):
    ax_sec.plot([cx0, cx0 + cw], [cy0 + j * ch / ncy] * 2, color=COL_MESH, lw=0.4, zorder=3)
ax_sec.add_patch(FancyBboxPatch((cx0, cy0), cw, ch,
                                boxstyle="round,pad=0,rounding_size=0.02", fill=False,
                                edgecolor="#333333", lw=1.2, zorder=4))
bx = [cx0, B / 2, B - cx0]
topy, boty, midy = H - COV, COV, H / 2
for (x, y) in [(x, topy) for x in bx] + [(x, boty) for x in bx] + \
              [(cx0, midy), (B - cx0, midy)]:
    ax_sec.add_patch(Circle((x, y), 0.014, facecolor="k", zorder=5))
akw = dict(arrowstyle="-|>", lw=0.8, color="#333333", shrinkA=2, shrinkB=1)
ax_sec.annotate("cover", xy=(B * 0.93, H * 0.86), xytext=(B + 0.055, H * 0.93),
                fontsize=FS_TICK, ha="left", va="center", arrowprops=akw)
ax_sec.annotate("confined\ncore", xy=(cx0 + cw * 0.78, cy0 + ch * 0.56),
                xytext=(B + 0.055, H * 0.58), fontsize=FS_TICK, ha="left",
                va="center", arrowprops=akw)
ax_sec.annotate("rebar", xy=(bx[-1], boty), xytext=(B + 0.055, H * 0.14),
                fontsize=FS_TICK, ha="left", va="center", arrowprops=akw)
ax_sec.annotate("fibre", xy=(cx0 + cw / ncx * 0.5, cy0 + ch / ncy * 8.5),
                xytext=(-0.162, H * 0.93), fontsize=FS_TICK, ha="left",
                va="center", arrowprops=akw)
ax_sec.annotate("", (0, -0.035), (B, -0.035), arrowprops=dict(arrowstyle="<->", lw=0.7))
ax_sec.text(B / 2, -0.068, f"$b$ = {int(B * 1000)} mm", ha="center", va="top",
            fontsize=FS_TICK)
ax_sec.text(-0.105, H / 2, f"$h$ = {int(H * 1000)} mm", va="center", ha="center",
            rotation=90, fontsize=FS_TICK)
fig.text(R_SEC[0], 0.916, "(b) Fibre section", fontsize=FS_TAG, fontweight="bold",
         ha="left", va="bottom")

# ---- (c) cyclic moment-curvature ------------------------------------------
ax_mphi.set_axisbelow(True)
ax_mphi.grid(True, which="major", color="0.87", lw=0.5)
ax_mphi.axhline(0, color="0.35", lw=0.7, zorder=1)
ax_mphi.axvline(0, color="0.35", lw=0.7, zorder=1)
ax_mphi.plot(phis, Mrec, color=COL_FRAME, lw=LW_SER, zorder=3)
ax_mphi.set_xlabel(r"Curvature $\varphi$ (m$^{-1}$)", fontsize=FS_LAB, labelpad=2)
ax_mphi.set_ylabel("Moment $M$ (kN\u00b7m)", fontsize=FS_LAB, labelpad=2)
ax_mphi.tick_params(which="major", direction="in", length=3.4, width=0.8,
                    labelsize=FS_TICK, top=True, right=True)
ax_mphi.tick_params(which="minor", direction="in", length=1.9, width=0.6,
                    top=True, right=True)
ax_mphi.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_mphi.yaxis.set_minor_locator(AutoMinorLocator(2))
ax_mphi.set_xlim(-0.037, 0.037)
ax_mphi.text(0.012, 0.975,
             "Concrete04 (Mander)\n"
             "+ Steel02 (Giuffr\u00e9\u2013Menegotto\u2013Pinto)\n"
             f"$N$ = {Naxial / 1e3:.0f} kN, constant",
             transform=ax_mphi.transAxes, fontsize=FS_TICK, va="top", ha="left",
             color="0.25", linespacing=1.35)
fig.text(LEFT, 0.443, "(c) Cyclic moment\u2013curvature", fontsize=FS_TAG,
         fontweight="bold", ha="left", va="bottom")

fig.suptitle("RC frame fibre model: configurations, section discretisation and cyclic response",
             fontsize=11.5, fontweight="bold", y=0.988)

cap = ("Depiction: (a) regular and soft-storey frame configurations, drawn to a common vertical scale; the light panels mark "
       "where masonry infill is present in the real stock and are not represented in the numerical model, which admits infill "
       f"only through the ground-to-upper-storey stiffness ratio $r_k$ of Eq. (4); (b) fibre discretisation of the "
       f"{int(B * 1000)}\u00d7{int(H * 1000)} mm beam\u2013column section into unconfined cover, confined core and longitudinal "
       "bars; (c) cyclic moment\u2013curvature obtained by integrating the section fibres under symmetric curvature cycles, with "
       "Concrete04 (Mander/Popovics envelopes, separate confined and unconfined branches, Karsan\u2013Jirsa unloading, linear "
       f"tensile softening to zero residual stress) and Steel02 (Giuffr\u00e9\u2013Menegotto\u2013Pinto, $b$ = {bH}, isotropic "
       f"hardening) at constant axial load $N$ = {Naxial / 1e3:.0f} kN. Software used for producing graph: Python 3.12.3 with "
       f"Matplotlib {matplotlib.__version__} (pyplot, patches, lines, ticker, patheffects) and NumPy {np.__version__}. "
       "Source: authors.")
fig.text(0.5, 0.006, cap, ha="center", va="bottom", fontsize=FS_CRED,
         color="#333333", wrap=True)

assert_not_dejavu(fig)
fig.savefig(FIGDIR / "fig04_frame_fibre.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR / "fig04_frame_fibre.png", dpi=600, bbox_inches="tight",
            facecolor="white")

from PIL import Image                                                  # noqa: E402
_p = str(FIGDIR / "fig04_frame_fibre.png")
_im = Image.open(_p)
if _im.mode in ("RGBA", "LA"):
    _bg = Image.new("RGB", _im.size, "white")
    _bg.paste(_im, mask=_im.split()[-1])
    _bg.save(_p, dpi=(600, 600))

print("wrote fig04 | M %.0f..%.0f kN.m | frame yr %.2f | sec yr %.4f | sec xr %.3f"
      % (Mrec.min(), Mrec.max(), YLIM_FR[1] - YLIM_FR[0], YR_SEC, XR_SEC))
