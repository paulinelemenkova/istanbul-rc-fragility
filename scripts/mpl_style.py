"""
mpl_style.py -- house-style helpers for the scientific-plotting skill.

One-call application of the mandatory house rules: ticks (major + minor on both
axes, all four sides, inward), a thin two-tier grid, colour-blind-safe palettes,
compact layout, and legends placed in empty space. Import only what you need:

    from mpl_style import (set_rc, apply_style, palette, legend_in_gap,
                           tighten, finalize, HALO)

Dependencies: matplotlib + numpy only.

Typical use
-----------
    import matplotlib.pyplot as plt
    from mpl_style import set_rc, apply_style, palette, finalize, HALO

    set_rc()                                  # house rcParams (call once)
    fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    c = palette(3)                            # 3 tab10 colours
    ax.plot(x, y, color=c[0])
    apply_style(ax, xlabel="Year", ylabel="Magnitude $M$")
    finalize(fig, "fig01_demo", dpi=600, outdir="figs")
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator
from matplotlib.patheffects import withStroke

__all__ = ["HALO", "RC", "FONT_STACK", "OKABE_ITO", "MARKERS", "LW_MAIN",
           "LW_SERIES", "panel_tag", "set_rc",
           "ensure_nimbus", "assert_not_dejavu", "palette", "apply_style",
           "legend_in_gap", "direct_label", "tighten", "finalize"]

# Okabe-Ito: the house default categorical palette. Designed to stay
# distinguishable under every form of colour-vision deficiency, so it is
# preferred over tab10 whenever a figure carries more than a couple of series.
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# Redundant coding (see SKILL.md): colour is never the sole identifier, so every
# series also gets a distinct marker shape. Cycle these alongside palette(n).
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# House rule 2: exactly two weights for plotted data. LW_MAIN goes to the one
# series the figure is about (reference, fit, result); LW_SERIES to every other.
# The 2.0/1.2 step is visible without colour, so the ranking survives greyscale
# and colour-vision deficiency. Context ink (grid 0.5, spines 0.8, guides 0.7)
# stays well below LW_SERIES so it cannot be mistaken for data.
LW_MAIN = 2.0
LW_SERIES = 1.2

# White halo for the rare label that must sit on busy content (rule 1 fallback).
HALO = [withStroke(linewidth=1.7, foreground="white")]

# Nimbus Sans is mandatory and DejaVu Sans is forbidden (see SKILL.md,
# "Typography"). Helvetica is the only permitted fallback because it is
# metrically identical, so a macOS run and a Linux run lay out identically.
FONT_STACK = ["Nimbus Sans", "Helvetica"]
_URW_DIRS = (
    "/usr/share/fonts/opentype/urw-base35",
    "/usr/share/fonts/type1/urw-base35",
    "/usr/local/share/fonts/urw-base35",
    "/opt/homebrew/share/fonts",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
)


def ensure_nimbus(strict=True):
    """Register Nimbus Sans with matplotlib and confirm it resolved.

    matplotlib falls back to DejaVu Sans silently when the requested face is
    missing, so a figure can look finished and still be set in the wrong font.
    This registers the URW files directly and then checks what actually
    resolved, raising rather than letting the fallback through.

    Install the font with `apt-get install fonts-urw-base35` (Debian/Ubuntu) or
    `brew install --cask font-urw-base35` (macOS).
    """
    import glob
    from matplotlib import font_manager
    for d in _URW_DIRS:
        for f in glob.glob(os.path.join(d, "NimbusSans-*.otf")) + \
                 glob.glob(os.path.join(d, "NimbusSans-*.ttf")):
            try:
                font_manager.fontManager.addfont(f)
            except Exception:
                pass
    names = {f.name for f in font_manager.fontManager.ttflist}
    have = [n for n in FONT_STACK if n in names]
    if not have:
        msg = ("Nimbus Sans not found and no permitted fallback is installed. "
               "The house style forbids DejaVu Sans, so no figure may be "
               "exported until the font is available. Install it with "
               "`apt-get install fonts-urw-base35` or "
               "`brew install --cask font-urw-base35`.")
        if strict:
            raise RuntimeError(msg)
        print("WARNING: " + msg)
    return have


def assert_not_dejavu(fig):
    """Fail if a rendered figure actually resolved to DejaVu Sans."""
    from matplotlib.font_manager import findfont, FontProperties
    fp = FontProperties()
    fp.set_family("sans-serif")
    resolved = findfont(fp)
    if "DejaVu" in resolved:
        raise RuntimeError(
            f"figure resolved to {resolved}: DejaVu Sans is forbidden by the "
            "house style. Install Nimbus Sans and re-run.")
    return resolved


# Global rcParams that encode the house style; apply once with set_rc().
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.labelpad": 2, "axes.titlepad": 3, "axes.linewidth": 0.8,
    "lines.linewidth": LW_SERIES,          # rule 2: 1.2 default, 2.0 by hand
    "mathtext.default": "regular",
    "legend.framealpha": 0.9, "legend.edgecolor": "0.6",
    "savefig.dpi": 600, "figure.dpi": 120,
}


def set_rc(strict_font=True, **overrides):
    """Apply the house rcParams (call once at the top of a figure script).

    Registers Nimbus Sans first and raises if it is unavailable, so a script
    cannot quietly export a figure set in the forbidden DejaVu Sans. Pass
    ``strict_font=False`` to downgrade that to a warning while drafting.
    Keyword overrides tweak individual entries, e.g.
    ``set_rc(**{'font.size': 9})``.
    """
    ensure_nimbus(strict=strict_font)
    rc = dict(RC)
    rc.update(overrides)
    plt.rcParams.update(rc)
    return rc


def palette(n, name="okabe_ito"):
    """Return *n* colour-blind-safe colours from a categorical palette.

    Default ``okabe_ito`` (8 colours, safe under every colour-vision
    deficiency); ``tab10`` is acceptable for <=5 well-separated series. Past
    about 8 categories no qualitative palette works -- switch to direct
    labelling instead of asking for more colours. For a *continuous* variable
    map the value through viridis directly (``cmap="viridis"``).

    Falls back to the built-in hex list when the running matplotlib is older
    than the release that registered ``okabe_ito`` as a colormap.
    """
    if n > 8:
        raise ValueError(
            f"asked for {n} categorical colours: past ~8 categories colour "
            "stops working as an identifier. Use direct labelling, or colour "
            "only a coarse grouping (see SKILL.md, 'Colour').")
    if name == "okabe_ito":
        try:
            listed = list(plt.get_cmap("okabe_ito").colors)
        except Exception:
            listed = list(OKABE_ITO)
        return [listed[i] for i in range(n)]
    cmap = plt.get_cmap(name)
    listed = getattr(cmap, "colors", None)
    if listed is not None:                       # qualitative listed colormap
        if n > len(listed):
            raise ValueError(f"{name} provides {len(listed)} colours; asked {n}")
        return [tuple(listed[i]) for i in range(n)]
    return [cmap(i / max(n - 1, 1)) for i in range(n)]   # continuous fallback


def apply_style(ax, xlabel=None, ylabel=None, title=None,
                xlog=False, ylog=False, grid="both", minor_grid=False,
                nbins=6):
    """Apply house rules 5 and 6 to one Axes in a single call.

    Ticks: major + minor on both axes, on all four sides, pointing inward, with
    the major locator asked for roughly ``nbins`` human-friendly intervals
    (4-7 major ticks) rather than matplotlib's denser default.

    Grid: light major (lw 0.5, grey 0.85) drawn *under* the data. ``grid``
    selects the orientation, which should be perpendicular to the quantity the
    reader is meant to judge:

      ``"y"``     horizontal lines -- time series and anything read off y
      ``"x"``     vertical lines   -- horizontal bar charts
      ``"both"``  full grid        -- scatter with no primary axis (default)
      ``False``   none             -- then keep visible axis lines so the data
                                     are not left floating

    ``minor_grid=True`` adds an even fainter minor grid (lw 0.3, grey 0.94);
    it is off by default because a second grid tier usually competes with the
    data. The minor *ticks* are always drawn regardless.

    Optional ``xlabel``/``ylabel``/``title`` are set here for convenience
    (``title`` is left-aligned, the house convention for panel captions).
    """
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, loc="left")

    ax.tick_params(which="both", direction="in", top=True, right=True, pad=2)
    ax.tick_params(which="major", length=4.5, width=0.8)
    ax.tick_params(which="minor", length=2.5, width=0.5)

    if xlog:
        ax.xaxis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ylog:
        ax.yaxis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Human-friendly major tick counts; the default locator is often too dense
    # and its labels then collide (rule 1) or carry excess precision.
    if not xlog:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]))
    if not ylog:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]))

    if grid:
        axis = "both" if grid is True else grid
        ax.grid(which="major", axis=axis, lw=0.5, color="0.85")
        if minor_grid:
            ax.grid(which="minor", axis=axis, lw=0.3, color="0.94")
        ax.set_axisbelow(True)          # grid behind the data, always
    return ax


def _data_points_display(ax):
    """Stack all plotted data points/vertices of *ax* in display (pixel) space."""
    chunks = []
    for ln in ax.get_lines():
        d = np.column_stack(ln.get_data())
        if d.shape[0] > 2:                       # skip 2-point guide lines
            chunks.append(ax.transData.transform(d))
    for col in ax.collections:
        off = col.get_offsets()
        if len(off):
            chunks.append(ax.transData.transform(np.asarray(off)))
    return np.vstack(chunks) if chunks else np.empty((0, 2))


def _visual_order(ax, handles, labels):
    """Reorder legend entries top-to-bottom as the series appear on the plot.

    Software sorts legends by plotting or alphabetical order; readers sort by
    which curve is highest. When the two disagree, matching them up costs real
    effort -- and it is what rescues the figure in greyscale, where only the
    ordering survives. Ranks each labelled line by the mean of its last few
    y-values (its right-hand end, where the eye leaves the curve).
    """
    rank = {}
    for ln in ax.get_lines():
        lab = ln.get_label()
        if not lab or lab.startswith("_"):
            continue
        y = np.asarray(ln.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            rank[lab] = float(np.mean(y[-max(1, y.size // 20):]))
    if not rank:
        return handles, labels
    pairs = sorted(zip(handles, labels),
                   key=lambda hl: -rank.get(hl[1], -np.inf))
    return [h for h, _ in pairs], [l for _, l in pairs]


def direct_label(ax, artist, text=None, dx=6, color=None, **kwargs):
    """Label a curve at its right-hand end instead of adding a legend entry.

    A legend forces the reader to hold a colour->name mapping in memory and
    shuttle across the figure; a label next to the curve does not. Places the
    text just past the last finite data point, in the line's own colour by
    default, vertically centred on the line.

    Widen the axes (or set ``ax.margins(x=...)``) so the label has room, and
    check afterwards that it has not landed on another curve.
    """
    x = np.asarray(artist.get_xdata(), dtype=float)
    y = np.asarray(artist.get_ydata(), dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return None
    xe, ye = x[ok][-1], y[ok][-1]
    if text is None:
        text = artist.get_label()
    if color is None:
        color = artist.get_color()
    kw = dict(va="center", ha="left", color=color,
              fontsize=plt.rcParams["legend.fontsize"],
              xytext=(dx, 0), textcoords="offset points")
    kw.update(kwargs)
    return ax.annotate(text, xy=(xe, ye), **kw)


def legend_in_gap(ax, pad=0.02, order="visual", **kwargs):
    """Place the legend in the emptiest corner of *ax* (rule 7).

    Renders once, counts data points falling in each corner quadrant in pixel
    space, and anchors the legend in the corner with the fewest underneath.
    ``order="visual"`` (the default) also reorders the entries to match the
    top-to-bottom order of the curves; pass ``order=None`` to keep the order
    the artists were added in. Extra keyword arguments are forwarded to
    ``ax.legend``. Returns the Legend (or ``None`` if there are no labelled
    artists).

    Prefer ``direct_label`` where the figure can carry labels on the curves --
    the best legend is usually no legend.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if order == "visual":
        handles, labels = _visual_order(ax, handles, labels)
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    P = _data_points_display(ax)

    def score(name):
        x0, y0, w, h = bb.x0, bb.y0, bb.width, bb.height
        xlo, xhi = (x0 + 0.55 * w, x0 + w) if "right" in name else (x0, x0 + 0.45 * w)
        ylo, yhi = (y0 + 0.55 * h, y0 + h) if "upper" in name else (y0, y0 + 0.45 * h)
        if len(P) == 0:
            return 0
        return int(((P[:, 0] >= xlo) & (P[:, 0] <= xhi) &
                    (P[:, 1] >= ylo) & (P[:, 1] <= yhi)).sum())

    best = min(["upper right", "upper left", "lower right", "lower left"], key=score)
    kw = dict(loc=best, framealpha=0.9, edgecolor="0.6",
              borderpad=0.4, handletextpad=0.5)
    kw.update(kwargs)
    leg = ax.legend(handles, labels, **kw)
    leg.get_frame().set_linewidth(0.5)
    return leg


def tighten(fig, w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0):
    """Push a figure toward minimal whitespace (rules 9 and 11).

    For a ``constrained_layout`` figure this shrinks the layout pads; otherwise
    it falls back to a tight ``subplots_adjust``. Anchoring equal-aspect panels
    together is layout-specific -- do it in the caller with
    ``ax.set_anchor("E"/"W"/"N"/"S")`` when centred square axes leave gaps.
    """
    try:
        fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad,
                                        wspace=wspace, hspace=hspace)
    except Exception:
        fig.subplots_adjust(wspace=max(wspace, 0.06), hspace=max(hspace, 0.06))
    return fig


def finalize(fig, name, dpi=600, outdir=".", formats=("pdf", "png"),
             tighten_layout=True):
    """Trim whitespace and export a vector PDF plus a high-dpi PNG.

    Writes ``<outdir>/<name>.pdf`` and ``<outdir>/<name>.png`` (vector first;
    PNG at >=600 dpi for submission). ``name`` should follow the ``figNN_short``
    convention. Returns the list of written paths.
    """
    if tighten_layout:
        tighten(fig)
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in formats:
        p = os.path.join(outdir, f"{name}.{ext}")
        kw = dict(bbox_inches="tight", pad_inches=0.02)
        if ext.lower() in ("png", "jpg", "jpeg", "tif", "tiff"):
            kw["dpi"] = dpi
        fig.savefig(p, **kw)
        paths.append(p)
    return paths


def panel_tag(ax, letter, corner=None, fontsize=10.5, pad=0.018, **kwargs):
    """Place a panel tag, top-left by default (rule 4).

    The top-left is where the eye starts and what most journals expect, so it is
    the default in every panel. The tag moves to the top-right only when the
    top-left is *measurably* occupied -- this renders once, counts data points
    under the prospective tag box, and falls back only if that count is non-zero.

    Pass ``corner="upper left"`` or ``"upper right"`` to pin the choice, which is
    what you want once one panel of a set has had to move: consistency across
    panels beats optimising each one separately.

    Returns the Text artist.
    """
    from check_overlap import covers_data

    def place(loc):
        x, ha = (pad, "left") if loc == "upper left" else (1 - pad, "right")
        kw = dict(transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
                  va="top", ha=ha)
        kw.update(kwargs)
        return ax.text(x, 1 - pad, f"({letter})", **kw)

    if corner:
        return place(corner)

    t = place("upper left")
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    if covers_data(t.get_window_extent(r), ax) == 0:
        return t
    t.remove()                                  # top-left is taken -> top-right
    return place("upper right")
