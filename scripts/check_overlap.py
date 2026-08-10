"""
check_overlap.py -- verify the anti-overlap house rules (1, 7, 10) by measurement.

The most common Q1 defect is a legend or label sitting on the data; eyeballing
misses it. These helpers measure it in pixel space after a draw.

API
---
    covers_data(bbox, ax)   -> int   # data points/vertices under bbox (MUST be 0)
    arrow_connects(ann, ax)  -> dict # leader-arrow gaps in points (MUST be small)
    text_bbox(artist, r)    -> Bbox  # text-only box (EXCLUDES an annotation arrow)
    check_figure(fig)       -> list  # every legend/label/annotation vs data + each
                                     # other; returns [(kind, axes_idx, detail), ...]

Programmatic use (preferred):
    from check_overlap import check_figure
    problems = check_figure(fig)
    assert not problems, problems

Shell use on a script that builds a module-level ``fig`` (or calls plt.show/savefig):
    python check_overlap.py my_figure.py
"""
from __future__ import annotations

import sys
import numpy as np
from matplotlib.text import Text


def covers_data(bb, ax):
    """Count plotted data points/vertices of *ax* inside display-space bbox *bb*.

    Checks lines (skipping 2-point guide lines), scatter/poly collections and
    patches (bars, wedges, fills). Returns the hit count; it MUST be 0 for any
    legend, label or annotation text box (rule 10).
    """
    hits = 0

    def c(xy):
        p = ax.transData.transform(np.asarray(xy))
        return int(((p[:, 0] >= bb.x0) & (p[:, 0] <= bb.x1) &
                    (p[:, 1] >= bb.y0) & (p[:, 1] <= bb.y1)).sum())

    for ln in ax.get_lines():
        d = np.column_stack(ln.get_data())
        if d.shape[0] > 2:
            hits += c(d)
    for col in ax.collections:
        off = col.get_offsets()
        if len(off):
            hits += c(off)
        for pth in col.get_paths():
            if len(pth.vertices):
                hits += c(pth.vertices)
    for pch in ax.patches:
        vs = pch.get_path().transformed(pch.get_patch_transform()).vertices
        if len(vs):
            hits += c(vs)
    return hits


def text_bbox(artist, renderer):
    """Text-only display bbox. For an Annotation this EXCLUDES the leader arrow
    (a leader arrow is allowed to cross the data; only its text must be clear)."""
    return Text.get_window_extent(artist, renderer=renderer)


def _overlap(b1, b2):
    return not (b1.x1 <= b2.x0 or b2.x1 <= b1.x0 or
                b1.y1 <= b2.y0 or b2.y1 <= b1.y0)


def check_figure(fig, verbose=True):
    """Scan a figure for overlaps and return a list of problems.

    Each problem is a tuple ``(kind, axes_index, detail)`` where *kind* is
    ``"legend-on-data"``, ``"label-on-data"`` or ``"text-on-text"``. An empty
    list means the figure passes rules 1, 7 and 10.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []

    for i, ax in enumerate(fig.axes):
        text_artists = []   # (name, bbox, is_text_label)

        # axis labels + both possible title positions
        for nm, art in [("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label),
                        ("title", ax.title),
                        ("ltitle", getattr(ax, "_left_title", None)),
                        ("rtitle", getattr(ax, "_right_title", None))]:
            if art is not None and art.get_text():
                text_artists.append((nm, art.get_window_extent(r), True))

        # free-standing text + annotations (measure text box only)
        for t in ax.texts:
            if t.get_text():
                text_artists.append(("annotation", text_bbox(t, r), True))

        # legend
        leg = ax.get_legend()
        if leg is not None:
            lb = leg.get_window_extent(r)
            text_artists.append(("legend", lb, False))
            if covers_data(lb, ax):
                problems.append(("legend-on-data", i,
                                 f"{covers_data(lb, ax)} data pts under legend"))

        # every label/annotation vs data
        for nm, bb, is_label in text_artists:
            if nm in ("legend",):
                continue
            n = covers_data(bb, ax)
            if n:
                problems.append(("label-on-data", i, f"{nm}: {n} data pts under it"))

        # text vs text within this Axes
        for a in range(len(text_artists)):
            for b in range(a + 1, len(text_artists)):
                if _overlap(text_artists[a][1], text_artists[b][1]):
                    problems.append(("text-on-text", i,
                                     f"{text_artists[a][0]} overlaps {text_artists[b][0]}"))

    if verbose:
        if problems:
            print(f"check_overlap: {len(problems)} problem(s) found:")
            for kind, i, detail in problems:
                print(f"  [axes {i}] {kind}: {detail}")
        else:
            print("check_overlap: OK -- no legend/label/text overlaps detected.")
    return problems


def _run_on_file(path):
    """Exec a figure script, then check the resulting figure(s)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = {"__name__": "__check_overlap__", "__file__": path}
    with open(path) as f:
        code = f.read()
    exec(compile(code, path, "exec"), ns)
    fig = ns.get("fig")
    figs = [fig] if fig is not None else list(map(plt.figure, plt.get_fignums()))
    total = 0
    for k, fg in enumerate(figs):
        print(f"--- figure {k} ---")
        total += len(check_figure(fg))
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    _run_on_file(sys.argv[1])


def arrow_connects(ann, ax, tol=4.0, renderer=None):
    """Check that an annotation's leader arrow joins its text to its object.

    House rule 3: an arrow whose head stops in blank space, or whose tail floats
    away from its own label, is worse than no arrow -- the reader cannot tell
    what is being annotated. Both ends are measured in display space:

      head_gap  distance (points) from the arrow head to the nearest plotted
                data point/vertex; the head should land ON the object
      tail_gap  distance (points) from the arrow tail to the text bounding box;
                0 means the tail starts at the label, as it should

    Returns ``{"head_gap": float, "tail_gap": float, "ok": bool}``. ``ok`` is
    True when both gaps are within *tol* points. Use one ``ax.annotate`` call
    with both ``xy`` (object) and ``xytext`` (text) and this passes by
    construction; it fails when the arrow and the caption were drawn as two
    unrelated artists.
    """
    fig = ax.figure
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    dpi = fig.dpi

    head = ax.transData.transform(np.asarray(ann.xy, dtype=float))
    tail = ann.get_transform().transform(np.asarray(ann.get_position(), dtype=float))

    pts = []
    for ln in ax.get_lines():
        d = np.column_stack(ln.get_data())
        if len(d):
            pts.append(ax.transData.transform(d))
    for col in ax.collections:
        off = col.get_offsets()
        if len(off):
            pts.append(ax.transData.transform(np.asarray(off)))
    for pch in ax.patches:
        v = pch.get_path().transformed(pch.get_patch_transform()).vertices
        if len(v):
            pts.append(ax.transData.transform(v) if not pch.get_transform() else v)
    P = np.vstack(pts) if pts else np.empty((0, 2))
    head_gap = (float(np.hypot(*(P - head).T).min()) if len(P) else float("inf"))

    bb = text_bbox(ann, renderer)
    dx = max(bb.x0 - tail[0], 0.0, tail[0] - bb.x1)
    dy = max(bb.y0 - tail[1], 0.0, tail[1] - bb.y1)
    tail_gap = float(np.hypot(dx, dy))

    head_gap *= 72.0 / dpi
    tail_gap *= 72.0 / dpi
    return {"head_gap": head_gap, "tail_gap": tail_gap,
            "ok": head_gap <= tol and tail_gap <= tol}


def check_arrows(fig, tol=4.0):
    """Run arrow_connects over every arrow-bearing annotation in *fig*.

    Returns a list of ``(axes_index, text, head_gap, tail_gap)`` for the arrows
    that fail; an empty list means every leader arrow connects (rule 3).
    """
    from matplotlib.text import Annotation
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for i, ax in enumerate(fig.axes):
        for a in ax.texts:
            if isinstance(a, Annotation) and getattr(a, "arrow_patch", None) is not None:
                res = arrow_connects(a, ax, tol=tol, renderer=r)
                if not res["ok"]:
                    bad.append((i, a.get_text()[:40],
                                res["head_gap"], res["tail_gap"]))
    return bad
