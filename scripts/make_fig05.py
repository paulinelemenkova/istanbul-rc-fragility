#!/usr/bin/env python3
"""
fig05_groundmotions - the selected ground-motion suite.

  (a) 5%-damped acceleration response spectra of the selected records, with the
      suite median, the 16-84 % band and the EC8 elastic target.
  (b) distributions of the intensity measures that characterise the suite.

WHY THIS SCRIPT WAS REWRITTEN
-----------------------------
The previous version SYNTHESISED the suite: it drew lognormal scatter about an
EC8 target with Baker-Jayaram period correlation. Section 4.3 of the manuscript
states that the suite is *recorded* motions from AFAD-TADAS and ESM, and the
reply to Reviewer 3 comment 3 states that no figure in the paper rests on
synthetic data. A synthetic Figure 5 falsifies both. This script therefore
builds the figure from real spectra and REFUSES to produce a publication figure
from simulated data.

DATA REQUIRED (all in the SMD-TR download, DOI 10.17603/ds2-f21x-s189)
---------------------------------------------------------------------
  Intensity Measures/IM_RotD50.csv   5%-damped spectral ordinates per record
  Metadata.csv                       event/station metadata (Mw, Rjb, PGA, Vs30)
  records.txt (optional)             the record identifiers of YOUR 30-record
                                     suite, one per line; without it the script
                                     applies the documented filter below and
                                     prints exactly which records it selected,
                                     so the selection is reproducible either way.

USAGE
-----
  python3 make_fig05.py --header            # print the columns of IM_RotD50.csv
  python3 make_fig05.py                     # build from real data
  python3 make_fig05.py --records records.txt
  python3 make_fig05.py --demo              # layout preview only; the output is
                                            # stamped SYNTHETIC and must not be
                                            # submitted
"""
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator

# Repository layout: this file lives in <repo>/scripts/.
ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Helvetica"],
    "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold",
    "mathtext.cal": "Nimbus Sans:italic", "mathtext.sf": "Nimbus Sans",
    "mathtext.tt": "Nimbus Sans", "mathtext.default": "it",
    "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white",
})

FS_TAG, FS_LAB, FS_TICK = 10.0, 9.5, 8.5


def read_cpt(path):
    """Return the RGB entries of a cpt-city .cpt palette as hex strings."""
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line[0] in "#BFN":
            continue
        parts = line.split()
        if len(parts) >= 2 and "/" in parts[1]:
            r, g, b = (int(v) for v in parts[1].split("/"))
            hexv = f"#{r:02x}{g:02x}{b:02x}"
            if hexv not in out:
                out.append(hexv)
    return out


# ssz/qual-mixed-12 (Statistik Stadt Zurich, CC BY-SA 4.0, cpt-city):
# twelve qualitative colours in six light/dark pairs. One pair is assigned to
# each of the five panels, so no two panels share a hue.
CPT = os.environ.get("QUAL12_CPT",
                     str(Path(__file__).resolve().parent / "qual-mixed-12.cpt"))
Q = read_cpt(CPT) if os.path.exists(CPT) else [
    "#b6cee5", "#5884b3", "#e5b5c5", "#cc6686", "#f2cec1", "#e87b70",
    "#f9ebaa", "#e5cf6c", "#cce5b5", "#91be64", "#b6e3d1", "#5bbe94"]

C_REC, C_BAND, C_MED = "0.68", Q[0], Q[1]          # panel (a): blue pair
C_TGT = Q[3]                                       #            rose accent
C_PGA, C_SA = Q[5], Q[7]                           # (b) coral, gold
C_MW, C_RJB = Q[9], Q[11]                          # (b) green, teal
C_PGA_L, C_SA_L, C_MW_L, C_RJB_L = Q[4], Q[6], Q[8], Q[10]

# The two SMD-TR tables are third-party and are NOT redistributed with this
# repository. Download them from DOI 10.17603/ds2-f21x-s189 (NHERI DesignSafe
# PRJ-3950, v3) and place them in <repo>/data/external/, or point SMDTR_DIR
# at wherever you keep them.
SMDTR = os.environ.get("SMDTR_DIR", str(ROOT / "data" / "external"))
IM_CSV = os.path.join(SMDTR, "IM_RotD50.csv")
META_CSV = os.path.join(SMDTR, "Metadata.csv")
FIGDIR = ROOT / "figures"
OUT = str(FIGDIR / "fig05_groundmotions")

DEMO = "--demo" in sys.argv
HEADER = "--header" in sys.argv
RECFILE = None
if "--records" in sys.argv:
    RECFILE = sys.argv[sys.argv.index("--records") + 1]

# selection filter used when no explicit record list is supplied; every value is
# printed at run time so the selection can be quoted in Section 4.3
SEL = dict(n_records=30, mw_min=5.5, rjb_max=60.0,
           lat=(39.3, 41.9), lon=(26.0, 42.0))
TREF = 1.0        # reference period for reporting Sa in panel (b), in seconds


# ---------------------------------------------------------------------------
def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find(header, want, avoid=()):
    for h in header:
        n = norm(h)
        if all(w in n for w in want) and not any(a in n for a in avoid):
            return h
    return None


def period_columns(header):
    """Columns whose name encodes a spectral period, e.g. 'SA_0.200_(g)'."""
    out = []
    for h in header:
        m = (re.fullmatch(r"t([0-9]*\.?[0-9]+)s_?\(g\)", str(h).strip().lower())
             or re.search(r"(?:sa|psa)[^0-9]*([0-9]+(?:[.,][0-9]+)?)", str(h).lower()))
        if m:
            try:
                out.append((float(m.group(1).replace(",", ".")), h))
            except ValueError:
                pass
    return sorted(out)


def load_real():
    """Build the suite from the SMD-TR flatfile (real recorded motions).

    Selection, applied to Metadata.csv and printed at run time:
      strike-slip focal mechanism (SoF = SS), moment magnitude (Mag_Type = Mw),
      Mw >= 5.5, source-to-site distance <= 60 km, measured Vs30 available, and
      an epicentre inside the North Anatolian fault zone
      (39.3-41.9 N, 26-42 E).
    All 1999 Izmit and Duzce records that pass are retained, since Section 4.3
    names them; the suite is completed to 30 by a deterministic greedy max-min
    spread in standardised (Mw, log R, log Vs30) space, so that magnitude,
    distance and site class are covered as widely as the archive allows.
    """
    import pandas as pd

    for f in (IM_CSV, META_CSV):
        if not os.path.exists(f):
            sys.exit(f"not found: {f}\nSet SMDTR_DIR to the SMD-TR download folder.")

    md = pd.read_csv(META_CSV, low_memory=False, encoding="utf-8-sig")
    md.columns = [c.strip("\ufeff") for c in md.columns]
    md["R"] = pd.to_numeric(md["RJB_(km)"], errors="coerce").fillna(
        pd.to_numeric(md["Repi_(km)"], errors="coerce"))
    md["Vs30"] = pd.to_numeric(md["Vs30_(m/s)"], errors="coerce")
    cand = md[(md.SoF == "SS") & (md.Mag_Type == "Mw") & (md.Mag >= SEL["mw_min"]) &
              md.EQ_Lat.between(*SEL["lat"]) & md.EQ_Lon.between(*SEL["lon"]) &
              (md.R <= SEL["rjb_max"]) & (md.Vs30 > 0)].copy()
    print(f"  candidates: {len(cand)} records / {cand.EQID.nunique()} events")

    if RECFILE:
        keep = {l.strip() for l in open(RECFILE) if l.strip()}
        chosen = cand[cand.WFID.astype(str).isin(keep)]
    else:
        named = cand[cand["Location_(AFAD)"].astype(str).str.contains(
            "IZMIT|DÜZCE|DUZCE|GÖLCÜK|GOLCUK", case=False, na=False)]
        X = np.column_stack([cand.Mag.values, np.log10(cand.R.values + 1.0),
                             np.log10(cand.Vs30.values)])
        X = (X - X.mean(0)) / X.std(0)
        idx = {w: k for k, w in enumerate(cand.WFID.values)}
        picked = [idx[w] for w in named.WFID.values]
        while len(picked) < SEL["n_records"] and len(picked) < len(cand):
            d = np.min(np.linalg.norm(X[:, None, :] - X[None, picked, :], axis=2), axis=1)
            d[picked] = -1
            picked.append(int(np.argmax(d)))
        chosen = cand.iloc[sorted(picked)]

    wf = set(chosen.WFID.astype(str))
    with open(IM_CSV, newline="", encoding="utf-8-sig", errors="replace") as f:
        rdr = csv.DictReader(f)
        hdr = [h.strip("\ufeff") for h in (rdr.fieldnames or []) if h]
        if HEADER:
            print(f"{len(hdr)} columns in {IM_CSV}:")
            for h in hdr:
                print("   ", h)
            sys.exit(0)
        pcols = period_columns(hdr)
        cpga = find(hdr, ["pga"])
        rows = {}
        for r in rdr:
            rid = str(r.get(rdr.fieldnames[0], "")).strip()
            if rid in wf:
                rows[rid] = r
    print(f"  spectral ordinates: {len(pcols)} periods "
          f"({pcols[0][0]:g}-{pcols[-1][0]:g} s); matched {len(rows)} of {len(wf)} records")

    T = np.array([p for p, _ in pcols])
    spec, ids, mws, rjbs, pgas = [], [], [], [], []
    for _, row in chosen.iterrows():
        r = rows.get(str(row.WFID))
        if r is None:
            continue
        try:
            v = np.array([float(r[c]) for _, c in pcols])
            g = float(r[cpga]) if cpga else v[0]
        except (KeyError, TypeError, ValueError):
            continue
        if not np.all(np.isfinite(v)) or np.all(v <= 0):
            continue
        spec.append(v); ids.append(str(row.WFID)); mws.append(row.Mag)
        rjbs.append(row.R); pgas.append(g)
    if len(spec) < 5:
        sys.exit(f"only {len(spec)} usable records; relax SEL or supply --records.")
    Sa = np.array(spec).T
    SaT = np.array([np.interp(TREF, T, Sa[:, j]) for j in range(Sa.shape[1])])
    return T, Sa, np.array(pgas), SaT, np.array(mws), np.array(rjbs), ids


def load_demo():
    """Synthetic stand-in, ONLY for checking the layout. Never publishable."""
    rng = np.random.default_rng(7)
    N = 30
    T = np.logspace(np.log10(0.02), np.log10(4.0), 110)
    ag, S, TB, TC, TD = 0.40, 1.15, 0.20, 0.60, 2.00
    t = T
    tgt = np.where(t <= TB, ag * S * (1 + (t / TB) * 1.5),
                   np.where(t <= TC, ag * S * 2.5,
                            np.where(t <= TD, ag * S * 2.5 * (TC / t),
                                     ag * S * 2.5 * TC * TD / t ** 2)))
    sig = 0.55 + 0.10 * np.clip(np.log(T / 0.3) / np.log(4 / 0.3), 0, 1)
    e = rng.standard_normal((len(T), N))
    e = np.cumsum(e, axis=0) / np.sqrt(np.arange(1, len(T) + 1))[:, None]
    e -= e.mean(axis=1, keepdims=True)
    Sa = tgt[:, None] * np.exp(sig[:, None] * e)
    Mw = np.clip(rng.normal(7.0, 0.32, N), 6.1, 7.7)
    Rjb = np.clip(rng.lognormal(np.log(16), 0.5, N), 5, 45)
    SaT = np.array([np.interp(TREF, T, Sa[:, j]) for j in range(N)])
    return T, Sa, Sa[0], SaT, Mw, Rjb, [f"demo{j:02d}" for j in range(N)]


T, Sa, PGA, SaTref, Mw, Rjb, ids = load_demo() if DEMO else load_real()
N = Sa.shape[1]
median = np.exp(np.log(Sa).mean(axis=1))
p16, p84 = np.percentile(Sa, [16, 84], axis=1)

# EC8 Type 1, ground type C target, drawn for reference only
ag, S, TB, TC, TD = 0.40, 1.15, 0.20, 0.60, 2.00
target = np.where(T <= TB, ag * S * (1 + (T / TB) * 1.5),
                  np.where(T <= TC, ag * S * 2.5,
                           np.where(T <= TD, ag * S * 2.5 * (TC / T),
                                    ag * S * 2.5 * TC * TD / T ** 2)))

print(f"records: {N}")
print(f"  T range {T.min():.3g}-{T.max():.3g} s over {len(T)} periods")
print(f"  PGA {PGA.min():.3g}-{PGA.max():.3g} g | Sa({TREF:.0f} s) median "
      f"{np.median(SaTref):.3g} g")
if np.isfinite(Mw).any():
    print(f"  Mw {np.nanmin(Mw):.2f}-{np.nanmax(Mw):.2f} | "
          f"Rjb {np.nanmin(Rjb):.1f}-{np.nanmax(Rjb):.1f} km")
if not DEMO:
    print("  selection:", SEL if not RECFILE else f"from {RECFILE}")
    print("  ids:", ", ".join(ids))

# ===========================================================================
FIGW, FIGH = 6.89, 3.86
fig = plt.figure(figsize=(FIGW, FIGH))
axS = fig.add_axes([0.082, 0.194, 0.400, 0.682])
IMX, IMW, IMG = 0.540, 0.198, 0.042
axIM = [fig.add_axes([IMX, 0.588, IMW, 0.288]),
        fig.add_axes([IMX + IMW + IMG, 0.588, IMW, 0.288]),
        fig.add_axes([IMX, 0.194, IMW, 0.288]),
        fig.add_axes([IMX + IMW + IMG, 0.194, IMW, 0.288])]

# ---- (a) response spectra -------------------------------------------------
for j in range(N):
    axS.loglog(T, Sa[:, j], color=C_REC, lw=0.5, alpha=0.6, zorder=1)
axS.fill_between(T, p16, p84, color=C_BAND, alpha=0.45, zorder=2, label="16\u201384 %")
axS.loglog(T, median, color=C_MED, lw=1.2, zorder=4, label="suite median")
axS.loglog(T, target, color=C_TGT, lw=1.1, ls=(0, (5, 2)), zorder=5, label="EC8 target")
axS.axvline(TREF, color="0.35", lw=0.8, ls=":", zorder=3)
axS.loglog([], [], color=C_REC, lw=0.8, label=f"records ($n$={N})")
axS.set_xlim(0.02, 4); axS.set_ylim(3e-3, 3)
axS.set_xlabel("Period $T$ (s)", fontsize=FS_LAB, labelpad=2)
axS.set_ylabel("Spectral acceleration $S_a$ (g)", fontsize=FS_LAB, labelpad=2)
axS.set_axisbelow(True)
axS.grid(True, which="major", color="0.88", lw=0.5)
axS.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4])
axS.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "4"])
axS.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
axS.xaxis.set_minor_formatter(plt.NullFormatter())
axS.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
axS.yaxis.set_minor_formatter(plt.NullFormatter())
axS.tick_params(which="major", direction="in", length=3.4, width=0.8,
                labelsize=FS_TICK, top=True, right=True)
axS.tick_params(which="minor", direction="in", length=1.9, width=0.6, top=True, right=True)
h, l = axS.get_legend_handles_labels()
order = [l.index(x) for x in [f"records ($n$={N})", "16\u201384 %", "suite median", "EC8 target"]]
lg = axS.legend([h[i] for i in order], [l[i] for i in order], fontsize=FS_TICK,
                loc="upper left", bbox_to_anchor=(0.0, -0.155), ncol=4, frameon=False,
                borderpad=0.2, handlelength=1.5, handletextpad=0.45,
                columnspacing=1.1)
fig.text(0.082, 0.892, "(a) 5%-damped acceleration response spectra",
         fontsize=FS_TAG, fontweight="bold", ha="left", va="bottom")


def hist(ax, data, label, color, edge=None, ylabel=False):
    d = np.asarray(data, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        ax.set_axis_off(); return
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", color="0.90", lw=0.4)
    ax.hist(d, bins=9, color=color, edgecolor=edge or "white", lw=0.7,
            alpha=0.95, zorder=3)
    ax.axvline(np.median(d), color="#222222", lw=1.0, ls=(0, (4, 2)), zorder=4)
    ax.set_xlabel(label, fontsize=FS_TICK, labelpad=2)
    if ylabel:
        ax.set_ylabel("count", fontsize=FS_TICK, labelpad=2)
    ax.tick_params(which="major", direction="in", length=3.0, width=0.8,
                   labelsize=FS_TICK, top=True, right=True)
    ax.tick_params(which="minor", direction="in", length=1.7, width=0.6,
                   top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))


hist(axIM[0], PGA, "PGA (g)", C_PGA_L, C_PGA, ylabel=True)
hist(axIM[1], SaTref, f"$S_a$($T$ = {TREF:.0f} s) (g)", C_SA_L, C_SA)
hist(axIM[2], Mw, "Moment magnitude $M_w$", C_MW_L, C_MW, ylabel=True)
hist(axIM[3], Rjb, "$R_{JB}$ (km)", C_RJB_L, C_RJB)
fig.text(IMX, 0.892, "(b) Intensity-measure distributions",
         fontsize=FS_TAG, fontweight="bold", ha="left", va="bottom")

fig.suptitle("Selected ground-motion suite", fontsize=11.5, fontweight="bold", y=0.981)

if DEMO:
    fig.text(0.5, 0.5, "SYNTHETIC \u2014 LAYOUT PREVIEW ONLY\nNOT FOR PUBLICATION",
             ha="center", va="center", fontsize=26, color="#c1272d", alpha=0.30,
             rotation=24, fontweight="bold", zorder=50)

FIGDIR.mkdir(parents=True, exist_ok=True)
suffix = "_DEMO" if DEMO else ""
fig.savefig(OUT + suffix + ".pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT + suffix + ".png", dpi=600, bbox_inches="tight", facecolor="white")

from PIL import Image                                                    # noqa: E402
im = Image.open(OUT + suffix + ".png")
if im.mode in ("RGBA", "LA"):
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[-1])
    bg.save(OUT + suffix + ".png", dpi=(600, 600))
print("wrote", OUT + suffix + ".pdf/.png")
if DEMO:
    print("\n*** DEMO OUTPUT IS SYNTHETIC AND MUST NOT BE SUBMITTED ***")
