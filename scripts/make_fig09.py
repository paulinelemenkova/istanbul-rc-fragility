#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig09_fragility -- simulation-derived seismic fragility curves by frame class
(regular vs soft-storey) for the four EMS-98 damage states.

Method
------
The database is a MULTIPLE-STRIPE design: 900 analyses per class at each of six
discrete Sa(T1) levels (0.05, 0.15, 0.35, 0.65, 1.10, 2.00 g).  For each class
and damage state the number of analyses exceeding the drift threshold is counted
at every stripe, and a lognormal fragility

    P(DS >= ds | Sa) = Phi( ln(Sa/theta) / beta )

is fitted to those six binomial counts by maximum likelihood (the standard
estimator for stripe analysis; Baker, Earthq. Spectra 31, 2015).  Open circles
are the observed exceedance fractions, solid lines the fitted lognormals.
A Bayesian posterior over (theta, beta) with flat priors is evaluated on a grid
and its 95% credible intervals are printed for the fragility table.

Design and palette follow the previous version of this figure: two panels
sharing the y axis, four damage states coloured with turbo sampled over
0.12-0.88 (light -> collapse), fitted curves solid, empirical points as open
circles, legends in the same corners.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.ticker import AutoMinorLocator           # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402
from scipy.special import ndtr                           # noqa: E402
from scipy.optimize import minimize                      # noqa: E402

from mpl_style import set_rc, assert_not_dejavu          # noqa: E402
from check_overlap import check_figure                   # noqa: E402

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

CSV = str(RESULTS / "simulation_database.csv")
NAME = "fig09_fragility"
DS = [("Light", 0.005), ("Moderate", 0.012), ("Heavy", 0.030), ("Collapse", 0.060)]
DS_COL = plt.get_cmap("turbo")(np.linspace(0.12, 0.88, len(DS)))
CLASSES = [("Regular", False), ("Soft-storey", True)]
EPS = 1e-12

# ------------------------------------------------------------------ data ----
d = pd.read_csv(CSV, keep_default_na=False, na_values=[""])
LEVELS = np.sort(d["sa_target_g"].unique())


def stripes(sub, thr):
    """Exceedance count and sample size at each intensity stripe."""
    g = sub.assign(e=sub["midr"] >= thr).groupby("sa_target_g")["e"]
    a = g.agg(["sum", "count"]).reindex(LEVELS)
    return a["sum"].to_numpy(float), a["count"].to_numpy(float)


def nll(p, x, z, n):
    """Negative log-likelihood of the lognormal fragility over the stripes."""
    lnth, lnbe = p
    q = np.clip(ndtr((np.log(x) - lnth) / np.exp(lnbe)), EPS, 1 - EPS)
    return -np.sum(z * np.log(q) + (n - z) * np.log(1 - q))


def fit_mle(x, z, n):
    best, bestv = None, np.inf
    for lnth0 in np.log([0.3, 0.7, 1.2, 2.0, 3.0]):
        r = minimize(nll, [lnth0, np.log(0.4)], args=(x, z, n), method="Nelder-Mead",
                     options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
        if r.fun < bestv:
            best, bestv = r.x, r.fun
    return float(np.exp(best[0])), float(np.exp(best[1]))


def posterior(x, z, n, th_hat, be_hat):
    """Flat-prior posterior on a grid; returns 95% credible intervals."""
    th = np.exp(np.linspace(np.log(th_hat) - 1.2, np.log(th_hat) + 1.2, 401))
    be = np.linspace(max(0.02, be_hat - 0.45), be_hat + 0.45, 401)
    L = np.empty((th.size, be.size))
    for i, t in enumerate(th):
        for j, b in enumerate(be):
            L[i, j] = -nll([np.log(t), np.log(b)], x, z, n)
    P = np.exp(L - L.max())
    P /= P.sum()

    def ci(vals, marg):
        c = np.cumsum(marg)
        return (float(np.interp(0.025, c, vals)), float(np.interp(0.975, c, vals)))

    return ci(th, P.sum(axis=1)), ci(be, P.sum(axis=0))


fits, obs, rows = {}, {}, []
for cname, flag in CLASSES:
    sub = d[d["soft"] == flag]
    fits[cname], obs[cname] = [], []
    for dname, thr in DS:
        z, n = stripes(sub, thr)
        th, be = fit_mle(LEVELS, z, n)
        (th_lo, th_hi), (be_lo, be_hi) = posterior(LEVELS, z, n, th, be)
        fits[cname].append((th, be))
        obs[cname].append(z / n)
        rows.append(dict(cls=cname, ds=dname, theta=th, beta=be,
                         th_lo=th_lo, th_hi=th_hi, be_lo=be_lo, be_hi=be_hi,
                         n=int(n.sum())))
tab = pd.DataFrame(rows)

# ----------------------------------------------------------------- style ----
set_rc()
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8, "axes.labelpad": 2, "axes.titlepad": 4,
    "savefig.dpi": 600,
})

fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.55), sharey=True,
                         constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
axes[0].set_anchor("E"); axes[1].set_anchor("W")
Sa = np.linspace(0.02, 2.5, 500)

for ax, (cname, _), tag in zip(axes, CLASSES, "ab"):
    for k, (dname, _) in enumerate(DS):
        th, be = fits[cname][k]
        ax.plot(Sa, ndtr(np.log(Sa / th) / be), "-", color=DS_COL[k], lw=1.1,
                zorder=4,
                label=f"{dname} ($\\theta$={th:.2f}g, $\\beta$={be:.2f})")
        ax.plot(LEVELS, obs[cname][k], "o", ms=4, mfc="white", mec=DS_COL[k],
                mew=1.1, zorder=5)
    ax.axvline(LEVELS[-1], color="0.75", lw=0.5, zorder=1)   # sampled range
    ax.set_xlim(0, 2.5); ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"Spectral acceleration $S_a(T_1)$ (g)")
    ax.set_title(f"({tag}) {cname} frames")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.85")
    ax.grid(which="minor", lw=0.3, color="0.93")
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    if tag == "a":                      # the marker key rides with panel (a)
        handles.append(Line2D([0], [0], marker="o", mfc="white", mec="0.3",
                              ls="none", ms=5))
        labels.append("simulation (six stripes)")
    # The four sigmoids sweep the whole panel, so no in-axes corner is free of
    # data: the key sits immediately below its own panel instead.
    leg = ax.legend(handles, labels, loc="upper left",
                    bbox_to_anchor=(-0.02, -0.17), ncol=2, fontsize=7.4,
                    framealpha=1.0, edgecolor="0.7", fancybox=False,
                    borderpad=0.40, labelspacing=0.30, handlelength=1.5,
                    handletextpad=0.5, columnspacing=1.0)
    leg.get_frame().set_linewidth(0.5)

axes[0].set_ylabel(r"Probability of exceedance $P(DS\!\geq\!ds\,|\,S_a)$")

# ------------------------------------------------------ checks + export -----
fig.canvas.draw()
check_figure(fig)
assert_not_dejavu(fig)
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"{NAME}.{ext}", bbox_inches="tight", pad_inches=0.02)

pd.set_option("display.width", 200)
print(tab.round(3).to_string(index=False))
DERIVED.mkdir(parents=True, exist_ok=True)
tab.to_csv(DERIVED / "fragility_fits.csv", index=False)
piv = tab.pivot(index="ds", columns="cls", values="theta").reindex(
    [n for n, _ in DS])
print("\nsoft/regular median ratio:\n",
      (piv["Soft-storey"] / piv["Regular"]).round(3).to_string())
