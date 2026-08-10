#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pushover.py -- monotonic nonlinear static (pushover) capacity analysis.

Implements Section 4.6 of the manuscript: for every archetype of the frame
population, a displacement-controlled pushover under an inverted-triangular
lateral load pattern traces the base-shear--roof-displacement response up to
failure. The curve is reduced to an equivalent single-degree-of-freedom (SDOF)
system through the first-mode participation factor, idealised as
elastic--perfectly-plastic, and expressed in the acceleration--displacement
response-spectrum (ADRS) plane, giving the behaviour (force-reduction) factor
and the displacement ductility of Equation (4):

    q = F_E / F_Y ,      mu = U_M / U_Y

The capacity ordering this produces is the mechanics-based reference against
which the SHAP feature-importance ranking of Section 5.3 is interpreted. The
two are kept methodologically distinct: nothing here is fed to the surrogate.

SDOF reduction (Fajfar 2000)
----------------------------
With phi the first-mode shape normalised to unity at the roof and m_j the
floor masses,

    Gamma = sum(m_j phi_j) / sum(m_j phi_j^2)       participation factor
    m*    = sum(m_j phi_j)                          equivalent SDOF mass
    D*    = D_roof / Gamma ,   F* = V_base / Gamma  SDOF capacity curve
    S_a   = F* / m*                                 ADRS ordinate

Bilinear idealisation
---------------------
Equal-energy elastic--perfectly-plastic fit up to the ultimate displacement
U_M, taken at the first of: 20 % strength degradation from the peak, or the
drift cap. F_Y is the plateau force; the initial stiffness is secant to
0.6 F_Y (Eurocode 8 Part 1, Annex B convention). F_E = m* S_ae(T*) is the force
the frame would attain if it remained elastic at its own equivalent period.

Usage
-----
    python3 pushover.py                    # all 60 archetypes
    python3 pushover.py --fid 0 5 30 45    # selected frames
    python3 pushover.py --plot             # also write a capacity-curve figure

Writes results/derived/pushover_capacity.csv with one row per frame:
    fid, soft, deficient, n_storey, fc_MPa, rho_long, r_k, T1_s, T_star_s,
    Gamma, m_star_kg, V_max_N, D_max_m, U_Y_m, F_Y_N, U_M_m, F_E_N, q, mu,
    Sa_yield_g, Sd_yield_m, status
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import openseespy.opensees as ops

# NumPy 2 renamed trapz -> trapezoid; support both so the script runs under
# either major version.
_trapz = getattr(np, "trapezoid", None) or np.trapz

import rc_frame_model as M
import frame_population as FP

# ---------------------------------------------------------------- paths ----
# This file lives in <repo>/scripts/. Every path below is resolved from
# __file__, so the script runs from any working directory and on any machine.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DERIVED = RESULTS / "derived"
FIGDIR = ROOT / "figures"

G = M.G

# Push to 10 % roof drift, matching the collapse drift cap of the dynamic
# analysis (rc_frame_model.DRIFT_COLLAPSE), so that U_M is governed by strength
# degradation of the frame rather than by the end of the analysis. A shorter
# push truncates the ductile archetypes before they degrade and inflates mu.
ROOF_DRIFT_TARGET = 0.10
N_STEPS = 600
DEGRADE_FRAC = 0.80          # U_M at 20 % loss from peak base shear
SECANT_FRAC = 0.60           # initial stiffness secant to 0.6 F_Y (EC8-1 Annex B)

# Elastic demand for the force-reduction factor. F_E is the force the frame
# would attain if it remained elastic (Vidic et al. 1994), evaluated from the
# EC8 Type 1 elastic spectrum at the equivalent SDOF period T*:
#     F_E = m* S_ae(T*)
# so that q = F_E / F_Y measures strength reduction against a fixed seismic
# demand, independently of the displacement ductility mu. Defaults: reference
# peak ground acceleration on type A ground and EC8 ground type C, the median
# site class of the instrumented stations of Figure 2.
AG_DEFAULT = 0.40            # a_g / g, reference PGA on rock
SOIL_DEFAULT = "C"
# EC8-1 Table 3.2, Type 1 spectrum: S, T_B, T_C, T_D
EC8_TYPE1 = {"A": (1.00, 0.15, 0.40, 2.0), "B": (1.20, 0.15, 0.50, 2.0),
             "C": (1.15, 0.20, 0.60, 2.0), "D": (1.35, 0.20, 0.80, 2.0),
             "E": (1.40, 0.15, 0.50, 2.0)}


def ec8_sae(T, ag=AG_DEFAULT, soil=SOIL_DEFAULT, eta=1.0):
    """EC8-1 Type 1 elastic response spectrum S_ae(T), in g."""
    S, TB, TC, TD = EC8_TYPE1[soil]
    if T < 0:
        raise ValueError("period must be non-negative")
    if T < TB:
        return ag * S * (1.0 + T / TB * (eta * 2.5 - 1.0))
    if T < TC:
        return ag * S * eta * 2.5
    if T < TD:
        return ag * S * eta * 2.5 * TC / T
    return ag * S * eta * 2.5 * TC * TD / T ** 2


# ---------------------------------------------------------------------------
def first_mode_shape(nid, nx, nz):
    """First-mode horizontal shape at floor level, normalised to unity at roof.

    Read from the eigenvector at the leftmost column line; the frame is
    regular in plan, so any column line gives the same normalised shape.
    """
    phi = np.array([ops.nodeEigenvector(nid(0, j), 1, 1) for j in range(nz)],
                   dtype=float)
    if not np.isfinite(phi).all() or abs(phi[-1]) < 1e-12:
        return None
    phi = phi / phi[-1]
    phi[0] = 0.0                                   # base is fixed
    return phi


def push(f, verbose=False):
    """Displacement-controlled pushover of one frame.

    Returns dict with the raw capacity curve and the SDOF reduction, or a
    status flag if the model could not be built or the push did not converge
    before yielding.
    """
    nid, m_floor, nx, nz = M.build(f)
    T = M.eigen_T(3)
    if not np.isfinite(T[0]):
        return dict(status="eigfail")

    phi = first_mode_shape(nid, nx, nz)
    if phi is None:
        return dict(status="modefail")

    if not M.gravity(f, nid, m_floor, nx, nz):
        return dict(status="gravfail")

    # --- SDOF reduction factors (masses are lumped equally across nx nodes) --
    m_j = np.full(nz, m_floor)
    m_j[0] = 0.0                                   # no mass at the fixed base
    Gamma = float(np.sum(m_j * phi) / np.sum(m_j * phi ** 2))
    m_star = float(np.sum(m_j * phi))

    # --- inverted-triangular lateral pattern, scaled by the mode shape ------
    # F_j proportional to m_j phi_j, the standard first-mode-proportional
    # pattern; distributed equally across the nx nodes of each floor.
    ops.timeSeries("Linear", 3)
    ops.pattern("Plain", 3, 3)
    for j in range(1, nz):
        Fj = m_j[j] * phi[j]
        for i in range(nx):
            ops.load(nid(i, j), Fj / nx, 0.0, 0.0)

    roof = nid(0, nz - 1)
    H = float(f.levels[-1])
    d_target = ROOF_DRIFT_TARGET * H
    du = d_target / N_STEPS

    ops.wipeAnalysis()
    ops.system("UmfPack"); ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-7, 25)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", roof, 1, du)
    ops.analysis("Static")

    D, V = [0.0], [0.0]
    status = "ok"
    for _ in range(N_STEPS):
        ok = ops.analyze(1)
        if ok != 0:
            for alg in ("ModifiedNewton", "NewtonLineSearch"):
                ops.algorithm(alg)
                ok = ops.analyze(1)
                if ok == 0:
                    break
            ops.algorithm("Newton")
        if ok != 0:                                # subdivide, then give up
            sub = du
            for _ in range(5):
                sub /= 2.0
                ops.integrator("DisplacementControl", roof, 1, sub)
                ops.test("NormDispIncr", 1e-6, 50)
                ok = ops.analyze(1)
                ops.test("NormDispIncr", 1e-7, 25)
                if ok == 0:
                    break
            ops.integrator("DisplacementControl", roof, 1, du)
            if ok != 0:
                status = "terminated"
                break

        ops.reactions()
        base = -sum(ops.nodeReaction(nid(i, 0), 1) for i in range(nx))
        D.append(float(ops.nodeDisp(roof, 1)))
        V.append(float(base))

        # stop once the frame has shed 20 % of its peak strength
        if len(V) > 10 and V[-1] < DEGRADE_FRAC * max(V):
            break

    D, V = np.asarray(D), np.asarray(V)
    if len(D) < 10 or V.max() <= 0:
        return dict(status="nocapacity" if status == "ok" else status)

    if verbose:
        print(f"    fid={f.fid:2d}  T1={T[0]:.3f}s  Gamma={Gamma:.3f}  "
              f"V_max={V.max()/1e3:.0f} kN  D_max={D.max():.3f} m  [{status}]")

    return dict(status=status, D=D, V=V, T1=float(T[0]),
                Gamma=Gamma, m_star=m_star, phi=phi)


# ---------------------------------------------------------------------------
def bilinear(Dst, Fst):
    """Equal-energy elastic--perfectly-plastic idealisation of an SDOF curve.

    Returns (U_Y, F_Y, U_M, K_e). U_M is taken at 20 % strength degradation
    from the peak, or at the end of the curve if it never degrades that far.
    F_Y follows from equating the area under the real and idealised curves up
    to U_M; the elastic branch is the secant to 0.6 F_Y, iterated to
    convergence because that secant itself depends on F_Y.
    """
    i_peak = int(np.argmax(Fst))
    F_peak = float(Fst[i_peak])

    # ultimate displacement: first post-peak point below 80 % of the peak
    post = np.where(Fst[i_peak:] < DEGRADE_FRAC * F_peak)[0]
    i_u = i_peak + int(post[0]) if len(post) else len(Fst) - 1
    U_M = float(Dst[i_u])
    if U_M <= 0:
        return None

    E = float(_trapz(Fst[:i_u + 1], Dst[:i_u + 1]))   # area under curve

    F_Y = F_peak                                   # first guess
    for _ in range(50):
        # secant stiffness at 0.6 F_Y
        idx = np.argmax(Fst >= SECANT_FRAC * F_Y)
        if idx == 0 or Dst[idx] <= 0:
            return None
        K_e = float(Fst[idx] / Dst[idx])
        # equal-energy: E = F_Y (U_M - F_Y / (2 K_e))
        disc = U_M ** 2 - 2.0 * E / K_e
        if disc < 0:
            F_Y_new = K_e * U_M                    # degenerate: stays elastic
        else:
            F_Y_new = K_e * (U_M - np.sqrt(disc))
        if abs(F_Y_new - F_Y) < 1e-6 * max(F_Y, 1.0):
            F_Y = float(F_Y_new)
            break
        F_Y = float(F_Y_new)

    idx = np.argmax(Fst >= SECANT_FRAC * F_Y)
    K_e = float(Fst[idx] / Dst[idx]) if idx > 0 and Dst[idx] > 0 else np.nan
    U_Y = F_Y / K_e if np.isfinite(K_e) and K_e > 0 else np.nan
    return U_Y, F_Y, U_M, K_e


def capacity(f, res, ag=AG_DEFAULT, soil=SOIL_DEFAULT):
    """Reduce a pushover result to the ADRS quantities of Equation (4)."""
    D, V = res["D"], res["V"]
    Gamma, m_star = res["Gamma"], res["m_star"]

    Dst = D / Gamma                                # SDOF displacement
    Fst = V / Gamma                                # SDOF force

    bl = bilinear(Dst, Fst)
    if bl is None:
        return None
    U_Y, F_Y, U_M, K_e = bl
    if not (np.isfinite(U_Y) and U_Y > 0 and F_Y > 0):
        return None

    # equivalent SDOF period from the idealised elastic branch
    T_star = 2.0 * np.pi * np.sqrt(m_star / K_e)

    # F_E: the force the frame would attain if it remained elastic under the
    # code seismic demand at its own equivalent period (Vidic et al. 1994).
    # Taking F_E from the elastic spectrum rather than from K_e U_M keeps q
    # independent of mu; the latter definition makes the two identical by
    # construction and so carries no additional information.
    Sae = ec8_sae(T_star, ag=ag, soil=soil)
    F_E = m_star * Sae * G

    return dict(
        T_star_s=float(T_star), Gamma=float(Gamma), m_star_kg=float(m_star),
        V_max_N=float(V.max()), D_max_m=float(D.max()),
        U_Y_m=float(U_Y), F_Y_N=float(F_Y), U_M_m=float(U_M),
        F_E_N=float(F_E),
        q=float(F_E / F_Y), mu=float(U_M / U_Y),
        Sa_yield_g=float(F_Y / m_star / G), Sd_yield_m=float(U_Y),
        Sae_star_g=float(Sae), ag_g=float(ag), soil=soil,
    )


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--frames", default=str(DATA / "frame_population.csv"))
    ap.add_argument("--out", default=str(DERIVED / "pushover_capacity.csv"))
    ap.add_argument("--fid", type=int, nargs="*", default=None,
                    help="analyse only these frame ids (default: all)")
    ap.add_argument("--plot", action="store_true",
                    help="also write figures/pushover_capacity.pdf/.png")
    ap.add_argument("--ag", type=float, default=AG_DEFAULT,
                    help=f"reference PGA on rock, in g (default {AG_DEFAULT})")
    ap.add_argument("--soil", default=SOIL_DEFAULT, choices=list(EC8_TYPE1),
                    help=f"EC8 ground type (default {SOIL_DEFAULT})")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if not Path(a.frames).exists():
        sys.exit(f"frame population not found: {a.frames}\n"
                 f"Run:  python3 frame_population.py")
    frames = pd.read_csv(a.frames)
    if a.fid:
        frames = frames[frames.fid.isin(a.fid)]

    DERIVED.mkdir(parents=True, exist_ok=True)
    print(f"pushover: {len(frames)} frames, target roof drift "
          f"{ROOF_DRIFT_TARGET:.1%}, {N_STEPS} steps")
    print(f"elastic demand for q: EC8 Type 1, a_g = {a.ag:g} g, "
          f"ground type {a.soil}\n")

    rows, curves = [], {}
    for _, row in frames.iterrows():
        f = FP.to_frame(row)
        res = push(f, verbose=not a.quiet)
        base = dict(fid=int(row.fid), soft=bool(row.soft),
                    deficient=bool(row.deficient), n_storey=int(row.n_storey),
                    fc_MPa=round(row.fc / 1e6, 2), rho_long=float(row.rho_long),
                    r_k=float(row.r_k), T1_s=float(row.T1_s))
        if res["status"] not in ("ok", "terminated") or "D" not in res:
            rows.append({**base, "status": res["status"]})
            continue
        cap = capacity(f, res, ag=a.ag, soil=a.soil)
        if cap is None:
            rows.append({**base, "status": "idealisation_failed"})
            continue
        rows.append({**base, **cap, "status": res["status"]})
        curves[int(row.fid)] = (res["D"], res["V"], bool(row.soft),
                                bool(row.deficient))

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")

    ok = out[out.q.notna()] if "q" in out else out.iloc[:0]
    if len(ok):
        print(f"\n{len(ok)}/{len(out)} frames idealised successfully")
        print("\n  class                       n      q (range)        "
              "mu (range)")
        for (s, d), g in ok.groupby(["soft", "deficient"]):
            lab = f"{'soft-storey' if s else 'regular':11s}/"\
                  f"{'deficient' if d else 'code-conf':10s}"
            print(f"  {lab}  {len(g):3d}   "
                  f"{g.q.min():4.2f}-{g.q.max():4.2f}     "
                  f"{g.mu.min():4.2f}-{g.mu.max():4.2f}")
        print("\n  Compare with Section 4.6: code-conforming regular frames "
              "q~4.0-5.0, mu~4.5-6.0;")
        print("  deficient soft-storey frames q~1.5-2.3, mu~2.0-3.0.")

    if a.plot and curves:
        plot_curves(curves)


def plot_curves(curves):
    """Capacity curves grouped by archetype class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_style import set_rc, apply_style, assert_not_dejavu

    set_rc()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.8), constrained_layout=True)

    # Okabe-Ito: blue = regular/code-conforming, vermillion = soft-storey
    for fid, (D, V, soft, deficient) in sorted(curves.items()):
        c = "#D55E00" if soft else "#0072B2"
        ls = "--" if deficient else "-"
        ax.plot(D, V / 1e3, color=c, ls=ls, lw=0.8, alpha=0.55)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="#0072B2", ls="-", lw=1.2, label="regular, code-conforming"),
        Line2D([], [], color="#0072B2", ls="--", lw=1.2, label="regular, deficient"),
        Line2D([], [], color="#D55E00", ls="-", lw=1.2, label="soft-storey, code-conforming"),
        Line2D([], [], color="#D55E00", ls="--", lw=1.2, label="soft-storey, deficient"),
    ], loc="lower right", framealpha=0.9, edgecolor="0.6", borderpad=0.4)

    apply_style(ax, xlabel="Roof displacement $D$ (m)",
                ylabel="Base shear $V$ (kN)", grid="both")
    assert_not_dejavu(fig)
    for ext in ("pdf", "png"):
        kw = dict(bbox_inches="tight", pad_inches=0.02)
        if ext == "png":
            kw["dpi"] = 600
        fig.savefig(FIGDIR / f"pushover_capacity.{ext}", **kw)
    print(f"wrote {FIGDIR / 'pushover_capacity.pdf'} and .png")


if __name__ == "__main__":
    main()
