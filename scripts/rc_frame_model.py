#!/usr/bin/env python3
"""
rc_frame_model.py - parametric 2-D RC frame builder and analysis engine.

This is the real simulator: it builds a fibre-section moment frame in
OpenSeesPy, recovers its fundamental period from an eigenvalue analysis, and
runs nonlinear time-history analysis under a recorded accelerogram, following
Section 4.1-4.2 of the manuscript:

  * force-based forceBeamColumn elements, 5 Gauss-Lobatto integration points
  * Concrete04 (Mander) with separate confined and unconfined envelopes
  * Steel02 (Giuffre-Menegotto-Pinto)
  * P-Delta geometric transformation
  * Rayleigh damping calibrated at modes 1 and 3
  * Newmark average acceleration (gamma = 1/2, beta = 1/4)
  * collapse at 10 % inter-storey drift; on non-convergence the step is
    subdivided down to dt/64 before the run is abandoned

Units: N, m, s, kg.
"""
import numpy as np
import openseespy.opensees as ops

G = 9.80665


# ---------------------------------------------------------------------------
# frame definition
# ---------------------------------------------------------------------------
class Frame:
    """Parameters of one archetype realisation."""

    def __init__(self, n_storey=4, n_bay=3, bay=5.0, h_typ=3.0,
                 soft=False, r_k=1.0, deficient=False,
                 fc=30e6, fy=420e6, col_b=0.40, col_h=0.40,
                 beam_b=0.30, beam_h=0.50, rho_long=0.015,
                 conf_ratio=1.30, xi=0.05, load_kpa=8.0, fid=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        # soft storey: taller ground storey, so its stiffness ratio drops
        self.h_ground = h_typ / (r_k ** (1.0 / 3.0)) if soft else h_typ
        if soft:
            self.h_ground = h_typ * (1.0 / r_k) ** (1.0 / 3.0)
        self.heights = [self.h_ground] + [h_typ] * (n_storey - 1)
        self.levels = np.concatenate([[0.0], np.cumsum(self.heights)])

    def __repr__(self):
        return (f"Frame(id={self.fid}, {self.n_storey}st, soft={self.soft}, "
                f"r_k={self.r_k:.2f}, deficient={self.deficient}, "
                f"fc={self.fc/1e6:.0f}MPa)")


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def _fibre_section(tag, f, b, h, is_column):
    """Fibre section: unconfined cover + confined core + longitudinal bars."""
    cover = 0.04
    fc = f.fc
    fcc = f.conf_ratio * fc
    Ec = 5000.0 * np.sqrt(fc / 1e6) * 1e6
    ft = 0.33 * np.sqrt(fc / 1e6) * 1e6

    # deficient detailing: sparse hoops -> little confinement, low ductility
    ecu_c = 0.020 if not f.deficient else 0.008
    fcc_ = fcc if not f.deficient else 1.10 * fc

    mc, mu, ms = tag * 10 + 1, tag * 10 + 2, tag * 10 + 3
    ops.uniaxialMaterial("Concrete04", mc, -fcc_, -0.005, -ecu_c, Ec, ft, 0.001)
    ops.uniaxialMaterial("Concrete04", mu, -fc, -0.002, -0.006, Ec, ft, 0.001)
    ops.uniaxialMaterial("Steel02", ms, f.fy, 200e9, 0.01, 18.0, 0.925, 0.15)

    y1, z1 = h / 2.0, b / 2.0
    yc, zc = y1 - cover, z1 - cover
    ops.section("Fiber", tag)
    ops.patch("rect", mc, 8, 4, -yc, -zc, yc, zc)                 # core
    ops.patch("rect", mu, 8, 1, -y1, zc, y1, z1)                  # cover +z
    ops.patch("rect", mu, 8, 1, -y1, -z1, y1, -zc)                # cover -z
    ops.patch("rect", mu, 1, 4, yc, -zc, y1, zc)                   # cover +y
    ops.patch("rect", mu, 1, 4, -y1, -zc, -yc, zc)                 # cover -y

    Ag = b * h
    As_tot = f.rho_long * Ag
    n_side = 3
    A_bar = As_tot / (2 * n_side + (2 if is_column else 0))
    ops.layer("straight", ms, n_side, A_bar, yc, -zc, yc, zc)
    ops.layer("straight", ms, n_side, A_bar, -yc, -zc, -yc, zc)
    if is_column:
        ops.layer("straight", ms, 2, A_bar, 0.0, -zc, 0.0, zc)


def build(f, verbose=False):
    """Build the frame; return (node grid, floor masses)."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    nx, nz = f.n_bay + 1, f.n_storey + 1
    nid = lambda i, j: 1000 + 100 * j + i          # noqa: E731
    for j in range(nz):
        for i in range(nx):
            ops.node(nid(i, j), i * f.bay, f.levels[j])
    for i in range(nx):
        ops.fix(nid(i, 0), 1, 1, 1)

    # storey mass from a uniform floor load over the tributary width
    trib = f.n_bay * f.bay
    m_floor = f.load_kpa * 1e3 * trib * f.bay / G          # kg per floor
    for j in range(1, nz):
        for i in range(nx):
            ops.mass(nid(i, j), m_floor / nx, 1e-9, 1e-9)

    ops.geomTransf("PDelta", 1)
    _fibre_section(1, f, f.col_b, f.col_h, True)
    _fibre_section(2, f, f.beam_b, f.beam_h, False)
    ops.beamIntegration("Lobatto", 1, 1, 4)
    ops.beamIntegration("Lobatto", 2, 2, 4)

    e = 1
    for j in range(f.n_storey):                     # columns
        for i in range(nx):
            ops.element("forceBeamColumn", e, nid(i, j), nid(i, j + 1), 1, 1)
            e += 1
    for j in range(1, nz):                          # beams
        for i in range(f.n_bay):
            ops.element("forceBeamColumn", e, nid(i, j), nid(i + 1, j), 1, 2)
            e += 1

    if verbose:
        print(f"    built: {nx*nz} nodes, {e-1} elements, "
              f"m_floor = {m_floor/1e3:.1f} t")
    return nid, m_floor, nx, nz


def eigen_T(n=3):
    """Return the first n periods (s)."""
    lam = ops.eigen(n)
    lam = np.array(lam, float)
    lam[lam <= 0] = np.nan
    return 2.0 * np.pi / np.sqrt(lam)


def gravity(f, nid, m_floor, nx, nz):
    """Apply and hold gravity in a load-controlled static analysis."""
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for j in range(1, nz):
        for i in range(nx):
            ops.load(nid(i, j), 0.0, -m_floor / nx * G, 0.0)
    ops.system("UmfPack"); ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.test("EnergyIncr", 1e-8, 100)
    ops.algorithm("Newton"); ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    ok = ops.analyze(10)
    ops.loadConst("-time", 0.0)
    return ok == 0


# ---------------------------------------------------------------------------
# nonlinear time-history analysis
# ---------------------------------------------------------------------------
DRIFT_COLLAPSE = 0.10
DRIFT_HEAVY = 0.06


def run_nltha(f, accel, dt, scale=1.0, verbose=False):
    """Run one record on one frame.

    Returns dict with peak MIDR, peak roof drift, status and the period used.
    status: 'ok' | 'collapse' | 'nonconverged'
    """
    nid, m_floor, nx, nz = build(f)
    T = eigen_T(3)
    if not np.isfinite(T[0]):
        return dict(status="eigfail", midr=np.nan, T1=np.nan)
    if not gravity(f, nid, m_floor, nx, nz):
        return dict(status="gravfail", midr=np.nan, T1=float(T[0]))

    # Rayleigh damping at modes 1 and 3
    w1, w3 = 2 * np.pi / T[0], 2 * np.pi / T[min(2, len(T) - 1)]
    a0 = 2 * f.xi * w1 * w3 / (w1 + w3)
    a1 = 2 * f.xi / (w1 + w3)
    ops.rayleigh(a0, 0.0, 0.0, a1)

    ops.timeSeries("Path", 2, "-dt", dt, "-values", *(accel * scale * G).tolist())
    ops.pattern("UniformExcitation", 2, 1, "-accel", 2)

    ops.wipeAnalysis()
    ops.system("UmfPack"); ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-7, 25)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    ctrl = [nid(0, j) for j in range(nz)]
    peak = 0.0
    nsteps = len(accel)
    t_end = nsteps * dt
    status = "ok"
    step_dt = dt
    t = 0.0
    while t < t_end:
        ok = ops.analyze(1, step_dt)
        if ok != 0:                                   # relax, then subdivide
            for alg in ("ModifiedNewton", "NewtonLineSearch"):
                ops.algorithm(alg)
                ok = ops.analyze(1, step_dt)
                if ok == 0:
                    break
            ops.algorithm("Newton")
        if ok != 0:                                   # subdivide down to dt/64
            sub = step_dt
            for _ in range(6):
                sub /= 2.0
                ops.test("NormDispIncr", 1e-6, 50)
                ok = ops.analyze(1, sub)
                ops.test("NormDispIncr", 1e-7, 25)
                if ok == 0:
                    break
            if ok != 0:
                status = "collapse" if peak >= DRIFT_HEAVY else "nonconverged"
                break
            t += sub
        else:
            t += step_dt
        u = [ops.nodeDisp(n, 1) for n in ctrl]
        d = max(abs(u[j + 1] - u[j]) / f.heights[j] for j in range(f.n_storey))
        peak = max(peak, d)
        if peak >= DRIFT_COLLAPSE:
            status = "collapse"
            break
    if verbose:
        print(f"    T1={T[0]:.3f}s  scale={scale:.3f}  MIDR={peak:.4f}  {status}")
    return dict(status=status, midr=float(min(peak, DRIFT_COLLAPSE)),
                T1=float(T[0]), T3=float(T[min(2, len(T)-1)]))


def spectral_acceleration(accel_g, dt, T, xi=0.05):
    """5 %-damped pseudo-spectral acceleration (g) by the Nigam-Jennings
    exact solution for piecewise-linear excitation. Vectorised over samples."""
    if T <= 0:
        return float(np.max(np.abs(accel_g)))
    w = 2.0 * np.pi / T
    wd = w * np.sqrt(1.0 - xi ** 2)
    e = np.exp(-xi * w * dt)
    s_, c_ = np.sin(wd * dt), np.cos(wd * dt)
    a11 = e * (xi / np.sqrt(1 - xi ** 2) * s_ + c_)
    a12 = e / wd * s_
    a21 = -e * w / np.sqrt(1 - xi ** 2) * s_
    a22 = e * (c_ - xi / np.sqrt(1 - xi ** 2) * s_)
    xd = xi / (w * dt)
    b11 = e * ((2 * xi ** 2 - 1) / (w ** 2 * dt) + xi / w) * s_ / wd \
        + e * (2 * xi / (w ** 3 * dt) + 1 / w ** 2) * c_ - 2 * xi / (w ** 3 * dt)
    b12 = -e * ((2 * xi ** 2 - 1) / (w ** 2 * dt)) * s_ / wd \
        - e * (2 * xi / (w ** 3 * dt)) * c_ - 1 / w ** 2 + 2 * xi / (w ** 3 * dt)
    b21 = e * ((2 * xi ** 2 - 1) / (w ** 2 * dt) + xi / w) * (c_ - xi * w / wd * s_) \
        - e * (2 * xi / (w ** 3 * dt) + 1 / w ** 2) * (wd * s_ + xi * w * c_) \
        + 1 / (w ** 2 * dt)
    b22 = -e * ((2 * xi ** 2 - 1) / (w ** 2 * dt)) * (c_ - xi * w / wd * s_) \
        + e * (2 * xi / (w ** 3 * dt)) * (wd * s_ + xi * w * c_) - 1 / (w ** 2 * dt)
    p = -np.asarray(accel_g, float) * G
    u = v = 0.0
    umax = 0.0
    for i in range(len(p) - 1):
        u, v = (a11 * u + a12 * v + b11 * p[i] + b12 * p[i + 1],
                a21 * u + a22 * v + b21 * p[i] + b22 * p[i + 1])
        if abs(u) > umax:
            umax = abs(u)
    return float(umax * w ** 2 / G)
