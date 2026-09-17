# SPDX-License-Identifier: LGPL-3.0-or-later
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("FULL_BERREMAN_DIR", "."))

"""Gates G17: POLARIZANCE v2 — Brown differential Mueller decomposition
(Phase 11, plan §11; decision: BOTH routes, asymmetric — §11.2 DECISION block).

Validation ladder (fixed implementation order):
  G17b  Route A vs live BerreMueller (external oracle);
  G17a  Route B expm path vs the Jones group path on the SAME eigen data
        (mueller_from_diff vs mueller_from_jones(bulk jones)) — machine
        precision;
  G17a2 Route B vs the FULL SOLVER, per-axis Airy interface correction
        (G16a2-validated _airy_t form; diagonal layer) — machine precision;
  G17c  reductions: isotropic -> exp(-abs·L)·I; pure retarder orthogonal with
        v1 polarizance/diattenuation == 0; pure dichroic closed form;
        diff_matrix placement == live transcription;
  G17d  twisted staircase: Trotter self-convergence (12/24/48 monotone) +
        solver-difference stabilization (fixed outer-interface factor,
        O(1/n) trend — documented, not a machine-precision gate);
  G17e  brown_params small-L Taylor structure (SI S10–S11).

Literature: Brown 1999 (DOI 10.1117/12.366361); Salij–Goldsmith–Tempelaar
arXiv:2208.14461v2 SI §S2–S3 (extracted locally this session); Gil &
Ossikovski. Conventions: per-unit-length differentials; length enters only in
the exponentials; live's max-of-two-orientations absorbance hack NOT
inherited (both orientation sums compared independently)."""
import matplotlib.cm as _cm
if not hasattr(_cm, "get_cmap"):
    _cm.get_cmap = lambda name=None: __import__("matplotlib").colormaps[name]
import numpy as np
import pytest

from navette import berreman as bl

try:
    from berremueller import dielectric_tensor as bdt
    from berremueller import mueller as bmu

    HAVE_BM = True
except ImportError:
    HAVE_BM = False
print("live BerreMueller available:", HAVE_BM)

C0 = 299792458e9  # nm/s

WL = np.array([500.0, 600.0, 700.0, 850.0])


def _diag_eps(nx, ny, nz):
    e = np.zeros((3, 3), dtype=complex)
    e[0, 0], e[1, 1], e[2, 2] = nx * nx, ny * ny, nz * nz
    return e


EPS_DIA = _diag_eps(1.5 + 0j, 1.5 + 0.1j, 1.7 + 0j)   # birefringent + lossy
EPS_UNI = _diag_eps(1.5 + 0j, 1.5 + 0j, 1.7 + 0j)     # uniaxial, lossless
ZEROS = np.zeros((3, 3), dtype=complex)
EYE = np.eye(3, dtype=complex)


def _worst(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = np.abs(a - b)
    s = max(1.0, float(np.abs(b).max()))
    return float((d / s).max())


# ── G17b: Route A vs live (external oracle) ────────────────────────────────

@pytest.mark.skipif(not HAVE_BM, reason="live BerreMueller not importable")
def test_g17b_route_a_vs_live():
    """ld/ldp/lb/lbp vs live linear_optics_from_dielectric_tensor, shared
    fixtures (same spectrum values, length_over_c = 1)."""
    eps = EPS_DIA
    spectrum = 2 * np.pi * C0 / WL
    eps_w = np.broadcast_to(eps[..., np.newaxis], (3, 3, WL.size))  # live (3,3,n_wl)
    ours = bl.linear_optics(np.broadcast_to(eps, (WL.size, 3, 3)), wavelengths_nm=WL)
    lo = bdt.linear_optics_from_dielectric_tensor(eps_w, spectrum, length_over_c=1.0)
    assert _worst(ours["ld"], lo.ld) < 1e-12
    assert _worst(ours["ldp"], lo.ldp) < 1e-12
    assert _worst(ours["lb"], lo.lb) < 1e-12
    assert _worst(ours["lbp"], lo.lbp) < 1e-12
    # absorbance: both orientation sums, recomputed with live's own functions
    # (linear_optics_from_dielectric_tensor applies max() and hides them);
    # live's tensor convention is (3,3,n_wl) — wavelength LAST
    n_t = bdt.get_refractive_index_tensor(eps_w)
    rot = bdt.get_xy_rotation_matrix(3, -np.pi / 4)
    eps_p = np.einsum("ij,jkl->ikl", rot,
                      np.einsum("ijl,jk->ikl", eps_w, rot.T))
    n_p = bdt.get_refractive_index_tensor(eps_p)
    abs1 = (n_p[1, 1, :].imag + n_p[0, 0, :].imag) * spectrum
    abs2 = (n_t[1, 1, :].imag + n_t[0, 0, :].imag) * spectrum
    assert _worst(ours["absorbance_1"], abs1) < 1e-12
    assert _worst(ours["absorbance_2"], abs2) < 1e-12
    assert _worst(np.maximum(abs1, abs2), lo.absorbance) < 1e-15


@pytest.mark.skipif(not HAVE_BM, reason="live BerreMueller not importable")
def test_g17b_brown_params_vs_live():
    """a0..a3 vs live brown_params on shared (r_p, i_p, n_p) fixtures."""
    r_p = np.array([0.4, 0.0, 0.15, 0.3])
    i_p = np.array([0.0, 0.25, 0.05, 0.2])
    n_p = np.sqrt(r_p ** 2 + i_p ** 2)
    length = 1.3
    ours = bl.brown_params(r_p, i_p, n_p, length)
    lv = bmu.brown_params(r_p, i_p, n_p, length=length)
    for k in ("a0", "a1", "a2", "a3"):
        assert _worst(ours[k], lv["a0a1a2a3".index(k) * 0 + ["a0", "a1", "a2", "a3"].index(k)]) < 1e-12


@pytest.mark.skipif(not HAVE_BM, reason="live BerreMueller not importable")
def test_g17b_polarizance_decompose_vs_live():
    """(r_p, i_p, n_p) from (b, d) vs live decompose_polarizance."""
    rng = np.random.default_rng(11)
    b = rng.normal(size=(5, 3)) * 0.3
    d = rng.normal(size=(5, 3)) * 0.3
    pol = bmu.POLARIZANCE(
        np.vstack([b[:, 1], b[:, 0], np.zeros(5)]),   # birefringence rows (lb,lbp,0)
        np.vstack([d[:, 0], d[:, 1], np.zeros(5)]),   # diattenuation rows
        (b + 1j * d).T,
        np.zeros(5),
    )
    _, r_p, i_p, n_p = pol.decompose_polarizance()
    ours = bl.polarizance_decompose(b, d)
    assert _worst(ours["r_p"], r_p) < 1e-12
    assert _worst(ours["i_p"], i_p) < 1e-12
    assert _worst(ours["n_p"], n_p) < 1e-12


# ── G17a: expm path vs Jones group path (same eigen data) ──────────────────

def test_g17a_expm_vs_jones_path():
    for eps, wl in ((EPS_DIA, 600.0), (EPS_UNI, 600.0), (EPS_DIA, 500.0)):
        bd = bl.bulk_differential(eps, ZEROS, ZEROS, EYE, [wl], 700.0)
        assert bd["n_failed"] == 0
        m_expm = bl.mueller_from_diff(bd["b"], bd["d"], bd["absorbance"], 700.0)
        m_dir = bl.mueller_from_jones(bd["jones"])
        assert _worst(m_expm, np.asarray(m_dir, dtype=float).real) < 1e-14, (
            eps[0, 0], wl)


def test_g17a_pasteur_chirality_extension():
    """Route B generalization: Pasteur (rho, rhop) layer at kx = 0 — live's
    elementwise-sqrt path cannot express this. expm == Jones path still at
    machine precision, and the circular-birefringence slot |b[2]| = k0·kappa
    (n± = n ± kappa task-0 spectrum; rotation rate k0·kappa, G16d convention)."""
    kappa = 0.01
    bd = bl.bulk_differential(
        _diag_eps(1.5, 1.5, 1.5),
        -1j * kappa * np.eye(3),     # rho = -i·kappa·I3
        1j * kappa * np.eye(3),      # rhop = +i·kappa·I3 (pasteur conv.)
        EYE,
        [600.0],
        700.0,
    )
    assert bd["n_failed"] == 0
    m_expm = bl.mueller_from_diff(bd["b"], bd["d"], bd["absorbance"], 700.0)
    m_dir = bl.mueller_from_jones(bd["jones"])
    assert _worst(m_expm, np.asarray(m_dir, dtype=float).real) < 1e-14
    k0 = 2 * np.pi / 600.0
    b_arr = np.asarray(bd["b"])[0]
    d_arr = np.asarray(bd["d"])[0]
    # circular-birefringence slot: the Stokes rotation rate equals the
    # circular-eigenmode phase difference 2·k0·kappa (n± = n ± kappa;
    # the Jones polarization itself rotates at k0·kappa — G16d's exact
    # arg-ratio-vs-rotation factor of 2, here measured through the generator)
    assert abs(abs(b_arr[2]) - 2.0 * k0 * kappa) < 1e-12
    assert np.abs(d_arr).max() < 1e-12  # lossless chiral: no dichroism


# ── G17a2: vs the full solver, per-axis Airy interface correction ──────────

def _airy_parts(n0, n_bar, eta_eff, n2, lam, d):
    """Airy decomposition of one slab's scalar transmission (G16a2-validated
    structure): t_full = T_int·e^{ip}/D with the interface product
    T_int = t01·t_exit and the roundtrip denominator D = 1 − r_f·r_b·e^{2ip}.
    Textbook independent construction."""
    eta_a, eta_e, eta_l = 1.0 / n0, 1.0 / n2, eta_eff
    p = 2 * np.pi / lam * n_bar * d
    r_in_front = (eta_a - eta_l) / (eta_a + eta_l)
    r_back = (eta_e - eta_l) / (eta_e + eta_l)
    t01 = 2 * eta_l / (eta_a + eta_l)
    t_exit = 2 * eta_e / (eta_e + eta_l)
    return t01 * t_exit, 1.0 - r_in_front * r_back * np.exp(2j * p), p


def test_g17a2_vs_full_solver_interface_corrected():
    """Single diagonal layer between isotropic windows: x/y chains decouple,
    so the solver's J_trans = diag(t_x, t_y)·(bulk) — dividing out the
    per-axis Airy factors recovers the BULK Jones propagator; its Mueller
    must equal mueller_from_diff at machine precision."""
    for (na, ne, wl, L, ny) in ((1.0, 1.0, 600.0, 700.0, 1.5 + 0.1j),
                                (1.5, 1.2, 600.0, 700.0, 1.5 + 0.1j),
                                (1.0, 1.0, 500.0, 350.0, 1.5 + 0.08j)):
        eps = _diag_eps(1.5 + 0j, ny, 1.7 + 0j)
        st = bl.BerremanStack(
            [bl.Layer(eps=eps, thickness_nm=L)],
            wavelengths_nm=[wl], angles=[0.0], n_entry=na, n_exit=ne,
        )
        res = st.solve()
        jt = np.asarray(res["J_trans"], dtype=complex)
        assert np.abs(jt[0, 1]).max() < 1e-12 and np.abs(jt[1, 0]).max() < 1e-12
        nx = 1.5 + 0j
        tx_int, tx_d, _ = _airy_parts(na, nx, 1.0 / nx, ne, wl, L)
        ty_int, ty_d, _ = _airy_parts(na, ny, 1.0 / ny, ne, wl, L)
        # single-pass bulk = J_trans · D / T_int (drops the interface product
        # AND the Fabry-Pérot denominator, keeps the bulk phase)
        jbulk = np.array([[jt[0, 0] * tx_d / tx_int, 0.0],
                          [0.0, jt[1, 1] * ty_d / ty_int]], dtype=complex)
        m_solver = bl.mueller_from_jones(jbulk).real
        bd = bl.bulk_differential(eps, ZEROS, ZEROS, EYE, [wl], L)
        m_expm = bl.mueller_from_diff(bd["b"], bd["d"], bd["absorbance"], L)
        assert _worst(np.asarray(m_solver, dtype=float).real, m_expm) < 1e-13, (
            na, ne, wl, L)


# ── G17c: reductions ────────────────────────────────────────────────────────

def test_g17c_isotropic_reduction():
    n = 1.5 + 0.05j
    eps = _diag_eps(n, n, n)
    wl, L = 600.0, 800.0
    k0 = 2 * np.pi / wl
    bd = bl.bulk_differential(eps, ZEROS, ZEROS, EYE, [wl], L)
    assert bd["n_failed"] == 0
    assert np.abs(bd["b"]).max() < 1e-14
    assert np.abs(bd["d"]).max() < 1e-14
    assert abs(bd["absorbance"] - 2 * k0 * n.imag) < 1e-13
    m = bl.mueller_from_diff(bd["b"], bd["d"], bd["absorbance"], L)
    expect = np.exp(-bd["absorbance"] * L) * np.eye(4)
    assert np.abs(m - expect).max() < 1e-14
    ph = np.exp(1j * k0 * n * L)
    jj = np.asarray(bd["jones"])
    assert np.abs(jj - ph * np.eye(2)).max() < 1e-13


def test_g17c_pure_retarder_orthogonal_and_v1_metrics():
    B = np.array([0.35, -0.12, 0.0])
    m = np.asarray(bl.mueller_from_diff(B, np.zeros(3), 0.0, 0.7)).reshape(4, 4)
    assert np.abs(m - np.eye(4)).max() > 1e-3            # nontrivial rotation
    assert np.abs(m.T - np.linalg.inv(m)).max() < 1e-12  # orthogonal
    assert np.abs(np.asarray(bl.polarizance(m))).max() < 1e-12
    assert np.abs(np.asarray(bl.diattenuation(m))).max() < 1e-12


def test_g17c_pure_dichroic_closed_form():
    """d = (D,0,0), b = 0: expm gives the textbook [[cosh, sinh],[sinh, cosh]]
    chain (SI §S2 / Gil–Ossikovski differential structure)."""
    D, L = 0.3, 1.1
    m = bl.mueller_from_diff(np.zeros(3), np.array([D, 0.0, 0.0]), 0.0, L)
    dl = D * L
    expect = np.array([
        [np.cosh(dl), np.sinh(dl), 0, 0],
        [np.sinh(dl), np.cosh(dl), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    assert np.abs(m - expect).max() < 1e-14


def test_g17c_diff_matrix_placement_matches_live_transcription():
    """H placement == a numpy transcription of live POLARIZANCE.diff_matrix
    (mueller.py:713-735) — recorded from source, not memory."""
    b = np.array([0.4, -0.15, 0.22])
    d = np.array([0.1, 0.25, -0.3])
    h = bl.diff_mueller_matrix(b, d)
    hh = np.zeros((4, 4))
    hh[0, 1] = hh[1, 0] = d[0]
    hh[0, 2] = hh[2, 0] = d[1]
    hh[0, 3] = hh[3, 0] = d[2]
    hh[1, 2] = -b[2]; hh[2, 1] = b[2]
    hh[1, 3] = b[1];  hh[3, 1] = -b[1]
    hh[2, 3] = -b[0]; hh[3, 2] = b[0]
    assert np.abs(h - hh).max() < 1e-15


# ── G17d: twisted staircase ─────────────────────────────────────────────────

_EPS0 = np.diag([1.7 ** 2, 1.5 ** 2, 1.5 ** 2]).astype(complex)


def _staircase_inputs(npp, wl=600.0, total_nm=350.0, twist=np.pi / 2.0):
    layers = bl.twisted_stack(_EPS0, total_nm, twist, npp)
    eps_j = np.stack([l.eps for l in layers])              # (n,3,3)
    t_j = np.full(npp, total_nm / npp)
    bd = bl.bulk_differential(eps_j,
                              np.zeros((npp, 3, 3), complex),
                              np.zeros((npp, 3, 3), complex),
                              np.tile(np.eye(3, dtype=complex), (npp, 1, 1)),
                              np.full(npp, wl), t_j)
    assert bd["n_failed"] == 0
    m = bl.mueller_from_diff_stack_product(
        bd["b"], bd["d"], bd["absorbance"], t_j, npp)
    return m, eps_j, t_j


def test_g17d_staircase_trotter_self_convergence():
    """Composed differential product over the twisted staircase — Trotter
    self-convergence: staircase(n)⁻¹·staircase(2n) → I, monotone 12/24/48."""
    m12, _, _ = _staircase_inputs(12)
    m24, _, _ = _staircase_inputs(24)
    m48, _, _ = _staircase_inputs(48)
    r1 = np.abs(np.linalg.inv(m12) @ m24 - np.eye(4)).max()
    r2 = np.abs(np.linalg.inv(m24) @ m48 - np.eye(4)).max()
    assert r1 > r2, f"Trotter residual must decrease: {r1} vs {r2}"
    assert r2 < 0.75 * r1, f"residual decay too slow: {r1} vs {r2}"


def test_g17d_staircase_vs_solver_stabilization():
    """staircase(n) vs the SOLVER on the same n-slice stack: the difference is
    dominated by the fixed outer-interface factor; successive differences of
    that difference shrink (O(1/n) stabilization). Documented, NOT a
    machine-precision gate."""
    diffs = []
    prev = None
    for npp in (12, 24, 48, 96):
        m, eps_j, t_j = _staircase_inputs(npp)
        st = bl.BerremanStack(
            [bl.Layer(eps=e, thickness_nm=float(tt)) for e, tt in zip(eps_j, t_j)],
            wavelengths_nm=[600.0], angles=[0.0], n_entry=1.0, n_exit=1.0,
        )
        m_solver = np.asarray(st.solve()["M_trans"], dtype=float).reshape(4, 4)
        d = np.abs(m - m_solver).max()
        if prev is not None:
            diffs.append(abs(d - prev))
        prev = d
    assert diffs[0] > diffs[1] > diffs[2], (
        f"solver-difference must stabilize: {diffs}")


# ── G17e: brown_params small-L Taylor structure (SI S10–S11) ───────────────

def test_g17e_brown_params_small_l_taylor():
    """a0 = 1 + O(L⁴), a1 = L²/2, a2 = L − L³(R²−I²)/6, a3 = −RI·L³/6 through
    the Python wrapper (the SI's second-order consistency claims)."""
    r, i = 0.3, 0.2
    n_p = float(np.hypot(r, i))
    lt = 1e-3
    out = bl.brown_params(r, i, n_p, lt)
    assert abs(out["a0"] - 1.0) < 1e-12
    assert abs(out["a1"] - lt ** 2 / 2) < 1e-12
    assert abs(out["a2"] - (lt - lt ** 3 * (r ** 2 - i ** 2) / 6)) < 1e-15
    assert abs(out["a3"] - (-r * i * lt ** 3 / 6)) < 1e-12
