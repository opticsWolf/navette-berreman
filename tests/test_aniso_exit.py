# SPDX-License-Identifier: LGPL-3.0-or-later
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NOTE: G7b needs the live Berreman4x4 sources: B44_DIR points at the checkout
# root containing the `Berreman4x4` package dir (same pattern as
# FULL_BERREMAN_DIR in test_full.py).
sys.path.insert(0, os.environ.get("B44_DIR", "."))
"""Gate G7: anisotropic exit half-space.

G7a/b bit-identity + flux bridge live in Rust (transfer.rs tests).
G7b: slab exiting into a rotated-uniaxial half-space vs live B44 Structure
     (anisotropic back): Jr + R power direct (same (p,s) layout) <= 1e-9;
     transmission power via B44-side Poynting flux (see LAYOUT NOTE).
G7d: lossless anisotropic-exit energy bounds + flux reporting.
"""
import numpy as np

from navette import berreman as bl
from Berreman4x4 import Berreman4x4 as B44

# LAYOUT NOTE (measured, then explained from source): B44's T_ti carries the
# same (p-out, s-out) x (p-in, s-in) layout as our J_trans, but its VALUES
# differ (direct |dT| ~ 1.4): B44's back-HalfSpace.getTransitionMatrix
# normalizes exit eigenvectors its own way ("Ey = c1 + c2" rescale), while
# ours uses unit-norm numpy-eig columns. Exit-basis amplitudes are
# normalization-dependent, so Jt can only be compared as FIELDS or POWER.
# Reflection is exit-basis independent -> directly comparable (3.3e-16).
# B44's working basis is [Ex, Ey, Hx, Hy] (read off buildDeltaMatrix), hence
# F = 1/2 Re(psi0 psi3* - psi1 psi2*) with basis order (s+,s-,p+,p-).

EPS_SLAB = bl.rot_z(np.diag([2.25, 2.89, 2.25]).astype(complex), np.radians(35.0))
EPS_EXIT = bl.rot_z(np.diag([2.1, 2.7, 2.1]).astype(complex), np.radians(50.0))
WL = 550.0


def b44_flux_T(wl_nm, ang_deg):
    """Transmitted power per input pol from the B44 side (own bases/flux)."""
    k0 = 2 * np.pi / (wl_nm * 1e-9)
    Kx = 1.0 * np.sin(np.radians(ang_deg))
    st = B44.Structure(
        front=B44.IsotropicHalfSpace(B44.IsotropicNonDispersiveMaterial(n=1.0)),
        layers=[B44.HomogeneousLayer(B44.NonDispersiveMaterial(epsilon=EPS_SLAB),
                                     h=250e-9)],
        back=B44.HalfSpace(B44.NonDispersiveMaterial(epsilon=EPS_EXIT)))
    Tri, Tti = (np.asarray(m, dtype=complex) for m in st.getJones(Kx, k0))
    Lb = np.asarray(st.backHalfSpace.getTransitionMatrix(Kx, k0), dtype=complex)
    Lf = np.asarray(st.frontHalfSpace.getTransitionMatrix(Kx, k0), dtype=complex)

    def flux(psi):
        return 0.5 * float(np.real(psi[0] * psi[3].conj() - psi[1] * psi[2].conj()))

    out = {}
    for j, a in ((0, np.array([0, 0, 1, 0], dtype=complex)),   # p incidence
                 (1, np.array([1, 0, 0, 0], dtype=complex))):  # s incidence
        b = np.array([Tti[1, j], 0, Tti[0, j], 0], dtype=complex)  # (s+,s-,p+,p-)
        out[j] = flux(Lb @ b) / flux(Lf @ a)
    return Tri, out[0], out[1]


worst_r = 0.0
worst_t = 0.0
for ang in (0.0, 20.0, 40.0):
    for method in ("scattering", "transfer"):
        ours = bl.BerremanStack([bl.Layer(EPS_SLAB, 250.0)], [WL], [ang],
                                n_entry=1.0, exit_eps=EPS_EXIT,
                                method=method).solve()
        assert ours["n_failed"] == 0, (ang, method)
        assert np.all(np.asarray(ours["power_method"]) == "flux"), (ang, method)
        Tri, Tp_b44, Ts_b44 = b44_flux_T(WL, ang)
        Jr = np.asarray(ours["J_refl"])
        worst_r = max(worst_r, float(np.max(np.abs(Jr - Tri))))
        worst_r = max(worst_r, float(np.max(np.abs(np.asarray(ours["R"])
                                                  - np.abs(Tri) ** 2))))
        T = np.asarray(ours["T"])
        worst_t = max(worst_t, abs(float(T[0, 0]) - Tp_b44),
                      abs(float(T[1, 1]) - Ts_b44))
print(f"G7b Jr+R vs B44 worst: {worst_r:.3e}")
print(f"G7b T (B44-flux) worst: {worst_t:.3e}")
assert worst_r < 1e-9, "G7b reflection failed"
assert worst_t < 1e-9, "G7b transmission failed"
print("G7b PASS")

# ── G9d: B44 cross-check with Method::Exponential ───────────────────────────
# B44 *is* an EM engine (Padé); closes the triangle (SM/TM/EM + B44) using
# the G7b harness (deferred from Phase 3 for exactly this). EM-vs-TM already
# pinned at 8.5e-15, so the budget stays 1e-9 (B44 Padé-7 is the limit).
worst_em = 0.0
for ang in (0.0, 20.0, 40.0):
    ours = bl.BerremanStack([bl.Layer(EPS_SLAB, 250.0)], [WL], [ang],
                            n_entry=1.0, exit_eps=EPS_EXIT,
                            method="exponential").solve()
    assert ours["n_failed"] == 0, ang
    Tri, Tp_b44, Ts_b44 = b44_flux_T(WL, ang)
    Jr = np.asarray(ours["J_refl"])
    T = np.asarray(ours["T"])
    worst_em = max(worst_em, float(np.max(np.abs(Jr - Tri))),
                    abs(float(T[0, 0]) - Tp_b44), abs(float(T[1, 1]) - Ts_b44))
print(f"G9d EM vs B44 worst: {worst_em:.3e}")
assert worst_em < 1e-9, "G9d failed"
print("G9d PASS (triangle closed: SM/TM/EM + B44 agree)")

# ── G7d: lossless anisotropic-exit energy ────────────────────────────────────
res = bl.BerremanStack([bl.Layer(EPS_SLAB, 250.0)], [500.0, 550.0, 650.0],
                       [0.0, 30.0, 60.0], n_entry=1.0, exit_eps=EPS_EXIT,
                       method="scattering").solve()
assert res["n_failed"] == 0
assert np.all(np.asarray(res["power_method"]) == "flux")
R = np.asarray(res["R"])
T = np.asarray(res["T"])
assert np.all(np.isfinite(R)) and np.all(np.isfinite(T))
for col in (0, 1):
    tot = R[:, :, 0, col] + R[:, :, 1, col] + T[:, :, col, col]
    # different exit DOS: bound, don't pin to 1
    assert np.all(tot > -1e-9) and np.all(tot < 1 + 1e-9), tot
print("G7d PASS (R_col + T_col within [-1e-9, 1+1e-9], no NaN, all flux)")
