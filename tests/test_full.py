# SPDX-License-Identifier: LGPL-3.0-or-later
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ref_pyllama.py
# NOTE: requires full_berreman.py importable (set FULL_BERREMAN_DIR or place beside this file)
sys.path.insert(0, os.environ.get("FULL_BERREMAN_DIR", "."))
"""Validate the FULL magneto-optic Berreman path.

(1) Reduction: with rho = rho' = 0 and mu = I the full path must equal the
    reduced ("simple") path.
(2) Correctness: compare the full solve against a NumPy reference whose layer
    Delta matrix is built by the authoritative full_berreman.calc_berreman_matrix.
"""
import sys
import numpy as np
from numpy import linalg as la


from navette import berreman as bl
import ref_pyllama as ref
import full_berreman as fb_raw


def fb(eps, rho, rhop, mu, K_x):
    """full_berreman with the determinant-division typo corrected (num/det),
    which is the physically-intended form (reduces to the simple matrix)."""
    e, r, rp, m = (np.asarray(x, complex) for x in (eps, rho, rhop, mu))
    det = e[2, 2] * m[2, 2] - r[2, 2] * rp[2, 2]
    a1 = (rp[2, 0] * r[2, 2] - e[2, 0] * m[2, 2]) / det
    a2 = ((rp[2, 1] - K_x) * r[2, 2] - e[2, 1] * m[2, 2]) / det
    a3 = (m[2, 1] * r[2, 2] - r[2, 0] * m[2, 2]) / det
    a4 = (m[2, 1] * r[2, 2] - (r[2, 1] + K_x) * m[2, 2]) / det
    a5 = (rp[2, 2] * e[2, 0] - e[2, 2] * rp[2, 0]) / det
    a6 = (rp[2, 2] * e[2, 1] - (rp[2, 1] - K_x) * e[2, 2]) / det
    a7 = (rp[2, 2] * r[2, 1] - e[2, 2] * m[2, 0]) / det
    a8 = ((r[2, 1] + K_x) * rp[2, 2] - e[2, 2] * m[2, 1]) / det
    B = np.zeros((4, 4), complex)
    rpk, rmk = rp[1, 2] + K_x, r[1, 2] - K_x
    B[0, 0] = rp[1, 0] + rpk * a1 + m[1, 2] * a5
    B[0, 1] = m[1, 1] + rpk * a4 + m[1, 2] * a8
    B[0, 2] = rp[1, 1] + rpk * a2 + m[1, 2] * a6
    B[0, 3] = -(m[1, 0] + rpk * a3 + m[1, 2] * a7)
    B[1, 0] = e[0, 0] + e[0, 2] * a1 + rp[0, 2] * a5
    B[1, 1] = r[0, 1] + e[0, 2] * a4 + rp[0, 2] * a8
    B[1, 2] = e[0, 1] + e[0, 2] * a2 + rp[0, 2] * a6
    B[1, 3] = -(r[0, 0] + e[0, 2] * a3 + rp[0, 2] * a7)
    B[2, 0] = -(rp[0, 0] + rp[0, 2] * a1 + m[0, 2] * a5)
    B[2, 1] = -(m[0, 1] + rp[0, 2] * a4 + m[0, 2] * a8)
    B[2, 2] = -(rp[0, 1] + rp[0, 2] * a2 + m[0, 2] * a6)
    B[2, 3] = m[0, 0] + rp[0, 2] * a3 + m[0, 2] * a7
    B[3, 0] = e[1, 0] + e[1, 2] * a1 + rmk * a5
    B[3, 1] = r[1, 1] + e[1, 2] * a4 + rmk * a8
    B[3, 2] = e[1, 1] + e[1, 2] * a2 + rmk * a6
    B[3, 3] = -(r[1, 0] + e[1, 2] * a3 + rmk * a7)
    return B


def ref_solve_full(eps, rho, rhop, mu, t, eps_entry, eps_exit, wl, theta, method):
    """Single-layer reference solve using full_berreman for D."""
    n_entry, n_exit = np.sqrt(eps_entry), np.sqrt(eps_exit)
    k0 = 2 * np.pi / wl
    Kx = n_entry * np.sin(theta)
    Kz_entry = n_entry * np.cos(theta)
    theta_out = np.arcsin((n_entry / n_exit) * np.sin(theta + 0j))
    Kz_exit = n_exit * np.cos(theta_out)

    entry = ref.HalfSpace(np.diag([eps_entry] * 3), Kx, Kz_entry, k0)
    exit_ = ref.HalfSpace(np.diag([eps_exit] * 3), Kx, Kz_exit, k0)

    # Build a ref.Layer but override its Delta with the full Berreman matrix.
    L = ref.Layer.__new__(ref.Layer)
    L.eps = np.asarray(eps, dtype=complex)
    L.thickness = t
    L.Kx = Kx
    L.k0 = k0
    L.D = fb(eps, rho, rhop, mu, Kx)
    p, q, _ = L._calc_p_q_sorted()
    L.eigenvectors, L.eigenvalues = p, q
    L.P, L.Q = L.build_P_Q()

    if method == "TM":
        T = la.multi_dot((L.P, L.Q, la.inv(L.P)))
        TM = la.multi_dot((la.inv(exit_.P), T, entry.P))
        deno = TM[2, 2] * TM[3, 3] - TM[3, 2] * TM[2, 3]
        r_pp = (TM[3, 0] * TM[2, 3] - TM[2, 0] * TM[3, 3]) / deno
        r_ps = (TM[2, 0] * TM[3, 2] - TM[3, 0] * TM[2, 2]) / deno
        r_sp = (TM[3, 1] * TM[2, 3] - TM[2, 1] * TM[3, 3]) / deno
        r_ss = (TM[2, 1] * TM[3, 2] - TM[3, 1] * TM[2, 2]) / deno
        t_pp = TM[0, 0] + TM[0, 2] * r_pp + TM[0, 3] * r_ps
        t_ps = TM[1, 0] + TM[1, 2] * r_pp + TM[1, 3] * r_ps
        t_sp = TM[0, 1] + TM[0, 2] * r_sp + TM[0, 3] * r_ss
        t_ss = TM[1, 1] + TM[1, 2] * r_sp + TM[1, 3] * r_ss
        J_refl = np.array([[r_pp, r_sp], [r_ps, r_ss]])
        J_trans = np.array([[t_pp, t_sp], [t_ps, t_ss]])
    else:
        S = ref.s_combine(ref.s_to_next(entry, L), ref.s_to_next(L, exit_))
        J_refl = np.array([[S[2, 0], S[2, 1]], [S[3, 0], S[3, 1]]])
        J_trans = np.array([[S[0, 0], S[0, 1]], [S[1, 0], S[1, 1]]])
    factor = (Kz_exit / Kz_entry).real
    return J_refl, J_trans, np.abs(J_refl) ** 2, factor * np.abs(J_trans) ** 2


# ── (1) reduction check ─────────────────────────────────────────────────────
eps = bl.rot_z(np.diag([2.25, 2.89, 2.4]).astype(complex), np.radians(30.0))
I3 = np.eye(3, dtype=complex)
Z3 = np.zeros((3, 3), dtype=complex)

simple = bl.BerremanStack([bl.Layer(eps, 240.0)], [550.0], [25.0],
                          n_entry=1.0, n_exit=1.0, method="scattering").solve()
full_red = bl.BerremanStack([bl.Layer(eps, 240.0, rho=Z3, rhop=Z3, mu=I3)],
                            [550.0], [25.0], n_entry=1.0, n_exit=1.0,
                            method="scattering").solve()
red_err = max(np.max(np.abs(simple["R"] - full_red["R"])),
              np.max(np.abs(simple["T"] - full_red["T"])))
print(f"(1) full(rho=0,mu=I) vs simple, max|ΔR,ΔT| = {red_err:.3e}")

# ── (2) genuine magneto-optic correctness ───────────────────────────────────
# gyrotropic antisymmetric magneto-electric coupling + anisotropic mu
g = 0.03
rho = np.array([[0, 1j * g, 0], [-1j * g, 0, 0.5j * g], [0, -0.5j * g, 0]], dtype=complex)
rhop = -rho.conj().T * 0.8
mu = np.diag([1.05, 1.0, 0.98]).astype(complex)

worst = 0.0
for method, m in (("scattering", "SM"), ("transfer", "TM")):
    res = bl.BerremanStack([bl.Layer(eps, 240.0, rho=rho, rhop=rhop, mu=mu)],
                           [600.0], [35.0], n_entry=1.0, n_exit=1.3,
                           method=method).solve()
    Jr, Jt, R, T = ref_solve_full(eps, rho, rhop, mu, 240.0, 1.0, 1.3 ** 2,
                                  600.0, np.radians(35.0), m)
    worst = max(worst, np.max(np.abs(res["R"] - R)), np.max(np.abs(res["T"] - T)))
    for a, b in ((res["J_refl"], Jr), (res["J_trans"], Jt)):
        a = a.flatten(); b = b.flatten()
        k = int(np.argmax(np.abs(b))); ph = b[k] / a[k]; ph /= abs(ph)
        worst = max(worst, np.max(np.abs(a * ph - b)))

print(f"(2) full magneto-optic vs full_berreman.py reference, worst |Δ| = {worst:.3e}")
assert red_err < 1e-12 and worst < 1e-6
print("\nFULL-PATH PASS")
