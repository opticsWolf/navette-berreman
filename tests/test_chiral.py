# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gate G15: chiral media (Phase 10, Pasteur-Tellegen primary).

Task-0 spectrum/handedness pins live in Rust
(berreman::tests::pasteur_task0_*) — referenced, not repeated.
G15a: kappa = 0 reduction full vs simple; det-singularity graceful NaN.
G15b: normal-incidence Pasteur slab vs textbook single-slab Airy per circular
      channel (J_circ[0,0] <-> n+kappa, [1,1] <-> n-kappa); cross-circular 0.
G15c: optical rotation from arg(tR)-arg(tL) vs FULL Airy (complex-ratio form,
      no branch cuts); kappa-antisymmetry; per-channel energy.
Oblique: kappa-even/odd symmetry, reciprocity (stack reversal), energy.
G15e: chiral absorption balance (pins from_psi_full Ez end-to-end; oblique so
      Ez != 0 — the old simple-constitutive unpack fails this by ~1e-4).
Analytic-only per §10.1 (live full_berreman carries the det typo: with
rho22*rhop22 = k^2 != 0 its a_i differ from ours by det^2 = (eps*mu-k^2)^2
in EVERY a_i — shape recorded here as the expected cross-check signature,
never an oracle, never a failure).
"""
import warnings

import numpy as np

from navette import berreman as bl


def solve_chiral(n, kappa, d, wl, theta, n0=1.0, n2=None, mu=1.0):
    """One chiral slab solve; returns squeezed result dict."""
    if n2 is None:
        n2 = n0
    lay = bl.chiral_layer(n, kappa, d, mu=mu)
    st = bl.BerremanStack([lay], [wl], [theta], n_entry=n0, n_exit=n2)
    assert st._use_full
    return st.solve()


# ── G15a: reduction + singularity ────────────────────────────────────────────
# kappa = 0: full path vs simple path. NOT exactly 0.0 (draft said 0.0), and
# NOT fp noise either: the isotropic-full spectrum is exactly degenerate, so
# the cluster arm repairs vectors (Rayleigh residual ~1e-11 floor of
# shift-invert near a double eigenvalue) and substitutes the quotient — the
# residual q error ~3e-11 scales LINEARLY in k0*d (measured 3.3e-11/1.6e-10/
# 3.7e-10 at d=100/500/1000). Gate <= 1e-9 documents this honest floor
# (pre-fix: 0.78 silent garbage). Anisotropic kappa=0 below is exactly 0.0.
r_full = solve_chiral(1.5, 0.0, 500.0, 600.0, 10.0)["J_trans"]
eps = np.diag([1.5 ** 2] * 3).astype(complex)
r_simp = bl.BerremanStack(
    [bl.Layer(eps, 500.0)], [600.0], [10.0]).solve()["J_trans"]
d_red = float(np.max(np.abs(np.asarray(r_full) - np.asarray(r_simp))))
print(f"G15a kappa=0 reduction (iso, degenerate floor): {d_red:.3e}")
assert d_red < 1e-9, "G15a reduction failed"
# anisotropic kappa = 0: non-degenerate, singleton path untouched -> exact 0.0.
eps_a = np.diag([2.89, 2.25, 2.25]).astype(complex)
I3 = np.eye(3)
Z3 = np.zeros((3, 3))
fa = bl.BerremanStack(
    [bl.Layer(eps_a, 500.0, rho=Z3, rhop=Z3, mu=I3)], [600.0], [10.0]).solve()
sa = bl.BerremanStack(
    [bl.Layer(eps_a, 500.0)], [600.0], [10.0]).solve()
d_red_a = float(np.max(np.abs(np.asarray(fa["J_trans"])
                              - np.asarray(sa["J_trans"]))))
print(f"G15a kappa=0 reduction (aniso): {d_red_a:.3e}")
assert d_red_a == 0.0, "G15a aniso reduction failed"
# det singularity k^2 = n^2: graceful NaN + n_failed, never a panic.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # expected |k|>=0.2n regime warning
    bad = solve_chiral(1.5, 1.5, 500.0, 600.0, 10.0)
assert bad["n_failed"] > 0, "G15a singularity not flagged"
assert bool(np.isnan(np.asarray(bad["R"])).any()), "G15a singularity not NaN"
print(f"G15a singularity: n_failed={bad['n_failed']}, NaN ok")
print("G15a PASS")

# ── G15b: helicity-channel relations (CORRECTED reference) ─────────────────
# P10 FINDING — the draft's "textbook single-slab Airy with its own index
# n±" is WRONG (disproven at 0.4: co-channel vs Airy(n±) differs O(1)). The
# TRUE relations, exact to ~1e-15 across 4 thicknesses (measured below):
#   t_R = t0*e^{+iD}, t_L = t0*e^{-iD},  D = k0*kappa*d   (transmission)
#   Jr[1,0] = Jr[0,1] = r0                               (reflection)
#   all other circ entries ~0 (no helicity mixing in T, none preserved in R)
# where (r0, t0) = the ACHIRAL slab (same geometry, simple path — itself
# live-validated). Physics: reflection is a round trip (forward +D and
# return −D chiral phases cancel — J_z conservation forces helicity flip);
# transmission is one-way (phase ±D survives). CORROBORATION (not circular):
# (i) INDEPENDENT numpy/scipy-expm replication of the full Delta chain
# reproduces the solver exactly (no shared code); (ii) symmetry structure
# (diagonal-T, antidiagonal-R) is group-theoretically forced; (iii) energy
# |t±| = |t0|, |r| = |r0| per channel; (iv) rotation/antisymmetry (G15c).
# Literature tertiary citation is a review checkpoint (same bar as Condon).
def solve_achiral(n, d, wl, theta, n0=1.0, n2=None):
    if n2 is None:
        n2 = n0
    eps = np.diag([n ** 2] * 3).astype(complex)
    return bl.BerremanStack(
        [bl.Layer(eps, d)], [wl], [theta], n_entry=n0, n_exit=n2).solve()


worst_t, worst_r, worst_forbid = 0.0, 0.0, 0.0
for d in (200.0, 500.0, 1000.0):
    for kappa in (0.01, 0.05):
        for n in (1.0, 1.5):
            for (n0, n2) in ((1.0, 1.0), (1.0, 1.5)):
                sol = solve_chiral(n, kappa, d, 600.0, 0.0, n0, n2)
                assert sol["n_failed"] == 0
                ach = solve_achiral(n, d, 600.0, 0.0, n0, n2)
                jt = np.asarray(sol["J_trans_circ"])
                jr = np.asarray(sol["J_refl_circ"])
                t0 = np.asarray(ach["J_trans"])[0, 0]
                r0 = np.asarray(ach["J_refl"])[0, 0]
                D = (2 * np.pi / 600.0) * kappa * d
                worst_t = max(worst_t, abs(jt[0, 0] - t0 * np.exp(+1j * D)),
                              abs(jt[1, 1] - t0 * np.exp(-1j * D)))
                worst_r = max(worst_r, abs(jr[1, 0] - r0), abs(jr[0, 1] - r0))
                worst_forbid = max(worst_forbid, abs(jt[0, 1]), abs(jt[1, 0]),
                                   abs(jr[0, 0]), abs(jr[1, 1]))
print(f"G15b trans-phase: {worst_t:.3e}, refl: {worst_r:.3e}, "
      f"forbidden: {worst_forbid:.3e}")
assert worst_t < 1e-12, "G15b trans failed"
assert worst_r < 1e-12, "G15b refl failed"
# forbidden entries cancel O(1) sandwich terms to ~3e-14 (draft said 1e-15;
# fp-cancellation floor, measured) — still 12 orders below any signal.
assert worst_forbid < 1e-12, "G15b forbidden failed"
print("G15b PASS")

# ── G15c: optical rotation ───────────────────────────────────────────────────
# t_R/t_L = e^{2iD} EXACTLY (achiral t0 cancels — holds for asymmetric
# ambients too since the achiral slab is helicity-blind). The draft's warning
# ("do NOT gate against bare-propagation -k0 d k") assumed textbook-Airy
# per-channel Fresnel phases; under the TRUE relations the Fresnel part is
# COMMON and cancels, promoting bare-phase to an exact gate (plan §10.4
# correction). Complex-ratio form: no arg branch cuts.
worst_rot, worst_anti, worst_en = 0.0, 0.0, 0.0
for d in (500.0, 1000.0):
    for kappa in (0.01, 0.05):
        n = 1.5
        sol = solve_chiral(n, kappa, d, 600.0, 0.0)
        sol_m = solve_chiral(n, -kappa, d, 600.0, 0.0)
        jt = np.asarray(sol["J_trans_circ"])
        jt_m = np.asarray(sol_m["J_trans_circ"])
        rho_num = jt[0, 0] / jt[1, 1]
        D = (2 * np.pi / 600.0) * kappa * d
        worst_rot = max(worst_rot, abs(rho_num - np.exp(2j * D)))
        rot_deg = float(np.angle(rho_num) / 2 * 180 / np.pi)
        print(f"  d={d:.0f} k={kappa}: rotation {rot_deg:.6f} deg "
              f"(k0 d k = {D * 180 / np.pi:.6f})")
        rho_m = jt_m[0, 0] / jt_m[1, 1]
        worst_anti = max(worst_anti, abs(rho_num * rho_m - 1.0))
        jr = np.asarray(sol["J_refl_circ"])
        # per-helicity-channel energy: |t±|² + |r|² = 1 (lossless).
        for tch, rch in ((jt[0, 0], jr[1, 0]), (jt[1, 1], jr[0, 1])):
            worst_en = max(worst_en,
                           abs(abs(tch) ** 2 + abs(rch) ** 2 - 1.0))
print(f"G15c rotation ratio: {worst_rot:.3e}, antisym: {worst_anti:.3e}, "
      f"energy: {worst_en:.3e}")
assert worst_rot < 1e-12, "G15c rotation failed"
assert worst_anti < 1e-12, "G15c antisymmetry failed"
assert worst_en < 1e-9, "G15c energy failed"
print("G15c PASS")

# ── Oblique symmetries + reciprocity ─────────────────────────────────────────
th = 40.0
sp = solve_chiral(1.5, 0.03, 500.0, 600.0, th)
sm = solve_chiral(1.5, -0.03, 500.0, 600.0, th)
s0 = solve_chiral(1.5, 0.0, 500.0, 600.0, th)
jp, jm, j0 = (np.asarray(s["J_refl"]) for s in (sp, sm, s0))
# cross-pol odd in k, co-pol even (symmetric slab, entry = exit).
odd = max(abs(jp[0, 1] + jm[0, 1]), abs(jp[1, 0] + jm[1, 0]))
even = max(abs(jp[0, 0] - jm[0, 0]), abs(jp[1, 1] - jm[1, 1]))
# k -> 0 limit recovers the achiral Jones.
s_tiny = solve_chiral(1.5, 1e-9, 500.0, 600.0, th)
d_tiny = float(np.max(np.abs(np.asarray(s_tiny["J_refl"]) - j0)))
print(f"oblique: cross-odd {odd:.3e}, co-even {even:.3e}, k->0 {d_tiny:.3e}")
assert odd < 1e-12 and even < 1e-12 and d_tiny < 1e-9
# Lorentz reciprocity at AMPLITUDE level (corrected twice): for reciprocal
# media the transmission Jones TRANSPOSES under source/detector swap —
# t_{R->L}(S) = t_{L->R}(S)^T, and R-incidence on S == L-incidence on
# mirror(S) = reversal + kappa negation (mirrored helix = enantiomer). Hence
# J_trans(mirror) == J_trans(fwd).T EXACTLY (complex). Reflection has NO such
# relation (r_L vs r_R phases unconstrained; powers differ) — correctly
# ungated. The draft's power-equality probe was wrong physics (twice over:
# naive reversal ignores enantiomer-flip AND powers transpose); the solver
# was right all along (diagonals visibly identical, off-diagonals swapped).
l1 = bl.chiral_layer(1.5, 0.04, 300.0)
l2 = bl.chiral_layer(1.7, -0.02, 250.0)
l1m = bl.chiral_layer(1.5, -0.04, 300.0)
l2m = bl.chiral_layer(1.7, +0.02, 250.0)
mk = lambda ls: bl.BerremanStack(ls, [600.0], [th]).solve()  # noqa: E731
f, r = mk([l1, l2]), mk([l2m, l1m])
d_rec = float(np.max(np.abs(np.asarray(r["J_trans"])
                            - np.asarray(f["J_trans"]).T)))
print(f"Lorentz reciprocity J_trans(mirror) == J_trans(fwd).T: {d_rec:.3e}")
assert d_rec < 1e-12, "reciprocity failed"
# non-vacuity: naive (unnegated) reversal genuinely differs in power —
# chirality breaks mirror symmetry (pins the test is not vacuous).
rn = mk([l2, l1])
d_naive = float(np.max(np.abs(np.asarray(f["T"]) - np.asarray(rn["T"]))))
print(f"naive reversal power difference (must be >> 0): {d_naive:.3e}")
assert d_naive > 1e-6, "naive reversal unexpectedly matched (vacuous?)"
# energy bounds (flux arm): R_col + T_col in [-1e-9, 1+1e-9].
rt = np.asarray(f["R"]) + np.asarray(f["T"])
assert bool(np.all(rt >= -1e-9) and np.all(rt <= 1 + 1e-9)), "energy bounds"
print("oblique PASS")

# ── G15e: chiral absorption balance (oblique => Ez != 0) ─────────────────────
n_c = 1.5 + 0.05j  # lossy background + real kappa (rho anti-Hermitian:
# no kappa dissipation, so the Im(eps)-only absorption_density is complete)
lay = bl.Layer(eps=np.diag([n_c ** 2] * 3).astype(complex), thickness_nm=300.0,
               rho=-0.05j * np.eye(3), rhop=+0.05j * np.eye(3),
               mu=np.eye(3, dtype=complex))
st = bl.BerremanStack([lay], [600.0], [30.0], n_entry=1.0, n_exit=1.0)
sol = st.solve()
assert sol["n_failed"] == 0
zz_full = np.linspace(0.0, 300.0, 2002)
# ENDPOINT DISCIPLINE (P10 finding): z == total returns EXIT-side values;
# Ez jumps by eps_ratio across the interface (normal-D continuity — physical:
# observed |Ez|^2 ratio 5.06 == |n_c^2|^2). The absorption integral needs
# slab-side limits over the FULL [0, total]: drop exactly-total, append the
# 300-minus-epsilon limit point (last panel keeps O(h^2) accuracy).
zz = np.concatenate([zz_full[:-1], [300.0 - 1e-9]])
f = st.fields(zz)
assert f["n_failed"] == 0
C0 = 2.998e8
omega = 2 * np.pi * C0 / (600.0 * 1e-9)
eps_im = (n_c ** 2).imag
def _flux(E, H):
    return 0.5 * np.real(E[..., 0] * np.conj(H[..., 1])
                         - E[..., 1] * np.conj(H[..., 0]))


worst_e = 0.0
for fld in ("E_p", "E_s"):
    E = np.asarray(f[fld])
    A = 0.5 * (omega / C0) * eps_im * np.sum(np.abs(E) ** 2, axis=-1)
    A_int = float(np.trapezoid(A, zz)) * 1e-9
    hfld = "H_" + fld.split("_")[1]
    F_in = _flux(np.asarray(st.fields([-50.0])[fld])[0],
                 np.asarray(st.fields([-50.0])[hfld])[0])
    F_out = _flux(np.asarray(st.fields([350.0])[fld])[0],
                  np.asarray(st.fields([350.0])[hfld])[0])
    worst_e = max(worst_e, abs((A_int + F_out) / F_in - 1.0))
print(f"G15e chiral absorption balance: {worst_e:.3e}")
assert worst_e < 1e-6, "G15e failed"
print("G15e PASS")

# ── constructor validation ───────────────────────────────────────────────────
for bad_call in (lambda: bl.chiral_layer(0.0, 0.01, 100.0),
                 lambda: bl.chiral_layer(1.5, np.nan, 100.0),
                 lambda: bl.kappa_table([500.0, 400.0], [0.01, 0.02]),
                 lambda: bl.kappa_table([500.0], [0.01, 0.02]),
                 lambda: bl.twisted_stack(np.eye(2), 100.0, 1.0, 8),
                 lambda: bl.twisted_stack(np.eye(3), 100.0, 1.0, 0),
                 lambda: bl.twisted_stack(np.eye(3), 100.0, 1.0, 8, grid="xx")):
    try:
        bad_call()
        raise SystemExit("MISSING-ERROR: bad constructor accepted")
    except ValueError:
        pass
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    bl.chiral_layer(1.5, 0.4, 100.0)  # |k| >= 0.2n regime warning
    assert any("0.2*n" in str(x.message) for x in w), "regime warning missing"
wl, ka = bl.kappa_table([400.0, 600.0], [0.01, 0.02])
assert wl.shape == ka.shape == (2,)
print("constructors PASS")
