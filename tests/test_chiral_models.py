# Gates G16a-e: Condon ORD + DBF beta <-> kappa bridge (plan §10.9,
# guide CHIRAL_BRIDGE.md). Analytic-only per §10.1 discipline.
#
# Sources (live-fetched, see CHIRAL_BRIDGE.md §3): DBF definitions from
# Cho arXiv:1501.01078 eqs (2)-(3); DBF eigen-indices n± = n/(1∓x),
# x = beta*k0*n, derived in CHIRAL_BRIDGE.md §4 (plane-wave algebra from
# the cited equations); Condon single-oscillator kappa(omega) =
# omega*R/(omega0^2 - omega^2 - i*omega*Gamma) via the Wikipedia Condon
# model article (Condon-Altar-Eyring 1937; Lindell 1994; Akyurtlu-Werner
# 2004 FDTD-standard form). Our kappa is Task-0-pinned: n± = n ± kappa.
#
# G16a: DBF circular birefringence reproduced EXACTLY through the solver:
#       Jt00/Jt11 == exp(i k0 d * Δn_DBF) with RHS straight from beta
#       (no kappa on that side) + G15b re-pins (t± = t0 e^{±iD}, |t±|=|t0|).
# G16a2: independent DBF-slab reference J_DBF = A_DBF·diag(e^{±i k0 κ d})
#       (scalar Airy with DBF's effective n̄ = n/(1−x²), η_eff = (1−x²)/n):
#       our κ_sym slab matches to the documented O(x²) residue.
# G16a3: discriminator — with the single-branch κ_plus = n/(1−x)−n instead,
#       the rotation is off Δn_DBF by exactly 2k0d·n·x²/(1−x²) (algebra,
#       exact gate) — proves the κ_sym choice is doing real work.
# G16b: kappa_to_dbf_beta roundtrip (machine precision, incl. κ→0, κ<0).
# G16c: weak-chirality limit κ ≈ β·k₀·n² (error ≤ x²·1.1).
# G16d: Condon limits — γ=0 real dtype; static limit κ·λ → const (ω-numerator,
#       no static chirality; property stated in the source); R>0 below
#       resonance ⇒ κ>0 ⇒ arg(tR)-arg(tL) = +k0κd (G15c sense).
# G16e: lossy Condon (Γ>0): complex κ via the tensor path (ρ = −iκI recipe):
#       arg-ratio == 2k0·Re(κ)·d, |ratio| == exp(−2k0·Im(κ)·d), R channel damps.
# Measured (this session): G16a 3.764e-14; G16a2 sanity 9.875e-16, residue
# 4.268e-2 (= x²·k₀·d·n mean-index term, 0.0429 predicted); G16a3 offset
# 0.078737 == algebra EXACTLY; G16b 3.488e-16 (after rationalized inverse);
# G16c 1.0e-4 (= x²); G16d spread 1.5e-5 (= 1/(1−(ω/ω₀)²) at those λ);
# G16e ln|tL/tR| = 2.3e-3 == 2k₀·Im(κ)·d.
"""Gates G16a-e: Condon ORD + DBF beta<->kappa bridge."""
import numpy as np

from navette import berreman as bl

K0 = lambda lam: 2 * np.pi / lam  # vacuum wavenumber, nm^-1


def solve_chiral(n, kappa, d, wl, n0=1.0, n2=None):
    if n2 is None:
        n2 = n0
    lay = bl.chiral_layer(n, kappa, d)
    st = bl.BerremanStack([lay], [wl], [0.0], n_entry=n0, n_exit=n2)
    assert st._use_full
    return st.solve()


def solve_achiral(n, d, wl, n0=1.0, n2=None):
    if n2 is None:
        n2 = n0
    eps = np.diag([n**2] * 3).astype(complex)
    return bl.BerremanStack([bl.Layer(eps, d)], [wl], [0.0],
                            n_entry=n0, n_exit=n2).solve()


# ── G16a: DBF Δn exact through the solver ────────────────────────────────────
worst = 0.0
for n in (1.0, 1.5):
    for x in (0.01, 0.05):
        for d in (300.0, 1000.0):
            for lam in (550.0, 600.0):
                beta = x / (K0(lam) * n)
                kap = bl.dbf_beta_to_kappa(beta, n, lam)
                assert abs(kap - n * x / (1 - x * x)) < 1e-15
                sol = solve_chiral(n, kap, d, lam)
                assert sol["n_failed"] == 0
                ach = solve_achiral(n, d, lam)
                jt = np.asarray(sol["J_trans_circ"])
                t0 = np.asarray(ach["J_trans"])[0, 0]
                d_ind = n / (1 - x) - n / (1 + x)
                ratio = jt[0, 0] / jt[1, 1]
                worst = max(worst, abs(ratio - np.exp(1j * K0(lam) * d * d_ind)))
                assert abs(jt[0, 0] - t0 * np.exp(+1j * K0(lam) * kap * d)) \
                    < 1e-12
                assert abs(abs(jt[0, 0]) - abs(t0)) < 1e-12
print(f"G16a DBF-birefringence-through-solver worst: {worst:.3e}")
assert worst < 1e-13, "G16a failed"
print("G16a PASS")


# ── G16a2: independent DBF-slab reference ────────────────────────────────────
def _airy_t(n0, n_bar, eta_eff, n2, lam, d):
    """Scalar Airy transmission of one slab: interface factors from the
    impedances (eta = 1/n for mu_r=1 ambients, eta_eff given), roundtrip
    phase k0*n_bar*d. Textbook independent construction."""
    eta_a, eta_e, eta_l = 1.0 / n0, 1.0 / n2, eta_eff
    p = K0(lam) * n_bar * d
    # E-amplitude Fresnel: wave in medium 1 hitting 2: r = (eta_2 - eta_1)/(eta_2 + eta_1)
    r_in_front = (eta_a - eta_l) / (eta_a + eta_l)  # front, seen from inside
    r_back = (eta_e - eta_l) / (eta_e + eta_l)      # back, seen from inside
    t01 = 2 * eta_l / (eta_a + eta_l)               # = 1 - r_in_front
    t_exit = 2 * eta_e / (eta_e + eta_l)            # = 1 + r_back
    return (t01 * t_exit * np.exp(1j * p)
            / (1.0 - r_in_front * r_back * np.exp(2j * p)))


# sanity: the Airy construction itself must reproduce the achiral engine
worst_airy = 0.0
for n in (1.5, 2.0):
    for lam in (600.0,):
        for (n0, n2) in ((1.0, 1.0), (1.0, 1.5)):
            ach = solve_achiral(n, 400.0, lam, n0, n2)
            t0 = np.asarray(ach["J_trans"])[0, 0]
            worst_airy = max(worst_airy,
                             abs(t0 - _airy_t(n0, n, 1.0 / n, n2, lam, 400.0)))
print(f"G16a2 Airy-vs-engine sanity: {worst_airy:.3e}")
assert worst_airy < 1e-13, "G16a2 Airy sanity failed"

# DBF slab: n_bar = n/(1-x^2), eta_eff = (1-x^2)/n; per-channel e^{±i k0 κ d}
worst_res = 0.0
for n, n0, n2 in ((1.5, 1.0, 1.0), (1.5, 1.0, 1.5), (2.0, 1.0, 1.0)):
    for x in (0.01, 0.05):
        lam, d = 600.0, 1000.0
        beta = x / (K0(lam) * n)
        kap = bl.dbf_beta_to_kappa(beta, n, lam)
        sol = solve_chiral(n, kap, d, lam, n0, n2)
        jt = np.asarray(sol["J_trans_circ"])
        n_bar = n / (1 - x * x)
        a_dbf = _airy_t(n0, n_bar, (1 - x * x) / n, n2, lam, d)
        want0 = a_dbf * np.exp(+1j * K0(lam) * kap * d)
        want1 = a_dbf * np.exp(-1j * K0(lam) * kap * d)
        worst_res = max(worst_res, abs(jt[0, 0] - want0), abs(jt[1, 1] - want1))
print(f"G16a2 DBF-reference residue (O(x^2) expected): {worst_res:.3e}")
# residue ~ x^2*(k0*d*n + Fresnel) — measured, pinned with margin (a FLOOR,
# not physics; cf. CHIRAL_BRIDGE.md §8.5)
assert worst_res < 1.5 * 0.05**2 * (K0(600.0) * 1000.0 * 2.0 + 2.0), \
    "G16a2 residue larger than the documented O(x^2) floor"
print("G16a2 PASS")


# ── G16a3: κ_plus discriminator (exact algebra) ──────────────────────────────
n, x, lam, d = 1.5, 0.05, 600.0, 1000.0
kappa_plus = n / (1 - x) - n          # single-branch-exact alternative
lay = bl.chiral_layer(n, kappa_plus, d)
sol = bl.BerremanStack([lay], [lam], [0.0]).solve()
jt = np.asarray(sol["J_trans_circ"])
ratio = jt[0, 0] / jt[1, 1]
d_ind = n / (1 - x) - n / (1 + x)
offset = np.angle(ratio * np.exp(-1j * K0(lam) * d * d_ind))
want_off = 2 * K0(lam) * d * n * x * x / (1 - x * x)
print(f"G16a3 kappa_plus offset: {offset:.6f} vs algebra {want_off:.6f}")
assert abs(offset - want_off) < 1e-12, "G16a3 discriminator failed"
print("G16a3 PASS")


# ── G16b: beta <-> kappa roundtrip ───────────────────────────────────────────
worst = 0.0
for n in (1.0, 1.5, 2.0):
    for lam in (300.0, 600.0):
        k0 = K0(lam)
        for x in (-0.19, -0.01, 0.0, 0.01, 0.19):
            beta = x / (k0 * n)
            kap = bl.dbf_beta_to_kappa(beta, n, lam)
            beta2 = bl.kappa_to_dbf_beta(kap, n, lam)
            worst = max(worst, abs(beta2 - beta) / max(abs(beta), 1e-300))
print(f"G16b roundtrip worst rel: {worst:.3e}")
assert worst < 1e-14, "G16b failed"
print("G16b PASS")


# ── G16c: weak-chirality limit ───────────────────────────────────────────────
worst = 0.0
for n in (1.0, 1.5):
    for x in (1e-4, 1e-3, 1e-2):
        lam = 550.0
        beta = x / (K0(lam) * n)
        kap = bl.dbf_beta_to_kappa(beta, n, lam)
        lin = beta * K0(lam) * n * n
        worst = max(worst, abs(kap - lin) / abs(lin))
print(f"G16c weak-limit worst rel: {worst:.3e}")
assert worst < 0.02 * 0.02 * 1.1 + 1e-15, "G16c failed"
print("G16c PASS")


# ── G16d: Condon limits + sign discipline ────────────────────────────────────
lam0 = 200.0  # UV resonance
R = 3e14
kap0 = bl.condon_kappa([400.0, 600.0, 800.0], R, lam0)
assert kap0.dtype == np.float64, kap0.dtype          # (i) gamma=0 real
big = bl.condon_kappa([5e4, 1e5, 2e5], R, lam0)      # (ii) static limit
prod = big * np.array([5e4, 1e5, 2e5])
spread = (prod.max() - prod.min()) / abs(prod[0])
# kappa*lambda is constant to O((lambda0/lambda)^2) — bound 2*(lam0/5e4)^2
print(f"G16d static-limit spread: {spread:.3e}")
assert spread < 2.2 * (lam0 / 5e4) ** 2, "G16d static failed"
# leading-order form kappa ~ omega*R/(omega0^2 - omega^2) at omega << omega0:
om_stat = 2 * np.pi * bl._C0_NM_S / 5e4
om0 = 2 * np.pi * bl._C0_NM_S / lam0
expect = om_stat * R / (om0**2 - om_stat**2)
assert abs(big[0] - expect) / abs(expect) < 1e-13
# kappa -> 0 linearly in omega: kappa(2e5) ~ kappa(600nm)*600/2e5*(1+O((w/w0)^2))
kappa_vis = float(bl.condon_kappa([600.0], R, lam0)[0])
ratio = abs(big[2]) / abs(kappa_vis)
assert abs(ratio - (600.0 / 2e5) * (1 - (lam0 / 600.0) ** 2)
           / (1 - (lam0 / 2e5) ** 2)) < 1e-6
assert kap0[1] > 0                                    # (iii) R>0 => kappa>0
lam, d = 600.0, 500.0
kap = float(bl.condon_kappa([lam], R, lam0)[0])
sol = bl.BerremanStack([bl.chiral_layer(1.5, kap, d)], [lam], [0.0]).solve()
jt = np.asarray(sol["J_trans_circ"])
rot = np.angle(jt[0, 0] / jt[1, 1]) / 2.0   # rotation = half the arg-ratio
assert abs(rot - K0(lam) * kap * d) < 1e-12, rot      # (iv) G15c sense
print("G16d PASS")


# ── G16e: lossy Condon — dichroism sign through the full engine ──────────────
lam = 600.0
gam = 0.05 * 2 * np.pi * bl._C0_NM_S / lam0  # ~5% of omega0, rad/s
kap_c = complex(bl.condon_kappa([lam], R, lam0, gamma=gam)[0])
assert kap_c.imag > 0, kap_c
n, d = 1.5, 500.0
eps = np.broadcast_to(np.eye(3) * n * n, (1, 3, 3)).astype(complex).copy()
rho = (-1j * kap_c * np.eye(3))[None, ...]
rhop = (+1j * kap_c * np.eye(3))[None, ...]
st = bl.BerremanStack([bl.Layer(eps, d, rho=rho, rhop=rhop)], [lam], [0.0])
assert st._use_full
sol = st.solve()
assert sol["n_failed"] == 0
jt = np.asarray(sol["J_trans_circ"])
rot = np.angle(jt[0, 0] / jt[1, 1]) / 2.0   # rotation = half the arg-ratio
att = np.log(abs(jt[1, 1]) / abs(jt[0, 0]))
assert abs(rot - K0(lam) * kap_c.real * d) < 1e-12, rot
assert abs(att - 2 * K0(lam) * kap_c.imag * d) < 1e-9, att
assert att > 0, "Im(kappa) > 0 must damp the R channel"
print(f"G16e dichroism: ln|tL/tR| = {att:.4f} over {d:.0f} nm")
print("G16e PASS")

print("G16 ALL PASS")
