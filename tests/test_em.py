import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests dir
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref"))
# NOTE: G9c needs the live BerreMueller sources (pyllama.Structure EM arm).
sys.path.insert(0, os.environ.get("FULL_BERREMAN_DIR", "."))
"""Gate G9: exponential-matrix (EM) method.

G9a: expm unit tests live in src/expm.rs (cargo) — not repeated here.
G9b: EM == TM on the 5 canonical cases (simple path) + a gyrotropic case
     (full path): Jones <= 1e-12 (looser than SM/TM by design — independent
     algorithms sharing no code with eig4).
G9c: EM vs live pyllama method="EM" on iso + rotated slabs <= 1e-9.
G9d: B44 cross-check — DEFERRED to Phase 1 (reuses the G7b harness).
"""
import numpy as np

from navette import berreman as bl
import ref_pyllama as ref
from berremueller import pyllama as live_pl

# ── G9b: EM == TM ────────────────────────────────────────────────────────────
worst = 0.0
for name, layers, n_in, n_out, wl, theta in ref.test_cases():
    bl_layers = [bl.Layer(eps, d) for eps, d in layers]
    em = bl.BerremanStack(bl_layers, [wl], [np.degrees(theta)],
                          n_entry=n_in, n_exit=np.sqrt(n_out),
                          method="exponential").solve()
    tm = bl.BerremanStack(bl_layers, [wl], [np.degrees(theta)],
                          n_entry=n_in, n_exit=np.sqrt(n_out),
                          method="transfer").solve()
    assert em["n_failed"] == 0 and tm["n_failed"] == 0, name
    for f in ("J_refl", "J_trans", "R", "T"):
        worst = max(worst, float(np.max(np.abs(em[f] - tm[f]))))
    print(f"G9b {name}: EMvsTM done")

# full (magneto-optic) arm: gyrotropic rho/rhop + anisotropic mu
g = 0.03
rho = np.array([[0, 1j * g, 0], [-1j * g, 0, 0.5j * g],
                [0, -0.5j * g, 0]], dtype=complex)
rhop = -rho.conj().T * 0.8
mu = np.diag([1.05, 1.0, 0.98]).astype(complex)
eps_mo = bl.rot_z(np.diag([2.25, 2.89, 2.25]).astype(complex), np.radians(20.0))
for meth_pair in (("exponential", "transfer"),):
    a = bl.BerremanStack([bl.Layer(eps_mo, 240.0, rho=rho, rhop=rhop, mu=mu)],
                         [600.0], [35.0], n_entry=1.0, n_exit=1.3,
                         method=meth_pair[0]).solve()
    b = bl.BerremanStack([bl.Layer(eps_mo, 240.0, rho=rho, rhop=rhop, mu=mu)],
                         [600.0], [35.0], n_entry=1.0, n_exit=1.3,
                         method=meth_pair[1]).solve()
    for f in ("J_refl", "J_trans", "R", "T"):
        worst = max(worst, float(np.max(np.abs(a[f] - b[f]))))
print(f"G9b EM==TM worst (5 canonical + MO): {worst:.3e}")
assert worst < 1e-12, "G9b failed"
print("G9b PASS")

# ── G9c: EM vs live pyllama EM ───────────────────────────────────────────────
# Live Structure takes NORMALIZED wavevectors (Kx, Kz, k0); same construction
# as ref_pyllama.solve, then get_fresnel(method="EM").
worst_c = 0.0
for eps_l, d, wl, ang in (
    (np.diag([2.25, 2.25, 2.25]).astype(complex), 200.0, 550.0, 10.0),
    (bl.rot_z(np.diag([2.25, 2.89, 2.25]).astype(complex), np.radians(35.0)),
     250.0, 633.0, 30.0),
):
    ours = bl.BerremanStack([bl.Layer(eps_l, d)], [wl], [ang],
                            n_entry=1.0, n_exit=1.0,
                            method="exponential").solve()
    th = np.radians(ang)
    k0 = 2 * np.pi / wl
    Kx = 1.0 * np.sin(th)
    Kz = 1.0 * np.cos(th)
    entry = live_pl.HalfSpace(np.diag([1.0] * 3), Kx, Kz, k0)
    exit_ = live_pl.HalfSpace(np.diag([1.0] * 3), Kx, Kz, k0)
    st = live_pl.Structure(entry, exit_, Kx, 0.0, Kz, Kz, k0)
    st.layers = [live_pl.Layer(eps_l, d, Kx, k0)]
    Jr, Jt = st.get_fresnel(method="EM")
    # single-point solves squeeze to (2,2) — compare whole Jones matrices
    worst_c = max(worst_c, float(np.max(np.abs(np.asarray(ours["J_refl"]) - Jr))),
                  float(np.max(np.abs(np.asarray(ours["J_trans"]) - Jt))))
print(f"G9c EM vs live-EM worst: {worst_c:.3e}")
assert worst_c < 1e-9, "G9c failed"
print("G9c PASS (G9d lives in tests/test_aniso_exit.py, reusing the G7b harness)")
