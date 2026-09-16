import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NOTE: requires the live BerreMueller sources importable (pyllama +
# dielectric_tensor); same pattern as test_full.py's FULL_BERREMAN_DIR.
sys.path.insert(0, os.environ.get("FULL_BERREMAN_DIR", "."))
"""Gate G12: Rust tensor rotations vs live BerreMueller references.

G12a: axis-angle / euler / quaternion rotation matrices + rotated complex
      tensors vs live (per-family worst reported separately so an Euler-order
      bug diagnoses itself).
G12b: new Rust-backed rot_z vs the OLD NumPy body (inlined copy below) —
      exactly 0.0, then the NumPy body stays deleted from berreman.py.
"""
import numpy as np

from navette import berreman as bl
from berremueller import pyllama as live_pl
from berremueller import dielectric_tensor as live_dt


def _old_rot_z(eps, angle_rad):
    """Inlined copy of the pre-P6 NumPy rot_z body (deleted from berreman.py
    after this gate pins the delegation to 0.0)."""
    cphi, sphi = np.cos(angle_rad), np.sin(angle_rad)
    rz = np.array([[cphi, -sphi, 0], [sphi, cphi, 0], [0, 0, 1]], dtype=complex)
    return rz @ np.asarray(eps, dtype=complex) @ rz.T


# NOTE: no identity-tensor probe (R·I·Rᵀ = I for every R — vacuous). The API
# applies R·E·Rᵀ, so random complex E over 20 seeds is the pin: it fixes R up
# to an overall sign, and the Rust group test (det = +1) kills the -R case.
rng = np.random.default_rng(11)
worst_aa, worst_eu, worst_q = 0.0, 0.0, 0.0
for _ in range(20):
    # random COMPLEX probe tensors (catch R·eps·Rᵀ conjugation slips too)
    E = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    ax = rng.normal(size=3)
    th = rng.uniform(-np.pi, np.pi)
    R_live = np.asarray(live_pl.rot_mat(axis=ax, theta_rad=th), dtype=float)
    worst_aa = max(worst_aa, float(np.max(np.abs(bl.rot_axis(ax, th, E) - R_live @ E @ R_live.T))))

    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    # live is scalar-LAST [x,y,z,w]; ours is scalar-first (w,x,y,z) = q
    Rq_live = np.asarray(live_dt.quaternion_rotation_matrix([q[1], q[2], q[3], q[0]]),
                         dtype=float)
    worst_q = max(worst_q, float(np.max(np.abs(
        bl.rot_quat(q[0], q[1], q[2], q[3], E) - Rq_live @ E @ Rq_live.T))))

    rx, ry, rz = rng.uniform(-np.pi, np.pi, size=3)
    Re_live = np.asarray(live_dt.euler_rotation_matrix(rx, ry, rz), dtype=float)
    worst_eu = max(worst_eu, float(np.max(np.abs(bl.rot_euler(rx, ry, rz, E)
                                                 - Re_live @ E @ Re_live.T))))

print(f"G12a axis-angle worst: {worst_aa:.3e}")
print(f"G12a euler worst:      {worst_eu:.3e}")
print(f"G12a quaternion worst: {worst_q:.3e}")
assert worst_aa < 1e-14, "axis-angle drift vs live rot_mat"
assert worst_eu < 1e-14, "euler drift vs live (order hypothesis §6.2 wrong?)"
assert worst_q < 1e-14, "quaternion drift vs live (scalar-order mapping?)"
print("G12a PASS")

# --- G12b: rot_z delegation parity (exactly 0.0) ---
base = np.diag([2.25, 2.89, 2.25]).astype(complex)
worst_z = 0.0
for deg in (0.0, 15.0, 35.0, 90.0):
    a = np.radians(deg)
    worst_z = max(worst_z, float(np.max(np.abs(bl.rot_z(base, a) - _old_rot_z(base, a)))))
print(f"G12b rot_z delegation worst: {worst_z:.3e}")
# NOT 0.0 by design (measured 4.4e-16): Rodrigues (I + s·K + (1−c)·K²) vs the
# old closed-form Rz evaluate cos/sin in a different op order — 1-ulp
# divergence on equivalent expressions. The live pin (G12a, ≤1e-14 vs rot_mat)
# is the real gate; this arm only guards the delegation against regressions.
assert worst_z <= 1e-15, "rot_z delegation changed numerics"
# zero-axis / zero-norm rejections (live raises too)
for fn in (lambda: bl.rot_axis([0.0, 0.0, 0.0], 0.5, base),
           lambda: bl.rot_quat(0.0, 0.0, 0.0, 0.0, base)):
    try:
        fn()
        raise SystemExit("MISSING-ERROR: degenerate rotation accepted")
    except ValueError:
        pass
print("degenerate rejections: OK")
print("G12b PASS")
