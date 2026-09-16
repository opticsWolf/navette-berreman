import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref"))
# NOTE: G10c needs the live BerreMueller sources (StackModel N_per arm).
sys.path.insert(0, os.environ.get("FULL_BERREMAN_DIR", "."))
"""Gate G10: periodic/Bragg fast path (periods).

G10a: pow algebra lives in Rust (transfer.rs tests) — not repeated here.
G10b: periods=N == layers*N expansion (smooth + rough, SM+TM+EM) <= 1e-12.
      Fixtures: Bragg 10x bilayer cell x10 (stress-test stack) + twisted
      20-slice cell x3.
G10c: vs live StackModel(..., N_per=10) SM+TM <= 1e-9.
G10d: 40-period DBR sweep: periodic must beat expansion by >= 2x (prints both).
"""
import numpy as np

from navette import berreman as bl
import ref_pyllama as ref
from berremueller import pyllama as live

# ── fixtures ─────────────────────────────────────────────────────────────────
EPS_H = np.diag([4.0] * 3).astype(complex)
EPS_L = np.diag([2.25] * 3).astype(complex)
BRAGG_CELL = [bl.Layer(EPS_H, 70.0), bl.Layer(EPS_L, 90.0)]

BASE_TW = np.diag([2.89, 2.25, 2.25]).astype(complex)
TW_CELL = [bl.Layer(bl.rot_z(BASE_TW, a), 50.0)
           for a in np.linspace(0, np.pi / 2, 20)]


def upto_phase(a, b):
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    k = int(np.argmax(np.abs(b)))
    ph = (b[k] / a[k])
    ph /= abs(ph)
    return np.max(np.abs(a * ph - b))


# ── G10b: periodic == expansion ──────────────────────────────────────────────
worst = 0.0
cases = [
    ("bragg10", BRAGG_CELL, 10, [600.0], [10.0], 1.0, 1.0, False),
    ("bragg10-oblique", BRAGG_CELL, 10, [500.0, 700.0], [0.0, 30.0], 1.0, 1.5, False),
    ("twist20x3", TW_CELL, 3, [550.0], [25.0], 1.0, 1.0, False),
    ("bragg10-rough", BRAGG_CELL, 10, [600.0], [10.0], 1.0, 1.0, True),
]
for name, cell, per, wls, angs, n_in, n_out, rough in cases:
    kw = {}
    if rough:
        cell = [bl.Layer(l.eps, l.thickness_nm, roughness=(4, 2.0)) for l in cell]
        kw = {"exit_roughness": (4, 1.0)}
    exp_layers = cell * per
    for method in ("scattering", "transfer", "exponential"):
        if rough and method != "scattering":
            continue  # Route-1 is SM-only (existing gate)
        a = bl.BerremanStack(cell, wls, angs, n_entry=n_in, n_exit=n_out,
                             method=method, periods=per, **kw).solve()
        b = bl.BerremanStack(exp_layers, wls, angs, n_entry=n_in, n_exit=n_out,
                             method=method, **kw).solve()
        assert a["n_failed"] == 0 and b["n_failed"] == 0, (name, method)
        for f in ("R", "T"):
            worst = max(worst, float(np.max(np.abs(a[f] - b[f]))))
        worst = max(worst, upto_phase(a["J_refl"], b["J_refl"]),
                    upto_phase(a["J_trans"], b["J_trans"]))
        print(f"G10b {name} {method}: done")
print(f"G10b periodic==expansion worst: {worst:.3e}")
assert worst < 1e-12, "G10b failed"
# periods validation
for bad in (0, -2):
    try:
        bl.BerremanStack(BRAGG_CELL, [600.0], [10.0], periods=bad)
        raise SystemExit(f"MISSING-ERROR: periods={bad} accepted")
    except ValueError:
        pass
print("G10b PASS")

# ── G10c: vs live N_per ──────────────────────────────────────────────────────
worst_c = 0.0
for method, lm in (("scattering", "SM"), ("transfer", "TM")):
    ours = bl.BerremanStack(BRAGG_CELL, [600.0], [10.0], n_entry=1.0,
                            n_exit=1.0, method=method, periods=10).solve()
    m = live.StackModel(600.0, eps_list=np.array([EPS_H, EPS_L]),
                        thickness_nm_list=np.array([70.0, 90.0]),
                        n_entry=1.0, n_exit=1.0,
                        theta_in_rad=np.radians(10.0), N_per=10)
    Jr, Jt = m.structure.get_fresnel(method=lm)
    R, T = m.get_refl_trans(method=lm)
    worst_c = max(worst_c, float(np.max(np.abs(ours["R"] - R))),
                  float(np.max(np.abs(ours["T"] - T))),
                  upto_phase(ours["J_refl"], Jr), upto_phase(ours["J_trans"], Jt))
print(f"G10c vs live N_per=10 worst: {worst_c:.3e}")
assert worst_c < 1e-9, "G10c failed"
print("G10c PASS")

# ── G10d: perf smoke ─────────────────────────────────────────────────────────
wls = np.linspace(400.0, 800.0, 40)
angs = np.linspace(0.0, 60.0, 9)
t0 = time.perf_counter()
r1 = bl.BerremanStack(BRAGG_CELL * 40, wls, angs, method="scattering").solve()
t_exp = time.perf_counter() - t0
t0 = time.perf_counter()
r2 = bl.BerremanStack(BRAGG_CELL, wls, angs, method="scattering", periods=40).solve()
t_per = time.perf_counter() - t0
print(f"G10d 40-period DBR 40wl x 9ang: expanded {t_exp:.2f}s, periodic {t_per:.2f}s "
      f"(x{t_exp / max(t_per, 1e-9):.1f}), agree {np.max(np.abs(r1['R'] - r2['R'])):.2e}")
assert t_per < t_exp / 2, "G10d speedup gate failed"
print("G10d PASS")
