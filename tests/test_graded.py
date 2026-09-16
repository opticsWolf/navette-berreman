# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gates G13b-d: Route-2 (graded_stack) coverage, no smatrix needed.

G13b sigma=0 identity, G13c scattering-vs-transfer agreement, G13d sublayer
convergence. G13e (same file, bottom): Route-1 vs Route-2 cross-check — the
only independent check available for anisotropic roughness. The two routes are
different approximations (perturbative W-factors vs erf-graded sublayers), so
expect convergence as sigma -> 0, NOT equality.
"""
import numpy as np

from navette import berreman as bl

WL, ANG = 550.0, 30.0
N_ENTRY, N_EXIT = 1.0, 1.5


def solve_graded(sigmas, n_sub, method):
    eps1 = np.diag([2.25, 2.89, 2.25]).astype(complex)  # anisotropic fixture
    eps2 = np.diag([2.56]*3).astype(complex)
    base = [bl.Layer(eps1, 120.0), bl.Layer(eps2, 90.0)]
    g = bl.graded_stack(base, sigmas, n_entry=N_ENTRY, n_exit=N_EXIT,
                        n_sublayers=n_sub)
    return bl.BerremanStack(g, [WL], [ANG], n_entry=N_ENTRY, n_exit=N_EXIT,
                            method=method).solve()


def solve_base(method):
    eps1 = np.diag([2.25, 2.89, 2.25]).astype(complex)
    eps2 = np.diag([2.56]*3).astype(complex)
    base = [bl.Layer(eps1, 120.0), bl.Layer(eps2, 90.0)]
    return bl.BerremanStack(base, [WL], [ANG], n_entry=N_ENTRY, n_exit=N_EXIT,
                            method=method).solve()


def rt_key(r):
    return np.concatenate([np.asarray(r["R"]).ravel(), np.asarray(r["T"]).ravel()])


# --- G13b: sigma=0 identity (exactly 0.0: zero widths -> identical layers) ---
for m in ("scattering", "transfer"):
    g = solve_graded([0, 0, 0], 9, m)
    b = solve_base(m)
    d = float(np.max(np.abs(rt_key(g) - rt_key(b))))
    print(f"G13b method={m}: |graded(sigma=0) - base| = {d:.2e}")
    assert d == 0.0, "sigma=0 graded stack differs from base: construction bug"
print("G13b PASS")

# --- G13c: method agreement on a graded anisotropic stack ---
gs = solve_graded([8, 5, 12], 9, "scattering")
gt = solve_graded([8, 5, 12], 9, "transfer")
dRT = float(np.max(np.abs(rt_key(gs) - rt_key(gt))))
dJ = max(float(np.max(np.abs(np.asarray(gs[k]) - np.asarray(gt[k]))))
         for k in ("J_refl", "J_trans"))
print(f"G13c: SM-vs-TM R/T = {dRT:.2e}, Jones = {dJ:.2e} (pin <= 1e-12)")
assert dRT <= 1e-12 and dJ <= 1e-12
print("G13c PASS")

# --- G13d: sublayer convergence (monotone + bound) ---
# Measured convergence on this harsh fixture (high contrast, sigma up to 12 nm
# vs 90 nm films) is slow: err halves only ~2.8x per 1.6x refinement
# (9v15 = 9.2e-5), so the bound uses a (67, 107, 171) triple. Solves are
# milliseconds; the counts pin convergence, not production practice (~9).
r67 = rt_key(solve_graded([8, 5, 12], 67, "scattering"))
r107 = rt_key(solve_graded([8, 5, 12], 107, "scattering"))
r171 = rt_key(solve_graded([8, 5, 12], 171, "scattering"))
e1 = float(np.max(np.abs(r67 - r107)))
e2 = float(np.max(np.abs(r107 - r171)))
print(f"G13d: err(67v107) = {e1:.2e}, err(107v171) = {e2:.2e}")
assert e2 < e1, "sublayer convergence not monotone"
assert e2 <= 1e-6, "107-vs-171 sublayer error above bound"
print("G13d PASS")


# --- G13e: Route-1 vs Route-2 cross-check (converging, not equal) ---
print("--- G13e ---")
eps_aniso = bl.rot_z(np.diag([2.25, 2.89, 2.25]).astype(complex), np.radians(30.0))
errs = {}
for sig in (2.0, 1.0, 0.5):
    r1 = bl.BerremanStack([bl.Layer(eps_aniso, 250.0, roughness=(4, sig))],
                          [WL], [ANG], method="scattering").solve()
    g = bl.graded_stack([bl.Layer(eps_aniso, 250.0)], [sig, 0.0],
                        n_sublayers=15)
    r2 = bl.BerremanStack(g, [WL], [ANG], method="scattering").solve()
    err = max(float(np.max(np.abs(np.asarray(r1["R"]) - np.asarray(r2["R"])))),
              float(np.max(np.abs(np.asarray(r1["T"]) - np.asarray(r2["T"])))))
    errs[sig] = err
    print(f"G13e sigma={sig}: |Route1 - Route2| = {err:.2e}")
assert errs[0.5] < errs[1.0] < errs[2.0], "Route-1/Route-2 not converging"
assert errs[2.0] <= 5e-2, "model spread above absolute bound"
print("G13e PASS")
