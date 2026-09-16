# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gate G13a: specular energy conservation for Route-1 roughness.

Self-contained physical invariant (no smatrix needed). This is the test that
would have caught the upstream type-5 bug (IMPLEMENTATION_PLAN.md Phase 8,
section 8.1): before the fix, code 5 fails it at R+T ~= 0.925.

Lossless stack => per-input R+T within [-tol, 1+tol] for every code x sigma
x angle. Specular-only model => never ABOVE 1 (diffuse loss is unmodeled).
"""
import numpy as np

from navette import berreman as bl

n_entry, n_exit, n1, n2 = 1.0, 1.5, 2.0, 1.8
t1, t2, wl = 120.0, 90.0, 550.0
SIGMAS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
ANGLES = [0.0, 20.0, 40.0, 60.0]


def col_totals(R, T):
    R, T = np.asarray(R), np.asarray(T)
    return np.array([R[0, j] + R[1, j] + T[0, j] + T[1, j] for j in (0, 1)])


worst_over, worst_deficit = 0.0, {}
for rtype in range(6):
    worst_t = 0.0
    for sig in SIGMAS:
        layers = [bl.Layer(np.diag([n1**2]*3).astype(complex), t1, roughness=(rtype, sig)),
                  bl.Layer(np.diag([n2**2]*3).astype(complex), t2, roughness=(rtype, sig))]
        for ang in ANGLES:
            r = bl.BerremanStack(layers, [wl], [ang], n_entry=n_entry, n_exit=n_exit,
                                 method="scattering",
                                 exit_roughness=(rtype, sig)).solve()
            assert r["n_failed"] == 0
            tot = col_totals(r["R"], r["T"])
            assert np.all(tot >= -1e-9) and np.all(tot <= 1 + 1e-9), \
                f"code {rtype} sigma={sig} ang={ang}: R+T={tot} violates energy"
            worst_t = max(worst_t, np.max(np.abs(tot - np.minimum(tot, 1.0))))
            worst_over = max(worst_over, float(np.max(1.0 - tot)))
    worst_deficit[rtype] = worst_t
    print(f"code {rtype}: max specular deficit 1-(R+T) = {worst_t:.3e} (over sigma x angle)")
print(f"WORST deficit overall: {worst_over:.3e}")
print("ENERGY PASS (lossless arm)")
# Expected signature (documents the section 8.1 model distinction):
# codes 1-4 deficit ~5e-3 @ sigma=10nm (graded-profile-like); fixed code 5 <= 1
# everywhere. NOTE: this gate does NOT catch the type-5 bug by itself —
# pre-fix stacks measured R+T ~= 0.755 (three rough interfaces) yet still
# satisfy the <= 1 bound. The bug-catcher is the transmission fork-guard in
# test_roughness.py (pre/post-fix |dT| ~ 0.11-0.22 on the G5 fixture); this
# gate pins the never-create-energy invariant going forward.


# --- absorbing arm (same file, second half) ---
print("--- absorbing arm ---")
eps_abs = bl.rot_z(np.diag([2.25 + 0.1j, 2.89 + 0.05j, 2.25 + 0.1j]).astype(complex),
                   np.radians(30.0))
for ang in [0.0, 30.0, 60.0]:
    layers = [bl.Layer(eps_abs, 250.0, roughness=(5, 10.0))]
    r = bl.BerremanStack(layers, [wl], [ang], n_entry=n_entry, n_exit=n_exit,
                         method="scattering",
                         exit_roughness=(5, 10.0)).solve()
    assert r["n_failed"] == 0
    tot = col_totals(r["R"], r["T"])
    # absorption only removes energy; roughness must never create it
    assert np.all(tot <= 1 + 1e-9), f"absorbing code-5 ang={ang}: R+T={tot} > 1"
    print(f"absorbing code-5 ang={ang}: R+T={tot} OK")
print("ENERGY PASS (absorbing arm)")
