# SPDX-License-Identifier: LGPL-3.0-or-later
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ref_pyllama.py
"""End-to-end check of the compiled extension + wrapper against ref_pyllama."""
import sys
import numpy as np


from navette import berreman as bl
import ref_pyllama as ref

# A dispersion-free rotated-uniaxial slab on a glass substrate, swept over a
# grid of wavelengths and angles, both methods.
eps_layer = bl.rot_z(np.diag([2.25, 2.89, 2.25]).astype(complex), np.radians(35.0))
wls = np.array([450.0, 550.0, 633.0, 700.0])
angles = np.array([0.0, 15.0, 30.0, 45.0])  # degrees
n_entry, n_exit = 1.0, 1.5

worst = 0.0
for method in ("scattering", "transfer"):
    stack = bl.BerremanStack(
        [bl.Layer(eps_layer, 250.0)],
        wavelengths_nm=wls, angles=angles,
        n_entry=n_entry, n_exit=n_exit, method=method,
    )
    res = stack.solve()
    assert res["n_failed"] == 0, f"{method}: {res['n_failed']} failed points"

    m = "TM" if method == "transfer" else "SM"
    for iw, wl in enumerate(wls):
        for ia, ang in enumerate(angles):
            r = ref.solve(
                [(eps_layer, 250.0)], n_entry ** 2, n_exit ** 2,
                wl, np.radians(ang), method=m,
            )
            # R / T compared directly
            for fld_py, fld_ref in (("R", "R"), ("T", "T")):
                a = res[fld_py][iw, ia]
                b = r[fld_ref].real
                worst = max(worst, np.max(np.abs(a - b)))
            # Jones up to a global phase
            for fld_py, fld_ref in (("J_refl", "J_refl"), ("J_trans", "J_trans"),
                                    ("J_refl_circ", "J_refl_c")):
                a = res[fld_py][iw, ia].flatten()
                b = r[fld_ref].flatten()
                k = int(np.argmax(np.abs(b)))
                ph = (b[k] / a[k]); ph /= abs(ph)
                worst = max(worst, np.max(np.abs(a * ph - b)))

print(f"grid sweep (4 wl x 4 angles x 2 methods) worst abs error: {worst:.3e}")

# Mueller round-trip sanity: M from an identity Jones is the 4x4 identity.
M = bl.mueller_from_jones(np.eye(2))
print("Mueller(I) == I:", np.allclose(M, np.eye(4), atol=1e-12))

# Energy conservation on a lossless point.
stack = bl.BerremanStack([bl.Layer(eps_layer, 250.0)], [550.0], [30.0],
                         n_entry=1.0, n_exit=1.0, method="scattering")
res = stack.solve()
tot = (res["R"] + res["T"]).sum(axis=0)
print(f"lossless R+T per input pol: [{tot[0]:.8f}, {tot[1]:.8f}]")

assert worst < 1e-6, "grid sweep exceeded tolerance"
print("\nEND-TO-END PASS")
