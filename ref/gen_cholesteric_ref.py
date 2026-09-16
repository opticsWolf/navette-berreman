# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate ref/ref_out_cholesteric.json: live-B44 circular Bragg reference (G15d).

Fixture (mirrors tests/test_cholesteric.py G15d):
  uniaxial no=1.5, ne=1.7 (director along x at z=0), pitch 350 nm, N=5
  pitches (d=1750 nm), twist angle=+-2*pi*N, 48 slices/pitch (div=240),
  air entry/exit, normal incidence, B44 midpoint + Pade-2 slice propagators.
B44 side uses SI units (meters); tensor sampling verified midpoint-identical
to our twisted_stack(grid="midpoint") at 5.6e-17 (see session notes).
Circular channels via the frozen F/B sandwich (layouts match per G7b).

Usage: B44_DIR=... python ref/gen_cholesteric_ref.py
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("B44_DIR", "."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from Berreman4x4 import Berreman4x4 as B44

NO, NE, PITCH, NPER = 1.5, 1.7, 350.0, 5
D_NM = PITCH * NPER
DIV = NPER * 48
WL = np.arange(480.0, 641.0, 1.0)
F = np.array([[1, 1], [-1j, 1j]])
B = np.array([[1, 1], [1j, -1j]])
BINV = np.linalg.inv(B)


def b44_circ(angle):
    base = B44.NonDispersiveMaterial(
        epsilon=np.matrix(np.diag([NE ** 2, NO ** 2, NO ** 2])))
    tw = B44.TwistedMaterial(base, D_NM * 1e-9, angle, DIV)
    air = B44.IsotropicHalfSpace(B44.IsotropicNonDispersiveMaterial(n=1.0))
    st = B44.Structure(
        front=air,
        layers=[B44.InhomogeneousLayer(tw, "midpoint", "Padé", 2)],
        back=air)
    r_rr, r_ll, jcc = [], [], []
    for w in WL:
        k0 = 2 * np.pi / (w * 1e-9)
        tri, _ = (np.asarray(m, dtype=complex) for m in st.getJones(0.0, k0))
        jc = BINV @ tri @ F
        jcc.append([jc[0, 0].real, jc[0, 0].imag,
                    jc[1, 1].real, jc[1, 1].imag])
        r_rr.append(float(abs(jc[0, 0]) ** 2))
        r_ll.append(float(abs(jc[1, 1]) ** 2))
    return r_rr, r_ll, jcc


out = {"meta": {"no": NO, "ne": NE, "pitch_nm": PITCH, "n_periods": NPER,
                "div": DIV, "slices_per_pitch": 48, "d_nm": D_NM,
                "wl_nm": WL.tolist(), "angle_pos": 2 * np.pi * NPER,
                "method": "B44 InhomogeneousLayer midpoint Pade-2"},
       "helicity": {}}
for tag, angle in (("plus", 2 * np.pi * NPER), ("minus", -2 * np.pi * NPER)):
    r_rr, r_ll, jcc = b44_circ(angle)
    i = int(np.argmax(r_rr))
    print(f"angle {tag}: peak R_RR={r_rr[i]:.6f} at {WL[i]:.1f}nm; "
          f"peak R_LL={max(r_ll):.6f}", flush=True)
    out["helicity"][tag] = {"R_RR": r_rr, "R_LL": r_ll, "Jcc_diag": jcc}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "ref_out_cholesteric.json")
with open(path, "w") as fh:
    json.dump(out, fh)
print("wrote", path)
