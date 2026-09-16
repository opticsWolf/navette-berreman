# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare ref/rust_out.json against ref/ref_out.json with a tolerance.

Jones matrices are compared up to a global sign/phase per matrix (physically
irrelevant), R/T compared directly. Reports max abs error per field and an
overall PASS/FAIL.
"""
import json
import sys
import numpy as np

ref = json.load(open("/home/claude/ref/ref_out.json"))
rust = json.load(open("/home/claude/ref/rust_out.json"))

ATOL = 1e-6
RTOL = 1e-5


def to_c(m):
    return np.array([[complex(v[0], v[1]) for v in row] for row in m])


def to_r(m):
    return np.array(m, dtype=float)


def cmp_complex_upto_phase(a, b):
    """Max abs diff allowing one global phase between the two 2x2 matrices."""
    a = a.flatten()
    b = b.flatten()
    # pick the largest-magnitude reference entry to fix the phase
    k = int(np.argmax(np.abs(b)))
    if np.abs(b[k]) < 1e-12 or np.abs(a[k]) < 1e-12:
        # near-zero matrix; just compare directly and via -1
        d1 = np.max(np.abs(a - b))
        d2 = np.max(np.abs(a + b))
        return min(d1, d2)
    phase = (b[k] / a[k])
    phase = phase / abs(phase)
    return np.max(np.abs(a * phase - b))


worst = 0.0
fails = []
for case in ref:
    for method in ("SM", "TM"):
        r = ref[case][method]
        u = rust[case][method]
        # R, T compared directly
        for fld in ("R", "T"):
            d = np.max(np.abs(to_r(r[fld]) - to_r(u[fld])))
            tol = ATOL + RTOL * np.max(np.abs(to_r(r[fld])))
            worst = max(worst, d)
            if d > tol:
                fails.append(f"{case}/{method}/{fld}: max|Δ|={d:.3e} (tol {tol:.1e})")
        # Jones (refl/trans/refl_c) up to global phase
        for fld in ("J_refl", "J_trans", "J_refl_c"):
            d = cmp_complex_upto_phase(to_c(u[fld]), to_c(r[fld]))
            tol = ATOL + RTOL * np.max(np.abs(to_c(r[fld])))
            worst = max(worst, d)
            if d > tol:
                fails.append(f"{case}/{method}/{fld}: max|Δ|={d:.3e} (tol {tol:.1e})")

print(f"worst abs error across all fields: {worst:.3e}")
if fails:
    print(f"\n{len(fails)} FIELD(S) FAILED:")
    for f in fails:
        print("  " + f)
    sys.exit(1)
else:
    print("\nALL CASES PASS (SM + TM): J_refl, J_trans, J_refl_c, R, T")

# physical sanity: R+T-ish bookkeeping per case (energy <= 1 per input pol)
print("\nphysical check (sum of R column + T column per input pol, lossless cases):")
for case in ref:
    if "absorbing" in case:
        continue
    R = to_r(ref[case]["SM"]["R"])
    T = to_r(ref[case]["SM"]["T"])
    tot = (R + T).sum(axis=0)  # per input polarization column
    print(f"  {case:24s} R+T per in-pol = [{tot[0]:.6f}, {tot[1]:.6f}]")
