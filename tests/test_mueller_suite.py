# SPDX-License-Identifier: LGPL-3.0-or-later
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NOTE: G11a needs the live BerreMueller sources (mueller.py + berreman_mueller.py).
sys.path.insert(0, os.environ.get("FULL_BERREMAN_DIR", "."))
"""Gate G11: Mueller-matrix suite vs live (Phase 5).

Spike record (§5.2, transcribed from source, not memory):
- live has NO depolarization_index/di atten uation-vector scalar entry points.
  Pins used: get_purity_components_mueller_matrix_stack (needs (4,4,X) stack;
  D = |row0|/M00, P = |col0|/M00, mueller.py:1197-1210), norm_mueller_matrix_stack
  (mueller.py:1154, per-component D/P), create_pauli_stack (mueller.py:159-178,
  our cloude reference basis), get_m03_raw (berreman_mueller.py:776-782,
  == 2*M03 exactly, same sign — spike analytic+numeric check).
- DI reference: Gil-Bernabeu textbook formula in numpy + eigenvalue identity
  DI^2 = (4*sum(lam^2)/(sum lam)^2 - 1)/3 (also asserted in Rust).
- mpl get_cmap shim + sympy/pandas required to import live mueller.py.
"""
import matplotlib.cm as _cm
if not hasattr(_cm, "get_cmap"):
    _cm.get_cmap = lambda name=None: __import__("matplotlib").colormaps[name]
import numpy as np

from navette import berreman as bl
from berremueller import berreman_mueller as bm
from berremueller import mueller as live_mu

rng = np.random.default_rng(7)
worst = 0.0

# 20 random Mueller-Jones matrices: D/P vectors + CD + cloude vs live
for _ in range(20):
    J = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    M = np.real(np.asarray(bm.mueller_from_jones_matrix(J)))
    stack = M.reshape(4, 4, 1)
    pd, pp, _ = live_mu.get_purity_components_mueller_matrix_stack(stack)
    nM = np.asarray(live_mu.norm_mueller_matrix_stack(stack))[:, :, 0]
    d, p = bl.diattenuation(M), bl.polarizance(M)
    worst = max(worst,
                abs(np.linalg.norm(d) - float(pd[0])),
                abs(np.linalg.norm(p) - float(pp[0])),
                float(np.max(np.abs(d - nM[0, 1:]))),
                float(np.max(np.abs(p - nM[1:, 0]))))
    # DI vs textbook formula (not in live)
    di_ref = float(np.sqrt(max(np.sum(M * M) - M[0, 0] ** 2, 0.0))
                   / (np.sqrt(3.0) * M[0, 0]))
    worst = max(worst, abs(float(bl.depolarization_index(M)) - di_ref),
                abs(float(bl.depolarization_index(M)) - 1.0))  # Jones => DI == 1
    # CD vs live raw (factor-2 quirk on the TEST side, never absorbed)
    raw = float(np.real(bm.get_m03_raw(J.reshape(2, 2, 1)))[0])
    worst = max(worst, abs(float(bl.circular_dichroism(M)) - raw / (2 * M[0, 0])))

# 5 depolarizing fixtures: convex sums; cloude eigenvalues vs live's OWN basis
sig = live_mu.create_pauli_stack()  # default "optics", unnormalized
for _ in range(5):
    Js = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(3)]
    w = rng.dirichlet([1, 1, 1])
    Ms = [np.real(np.asarray(bm.mueller_from_jones_matrix(J))) for J in Js]
    M = sum(wi * m for wi, m in zip(w, Ms))
    H = sum(M[i, j] / 4 * np.kron(sig[:, :, i], np.conj(sig[:, :, j]))
            for i in range(4) for j in range(4))
    lam_live = np.sort(np.linalg.eigvalsh(H))[::-1]  # == eig(live C), L unitary
    c = bl.cloude(M)
    worst = max(worst, float(np.max(np.abs(c["lambda"] - lam_live))))
    # entropy vs direct base-4 computation from live eigenvalues
    pv = np.clip(lam_live / lam_live.sum(), 0, None)
    ent_live = float(-sum(p * np.log(p) / np.log(4.0) for p in pv if p > 0))
    worst = max(worst, abs(float(c["entropy"]) - ent_live))
    # DI eigenvalue cross-identity on a genuinely depolarizing M
    di = float(bl.depolarization_index(M))
    di2 = (4 * np.sum(lam_live ** 2) / lam_live.sum() ** 2 - 1) / 3
    worst = max(worst, abs(di * di - di2))
    assert 0.0 < di < 1.0, di
print(f"G11a mueller suite vs live worst: {worst:.3e}")
assert worst < 1e-12, "G11a failed"
print("G11a PASS")

# G11c end-to-end: coherent slab M_trans stays Mueller-Jones (DI = 1)
eps = np.diag([2.25] * 3).astype(complex)
res = bl.BerremanStack([bl.Layer(eps, 137.0)], [550.0], [20.0],
                       n_entry=1.0, n_exit=1.0).solve()
assert res["n_failed"] == 0
di_grid = bl.depolarization_index(res["M_trans"])
dev = float(np.max(np.abs(di_grid - 1.0)))
cl = bl.cloude(res["M_trans"])
lam_tail = np.max(np.abs(cl["lambda"][..., 1:]))
print(f"G11c slab DI deviation: {dev:.3e}, tail lambda: {lam_tail:.3e}")
assert dev < 1e-9 and lam_tail < 1e-9, "G11c failed"
print("G11c PASS")

# guard mapping: zero matrix -> NaN everywhere (not crash)
z = np.zeros((4, 4))
assert np.isnan(bl.depolarization_index(z))
assert np.all(np.isnan(bl.diattenuation(z)))
assert np.all(np.isnan(bl.cloude(z)["lambda"]))
print("G11 guards PASS")
