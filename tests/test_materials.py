# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gates G14a-c: materials (needs upstream `navette.materials` for G14a).

G14a: live parity vs upstream evaluate on every MODELS entry x 200-pt grid.
G14b: tensor identities (no upstream needed).
G14c: end-to-end (materials -> Layer -> BerremanStack).
"""
import numpy as np

from navette import berreman as bl
from navette import berreman_materials as M
import navette.materials as U

WL = np.linspace(300.0, 1200.0, 200)

CASES = {
    "Konstant": {"n": 1.5, "k": 0.01},
    "Table": {"n_data": ([300.0, 600.0, 1200.0], [1.6, 1.5, 1.48]),
              "k_data": ([300.0, 600.0, 1200.0], [0.02, 0.01, 0.0])},
    "Cauchy": {"A": 1.5, "B": 0.01, "C": 0.0001},
    "CauchyUrbach": {"A": 1.5, "B": 0.01, "C": 0.0001,
                     "alpha0": 100.0, "Eu": 0.1, "lambda_g": 350.0},
    "Sellmeier": {"B1": 1.03961212, "C1": 0.00600069867, "B2": 0.231792344,
                  "C2": 0.0200179144, "B3": 1.01046945, "C3": 103.560653},
    "SellmeierUrbach": {"B1": 1.03961212, "C1": 0.00600069867, "B2": 0.231792344,
                        "C2": 0.0200179144, "B3": 1.01046945, "C3": 103.560653,
                        "alpha0": 100.0, "Eu": 0.1, "lambda_g": 350.0},
    "Lorentz": {"osc": [[5.0, 0.5, 2.0], [7.0, 0.3, 1.0]], "epsilon_inf": 2.0},
    "Drude": {"omega_p": 2.5, "gamma": 0.3, "epsilon_inf": 3.5},
    "DrudeLorentz": {"omega_p": 2.5, "gamma_drude": 0.3, "epsilon_inf": 2.0,
                     "osc": [[5.0, 0.5, 1.0]]},
    "CodyLorentz": {"Eg": 1.5, "Et": 0.5, "Eu": 0.3,
                    "osc": [[6.0, 20.0, 1.0, 2.0]], "epsilon_inf": 1.0},
    "ForouhiBloomerSingle": {"n_inf": 2.0, "ib": [[3.0, 5.0, 1.0, 2.0]]},
    "ForouhiBloomerMulti": {"n_inf": 2.0,
                            "ib": [[3.0, 5.0, 1.0, 2.0], [5.0, 3.0, 0.5, 1.0]]},
    "ForouhiBloomerMetal": {"n_inf": 1.5, "A_fe": 2.0, "B_fe": 1.0, "C_fe": 0.5,
                            "ib": [[3.0, 5.0, 1.0, 2.0]]},
    "ForouhiBloomerMetal2021": {"n_inf": 1.5, "fe": [2.0, 1.0, 0.5],
                                "ib": [[3.0, 5.0, 1.0, 2.0]]},
    "TaucLorentz": {"Eg": 2.0, "osc": [[30.0, 4.0, 1.5]], "epsilon_inf": 1.0},
    "UBF": {"osc": [{"Eg": 2.0, "Ec": 5.0, "Eu": 0.3, "A": 50.0,
                     "Gamma": 1.0, "gamma": 1.0}], "epsilon_inf": 1.0},
    "Bruggeman": {"host": {"model": "Konstant", "params": {"n": 1.5}},
                  "inclusion": {"model": "Konstant", "params": {"n": 2.0, "k": 0.1}},
                  "fraction": 0.4},
    "MaxwellGarnett": {"host": {"model": "Konstant", "params": {"n": 1.5}},
                       "inclusion": {"model": "Lorentz",
                                     "params": {"osc": [[5.0, 0.5, 2.0]]}},
                       "fraction": 0.2},
    "Looyenga": {"host": {"model": "Konstant", "params": {"n": 1.5}},
                 "inclusion": {"model": "Konstant", "params": {"n": 2.2}},
                 "fraction": 0.5},
    "Lichtenecker": {"host": {"model": "Cauchy", "params": {"A": 1.5, "B": 0.01, "C": 0.0}},
                     "inclusion": {"model": "Konstant", "params": {"n": 2.0}},
                     "fraction": 0.3},
    "MoriTanaka": {"host": {"model": "Konstant", "params": {"n": 1.5}},
                   "inclusion": {"model": "Konstant", "params": {"n": 2.0}},
                   "fraction": 0.25, "L": 1.0 / 3.0},
    "PowerLaw": {"host": {"model": "Konstant", "params": {"n": 1.5}},
                 "inclusion": {"model": "Konstant", "params": {"n": 2.0}},
                 "fraction": 0.35, "alpha": 0.5},
    "Roughness": {"bottom": {"model": "Konstant", "params": {"n": 1.5}},
                  "top": {"model": "Konstant", "params": {"n": 2.0}}},
}

print("=== G14a: live parity vs upstream navette.materials ===")
worst_all = 0.0
for name, params in CASES.items():
    got = M.evaluate({"model": name, "params": params}, WL)
    ref = np.asarray(
        U.evaluate(U.MaterialSpec(model=name, params=dict(params)), WL), dtype=complex
    )
    d = float(np.max(np.abs(got - ref)))
    worst_all = max(worst_all, d)
    print(f"{name:<24} max|d| = {d:.2e}")
    assert d <= 1e-15, f"{name} parity {d:.2e} above 1e-15"
print(f"G14a WORST: {worst_all:.2e}")
print("G14a PASS")

print("=== G14b: tensor identities (no upstream) ===")
# uniaxial o==e equals isotropic exactly
iso = M.evaluate_tensor({"model": "Konstant", "params": {"n": 1.7, "k": 0.02}}, WL)
uni = M.evaluate_tensor({"uniaxial": {
    "ordinary": {"model": "Konstant", "params": {"n": 1.7, "k": 0.02}},
    "extraordinary": {"model": "Konstant", "params": {"n": 1.7, "k": 0.02}}}}, WL)
d = float(np.max(np.abs(iso - uni)))
print(f"uniaxial(o==e) vs isotropic: {d:.2e}")
assert d == 0.0
# EMA f=0/1 endpoints
for f, want in ((0.0, 1.5), (1.0, 2.0)):
    nk = M.evaluate({"model": "Bruggeman",
                     "params": {"host": {"model": "Konstant", "params": {"n": 1.5}},
                                "inclusion": {"model": "Konstant", "params": {"n": 2.0}},
                                "fraction": f}}, WL)
    d = float(np.max(np.abs(nk - want)))
    print(f"Bruggeman f={f}: endpoint err {d:.2e}")
    assert d == 0.0
# PowerLaw alpha->0 == Lichtenecker
pl = M.evaluate({"model": "PowerLaw",
                 "params": {"host": {"model": "Konstant", "params": {"n": 1.5}},
                            "inclusion": {"model": "Konstant", "params": {"n": 2.0}},
                            "fraction": 0.4, "alpha": 1e-9}}, WL)
li = M.evaluate({"model": "Lichtenecker",
                 "params": {"host": {"model": "Konstant", "params": {"n": 1.5}},
                            "inclusion": {"model": "Konstant", "params": {"n": 2.0}},
                            "fraction": 0.4}}, WL)
d = float(np.max(np.abs(pl - li)))
print(f"PowerLaw(1e-9) vs Lichtenecker: {d:.2e}")
assert d == 0.0
# Table exact at nodes + clamped outside
tn = M.evaluate({"model": "Table",
                 "params": {"n_data": ([400.0, 600.0], [1.6, 1.5])}}, WL)
i400 = int(np.argmin(np.abs(WL - 400.0)))
print(f"Table off-node sample: n({WL[i400]:.1f})={tn[i400].real:.6f}")
out_lo = M.evaluate({"model": "Table",
                     "params": {"n_data": ([400.0, 600.0], [1.6, 1.5])}},
                    np.array([200.0, 1500.0]))
assert out_lo[0].real == 1.6 and out_lo[1].real == 1.5, "table clamp broken"
print("Table clamp outside grid: OK (edge values)")
# Urbach k=0 above gap
cu = M.evaluate({"model": "CauchyUrbach",
                 "params": {"A": 1.5, "B": 0.0, "C": 0.0,
                            "alpha0": 100.0, "Eu": 0.1, "lambda_g": 350.0}},
                np.array([300.0]))  # E=4.1eV > Eg=3.54eV
print(f"Urbach above gap: k={cu[0].imag:.2e}")
assert cu[0].imag == 0.0
print("G14b PASS")

print("=== G14c: end-to-end ===")
# BK7-Sellmeier slab via materials -> Layer == direct-eps stack exactly
bk7 = {"model": "Sellmeier",
       "params": {"B1": 1.03961212, "C1": 0.00600069867, "B2": 0.231792344,
                  "C2": 0.0200179144, "B3": 1.01046945, "C3": 103.560653}}
wl1 = np.array([550.0])
eps_mat = M.evaluate_tensor(bk7, wl1)  # (1,3,3)
n_bk7 = M.evaluate(bk7, wl1)[0]
eps_direct = np.eye(3, dtype=complex) * n_bk7**2
lm = bl.BerremanStack([bl.Layer(eps_mat[0], 500.0)], [550.0], [30.0]).solve()
ld = bl.BerremanStack([bl.Layer(eps_direct, 500.0)], [550.0], [30.0]).solve()
d = max(float(np.max(np.abs(np.asarray(lm[k]) - np.asarray(ld[k]))))
        for k in ("R", "T", "J_refl", "J_trans"))
print(f"materials-slab vs direct-eps slab: {d:.2e}, n_failed={lm['n_failed']}")
assert d == 0.0 and lm["n_failed"] == 0
tot = np.asarray(lm["R"]) + np.asarray(lm["T"])
assert np.all(tot <= 1 + 1e-9), f"energy {tot}"
# uniaxial + FB-metal coverage
eps_uni = M.evaluate_tensor(
    {"uniaxial": {"ordinary": {"model": "Konstant", "params": {"n": 1.5}},
                  "extraordinary": CASES["ForouhiBloomerMetal"] and
                  {"model": "Konstant", "params": {"n": 1.7}}}}, wl1)
r = bl.BerremanStack([bl.Layer(eps_uni[0], 300.0)], [550.0], [20.0]).solve()
assert r["n_failed"] == 0
print("uniaxial e2e: OK")
print("G14c PASS")
print("ALL MATERIALS GATES PASS")
