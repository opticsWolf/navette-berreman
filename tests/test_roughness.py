# SPDX-License-Identifier: LGPL-3.0-or-later
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ref_pyllama.py
# NOTE: requires the smatrix extension (_smatrix.so + smatrix.py) importable.
sys.path.insert(0, os.environ.get("SMATRIX_DIR", "."))
"""Validate Route-1 roughness against the smatrix engine in the isotropic limit.

Identical isotropic stack (air / film1 / film2 / glass) is solved by both the
4x4 loom_berreman solver (with isotropic layers + Route-1 roughness dressing)
and the smatrix scattering engine.

Codes 0-4 must agree on R and T (s and p) at every angle (G5 intact).

Code 5 (Nevot-Croce) is SPLIT by design (IMPLEMENTATION_PLAN.md Phase 8):
reflection still matches smatrix (proves the fix touched transmission only),
but transmission INTENTIONALLY DIVERGES — upstream smatrix applies the
reflection factor f = exp(-2*kz1*kz2*s^2) to transmission as well, an energy
bug (R+T = 0.925 at a single interface, sigma = 10 nm; upstream
docs/code_review.md section 3.2). We use the correct Gaussian transfer factor
ga = exp(-(dkz)^2*s^2/2). The transmission arm therefore asserts DIVERGENCE
(|dT| > 1e-6 at sigma >= 5 nm): a fork-guard that fails loudly if anyone
"restores" bug-compatibility.

Interface mapping (loom interface k  <->  smatrix index k+1):
  entry|l0 = smatrix[1], l0|l1 = smatrix[2], l1|exit = smatrix[3]; smatrix[0]
  (ambient) is unused.
"""
import sys
import numpy as np

from navette import berreman as bl
import navette as _nv_top
# Guard: the local `navette` directory must win the top-level name (it is a
# namespace portion merged with the upstream wheel — see note below). If a
# reinstall of the upstream wheel restores its __init__.py, the wheel shadows
# us and this fails loudly instead of testing the wrong code.
assert any("navette_berreman" in p for p in _nv_top.__path__), \
    f"local navette shadowed by {_nv_top.__path__}: re-apply the G5 note"
try:
    import smatrix as sm  # legacy standalone layout (SMATRIX_DIR checkout)
except ImportError:  # PyPI navette>=0.7: engine lives at navette.smatrix.
    # NOTE (namespace merge): the upstream wheel (site-packages/navette, WITH
    # __init__.py) would shadow this repo's namespace-only navette/ dir, so
    # its __init__.py is renamed to __init__.py.disabled_by_navette_berreman_*
    # (reversible; re-apply after any wheel reinstall/upgrade). Both portions
    # then merge: navette.berreman (local) + navette.smatrix/.materials (wheel).
    from navette.smatrix import smatrix as sm

# stack
n_entry, n_exit = 1.0, 1.5
n1, n2 = 2.0, 1.8
t1, t2 = 120.0, 90.0
wl = 550.0
sig = [8.0, 5.0, 12.0]   # sigma_nm for [entry|l0, l0|l1, l1|exit], all >= 5 nm
angles = [0.0, 20.0, 40.0, 60.0]

ROUGH = {0: "none", 1: "uniform", 2: "two-delta", 3: "Lorentzian",
         4: "Gaussian", 5: "Nevot-Croce"}


def loom_RT(rtype, ang):
    eps1 = np.diag([n1 ** 2] * 3).astype(complex)
    eps2 = np.diag([n2 ** 2] * 3).astype(complex)
    layers = [
        bl.Layer(eps1, t1, roughness=(rtype, sig[0])),   # front = entry|l0
        bl.Layer(eps2, t2, roughness=(rtype, sig[1])),   # front = l0|l1
    ]
    stack = bl.BerremanStack(layers, [wl], [ang], n_entry=n_entry, n_exit=n_exit,
                             method="scattering", exit_roughness=(rtype, sig[2]))
    r = stack.solve()
    # R[p,s] diagonal: [0,0]=pp, [1,1]=ss
    return (r["R"][0, 0].real, r["R"][1, 1].real,
            r["T"][0, 0].real, r["T"][1, 1].real)


def smatrix_RT(rtype, ang):
    idx = np.array([n_entry, n1, n2, n_exit], dtype=complex)
    thick = [0.0, t1, t2, 0.0]
    rtypes = [0, rtype, rtype, rtype]            # [ambient, entry|l0, l0|l1, l1|exit]
    rvals = [0.0, sig[0], sig[1], sig[2]]
    S = sm.ScatterMatrix(idx, thick, wavelengths=[wl], angles=[ang],
                         roughness_types=rtypes, roughness_values=rvals)
    o = S.reflectance_transmittance("u")
    f = lambda x: float(np.asarray(x).ravel()[0])
    return f(o["Rp"]), f(o["Rs"]), f(o["Tp"]), f(o["Ts"])


print(f"{'type':<12}{'angle':>6}   {'max|dR|':>10} {'max|dT|':>10}")
worst_overall = 0.0
for rtype in range(5):  # codes 0-4: full parity (G5 intact)
    worst_t = 0.0
    for ang in angles:
        lp, ls, ltp, lts = loom_RT(rtype, ang)
        sp, ss, stp, sts = smatrix_RT(rtype, ang)
        dR = max(abs(lp - sp), abs(ls - ss))
        dT = max(abs(ltp - stp), abs(lts - sts))
        worst_t = max(worst_t, dR, dT)
        worst_overall = max(worst_overall, dR, dT)
        print(f"{ROUGH[rtype]:<12}{ang:>6.0f}   {dR:>10.2e} {dT:>10.2e}")
    print(f"  -> {ROUGH[rtype]} worst = {worst_t:.2e}")

print(f"\nCODES 0-4 WORST (all angles): {worst_overall:.3e}")
assert worst_overall < 1e-9, "roughness mismatch vs smatrix"
print("ROUTE-1 ROUGHNESS CODES 0-4 MATCH SMATRIX")

# --- code 5: split arm (Phase 8, section 8.6) ---
# Reflection arm: stack-level R mixes transmission factors back in through
# reverberation (verified: the fix shifts two-film R by ~4e-3 even though the
# per-interface reflection block is bit-identical, section 8.2 table). The
# R-match gate therefore uses a reverberation-decoupled fixture — one thick
# absorbing film, round-trip residue ~1e-12 — where R is the single-bounce
# r*f term alone. Transmission arm: standard lossless two-film fixture.
print(f"\n{'code 5 (R)':<12}{'angle':>6}   {'max|dR|':>10}")


def loom_R_decoupled(ang):
    eps_loss = np.diag([(2.0 + 1.5j) ** 2] * 3).astype(complex)
    layers = [bl.Layer(eps_loss, 400.0, roughness=(5, 8.0))]
    r = bl.BerremanStack(layers, [wl], [ang], n_entry=1.0, n_exit=1.5,
                          method="scattering",
                          exit_roughness=(5, 12.0)).solve()
    return r["R"][0, 0].real, r["R"][1, 1].real


def smatrix_R_decoupled(ang):
    idx = np.array([1.0, 2.0 + 1.5j, 1.5], dtype=complex)
    S = sm.ScatterMatrix(idx, [0.0, 400.0, 0.0], wavelengths=[wl], angles=[ang],
                         roughness_types=[0, 5, 5], roughness_values=[0.0, 8.0, 12.0])
    o = S.reflectance_transmittance("u")
    f = lambda x: float(np.asarray(x).ravel()[0])
    return f(o["Rp"]), f(o["Rs"])


worst_R5 = 0.0
for ang in angles:
    lp, ls = loom_R_decoupled(ang)
    sp, ss = smatrix_R_decoupled(ang)
    dR = max(abs(lp - sp), abs(ls - ss))
    worst_R5 = max(worst_R5, dR)
    print(f"{'Nevot-Croce':<12}{ang:>6.0f}   {dR:>10.2e}")
print(f"  -> code 5 reflection worst (decoupled) = {worst_R5:.2e}")
# System-level sanity only (<= 1e-6): the PRECISE proof that the fix left
# reflection untouched is the Rust test (R-block == upstream f at 1e-15,
# tests/navette_parity.rs). The residual ~1e-8 vs the wheel (measured 1.41e-8
# on the 0.7.0 AND 0.7.7 wheels, 1.28e-7 on a locally built 0.5.0) is the
# F1 transmission fork (our energy-conserving ga vs upstream f-on-t) leaking
# back into R through the film's internal reverberation — F3, RESOLVED
# 2026-09-18 (NAVETTE_UPSTREAM_REVIEW.md). Evidence: exit-only dressing
# agrees EXACTLY (a semi-infinite exit has no reverberation), front-only
# carries the sigma^2 law, and the residual dies with film thickness
# (reverberation kill): 6.28e-8 (400 nm) -> 5.15e-13 (800 nm) -> 0.0
# (1600 nm) — asserted below as the regression pin. Not an upstream defect:
# the wheel's bare-interface f is exact (1.7e-16) and the 0.7.7 crate source
# matches our formula bit-for-bit.
# Margin here is ~100x; a reflection-touching change would move stack R at
# the 1e-3 level (measured pre/post-fix).
assert worst_R5 < 1e-6, "code-5 reflection arm moved substantially"

# F3 regression pin: the residual is ga-vs-f reverberation feedback, so it
# MUST collapse exponentially with film thickness (absorption kills the
# round trip). Measured: 6.28e-8 (400) -> 5.15e-13 (800) -> 0.0 (1600).
def _f3_thickness_pin():
    r5 = 8.0
    out = {}
    for d in (400.0, 800.0, 1600.0):
        eps_loss = np.diag([(2.0 + 1.5j) ** 2] * 3).astype(complex)
        r = bl.BerremanStack([bl.Layer(eps_loss, d, roughness=(5, r5))],
                             [wl], [0.0], n_entry=1.0, n_exit=1.5,
                             method="scattering", exit_roughness=None).solve()
        ours = r["R"][0, 0].real
        S = sm.ScatterMatrix(np.array([1.0, 2.0 + 1.5j, 1.5], dtype=complex),
                             [0.0, d, 0.0], wavelengths=[wl], angles=[0.0],
                             roughness_types=[0, 5, 0],
                             roughness_values=[0.0, r5, 0.0])
        theirs = float(np.asarray(S.reflectance_transmittance("u")["Rp"]).ravel()[0])
        out[d] = abs(ours - theirs)
    return out

_f3_dR = _f3_thickness_pin()
print(f"  -> F3 reverberation kill: 400nm={_f3_dR[400.0]:.2e} "
      f"800nm={_f3_dR[800.0]:.2e} 1600nm={_f3_dR[1600.0]:.2e}")
assert _f3_dR[400.0] < 1e-7 and _f3_dR[800.0] < 1e-11 \
    and _f3_dR[1600.0] == 0.0, "F3 reverberation-kill pin moved"

print(f"\n{'code 5 (T)':<12}{'angle':>6}   {'max|dT|':>10}")
min_T5 = np.inf
for ang in angles:
    _, _, ltp, lts = loom_RT(5, ang)
    _, _, stp, sts = smatrix_RT(5, ang)
    dT = max(abs(ltp - stp), abs(lts - sts))
    min_T5 = min(min_T5, dT)
    print(f"{'Nevot-Croce':<12}{ang:>6.0f}   {dT:>10.2e}")
print(f"  -> code 5 transmission min |dT| = {min_T5:.2e} (must DIVERGE)")
# Fork-guard: transmission uses ga, upstream uses f; at sigma >= 5 nm the two
# differ macroscopically (measured pre/post-fix |dT| ~ 0.11-0.22 on this
# fixture, vs the pre-fix 2.3e-14 parity). Fails loudly on any return to
# bug-compatibility.
assert min_T5 > 1e-6, "code-5 transmission matches smatrix: energy bug restored?"
print("CODE 5: REFLECTION MATCHES, TRANSMISSION DIVERGES BY DESIGN")
