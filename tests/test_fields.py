# SPDX-License-Identifier: LGPL-3.0-or-later
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref"))
# NOTE: G8d needs the live BerreMueller sources (StackModel.get_in_plane_fields).
sys.path.insert(0, os.environ.get("FULL_BERREMAN_DIR", "."))
"""Gate G8: internal fields E(z)/H(z) (Phase 2).

G8a (amplitude bridge entry_amplitudes <-> Jones) lives in Rust
(fields::depth_tests::entry_amplitudes_match_solved_jones) — not repeated.
G8b: tangential (Ex,Ey,Hx,Hy) continuity across every interface <= 1e-9.
G8c: far-field asymptotes — exit pure-forward (|E| constancy), flux
     conservation (lossless), entry standing-wave contrast == |r| from solve().
G8d: vs live pyllama get_in_plane_fields (Ex,Ey at x=0) <= 1e-6 (float32).
G8e: numpy absorption profile integrated == 1 - R - T (self-normalized).
"""
import numpy as np

from navette import berreman as bl
import ref_pyllama as ref
from berremueller import pyllama as live

EPS_SLAB = np.diag([2.25] * 3).astype(complex)
SLAB = [bl.Layer(EPS_SLAB, 500.0)]
BASE_TW = np.diag([2.89, 2.25, 2.25]).astype(complex)
TW = [bl.Layer(bl.rot_z(BASE_TW, a), 50.0)
      for a in np.linspace(0, np.pi / 2, 20)]
N_ABS = 1.5 + 0.5j
ABS_SLAB = [bl.Layer(np.diag([N_ABS ** 2] * 3).astype(complex), 300.0)]
I3 = np.eye(3)
MO_SLAB = [bl.Layer(EPS_SLAB, 240.0, rho=0.02j * I3, rhop=0.02j * I3, mu=I3)]


def flux(E, H):
    """Time-averaged z-flux 1/2 Re(Ex Hy* - Ey Hx*) (Berreman units)."""
    return 0.5 * np.real(E[..., 0] * np.conj(H[..., 1])
                         - E[..., 1] * np.conj(H[..., 0]))


# ── G8b: interface continuity ────────────────────────────────────────────────
worst_b = 0.0
cases_b = [
    ("slab", SLAB, 600.0, 10.0, {}),
    ("twist20", TW, 550.0, 25.0, {}),
    ("mo", MO_SLAB, 550.0, 25.0, {}),
    ("aniso-exit", SLAB, 600.0, 10.0, {"exit_eps": np.diag([2.0] * 3)}),
]
for name, layers, wl, ang, kw in cases_b:
    st = bl.BerremanStack(layers, [wl], [ang], n_entry=1.0, n_exit=1.0, **kw)
    thick = np.array([l.thickness_nm for l in layers])
    bounds = np.concatenate([[0.0], np.cumsum(thick)])
    d = 1e-9
    pts = []
    for b in bounds:
        pts += [b - d, b + d]
    pts = np.array(sorted(set(p for p in pts)))
    f = st.fields(pts)
    assert f["n_failed"] == 0, name
    for b in bounds:
        zm, zp = b - d, b + d
        im = int(np.argmin(np.abs(pts - zm)))
        ip = int(np.argmin(np.abs(pts - zp)))
        assert abs(pts[im] - zm) < 1e-12 and abs(pts[ip] - zp) < 1e-12
        for fld in ("E_p", "E_s", "H_p", "H_s"):
            v = np.asarray(f[fld])
            # tangential components only (Ex, Ey, Hx, Hy)
            for c in (0, 1):
                worst_b = max(worst_b, abs(v[im, c] - v[ip, c]))
    print(f"G8b {name}: done")
print(f"G8b continuity worst: {worst_b:.3e}")
assert worst_b < 1e-9, "G8b failed"
print("G8b PASS")

# ── G8c: asymptotes ──────────────────────────────────────────────────────────
st = bl.BerremanStack(SLAB, [600.0], [10.0], n_entry=1.0, n_exit=1.0)
sol = st.solve()
total = 500.0
# exit pure-forward: |Ex|,|Ey| constant past the stack (both pols)
f = st.fields([total + 50.0, total + 150.0])
const = 0.0
for fld in ("E_p", "E_s"):
    v = np.asarray(f[fld])
    const = max(const, abs(abs(v[0, 0]) - abs(v[1, 0])),
                abs(abs(v[0, 1]) - abs(v[1, 1])))
print(f"G8c exit |E| constancy: {const:.3e}")
assert const < 1e-9, "G8c exit purity failed"
# flux conservation (lossless): entry-net == exit-net
fe = st.fields([-50.0, total + 50.0])
Fp_in = flux(np.asarray(fe["E_p"])[0], np.asarray(fe["H_p"])[0])
Fp_out = flux(np.asarray(fe["E_p"])[1], np.asarray(fe["H_p"])[1])
Fs_in = flux(np.asarray(fe["E_s"])[0], np.asarray(fe["H_s"])[0])
Fs_out = flux(np.asarray(fe["E_s"])[1], np.asarray(fe["H_s"])[1])
fcons = max(abs(Fp_in - Fp_out) / abs(Fp_in), abs(Fs_in - Fs_out) / abs(Fs_in))
print(f"G8c flux conservation rel: {fcons:.3e}")
assert fcons < 1e-9, "G8c flux failed"
# entry standing-wave contrast == |r| (normal incidence, single-pol)
st0 = bl.BerremanStack(SLAB, [600.0], [0.0], n_entry=1.0, n_exit=1.0)
sol0 = st0.solve()
zz = np.linspace(-600.0, -1.0, 400)
fz = st0.fields(zz)
cworst = 0.0
# driven transverse component: Ex for p-incidence, Ey for s (the other is
# identically zero at normal incidence on an isotropic slab)
for fld, comp, jr in (("E_p", 0, sol0["J_refl"][0, 0]),
                       ("E_s", 1, sol0["J_refl"][1, 1])):
    amp = np.abs(np.asarray(fz[fld])[:, comp])
    contrast = (amp.max() - amp.min()) / (amp.max() + amp.min())
    cworst = max(cworst, abs(contrast - abs(jr)))
print(f"G8c fringe contrast vs |r|: {cworst:.3e}")
assert cworst < 1e-6, "G8c contrast failed"
print("G8c PASS")

# ── G8d: vs live get_in_plane_fields ─────────────────────────────────────────
worst_d = 0.0
cases_d = [
    ("slab", np.array([EPS_SLAB]), np.array([500.0]), 600.0, 10.0),
    ("uniaxial-oblique", np.array([np.diag([2.89, 2.25, 2.25]).astype(complex)]),
     np.array([400.0]), 550.0, 30.0),
]
for name, eps_arr, th_arr, wl, ang in cases_d:
    layers = [bl.Layer(eps_arr[i], float(th_arr[i])) for i in range(len(th_arr))]
    st = bl.BerremanStack(layers, [wl], [ang], n_entry=1.0, n_exit=1.0)
    total = float(th_arr.sum())
    n_z = 41
    z = np.linspace(0.0, total, n_z)
    f = st.fields(z)
    m = live.StackModel(wl, eps_list=np.array([e if e.ndim == 2 else e[:, :, 0]
                                               for e in eps_arr]),
                        thickness_nm_list=th_arr, n_entry=1.0, n_exit=1.0,
                        theta_in_rad=np.radians(ang))
    for inp, fld in (([1.0, 0.0], "E_p"), ([0.0, 1.0], "E_s")):
        ex_m, ey_m, z_live, _ = m.get_in_plane_fields(
            np.asarray(inp), np.asarray([0.0]), n_z=n_z)
        assert np.max(np.abs(z_live - z)) < 1e-9, (name, "z grid mismatch")
        d = max(float(np.max(np.abs(np.asarray(f[fld])[:, 0] - ex_m[:, 0]))),
                float(np.max(np.abs(np.asarray(f[fld])[:, 1] - ey_m[:, 0]))))
        worst_d = max(worst_d, d)
        print(f"G8d {name} {'p' if inp[0] else 's'}: {d:.3e}")
print(f"G8d vs live worst: {worst_d:.3e}")
assert worst_d < 1e-6, "G8d failed"
print("G8d PASS")

# ── G8e: absorption integral ─────────────────────────────────────────────────
st = bl.BerremanStack(ABS_SLAB, [600.0], [0.0], n_entry=1.0, n_exit=1.0)
sol = st.solve()
assert sol["n_failed"] == 0
zz = np.linspace(0.0, 300.0, 2001)  # trapezoid error O(h^2); 2001 pts -> ~3e-7
f = st.fields(zz)
assert f["n_failed"] == 0
# Berreman-unit absorption density: A = 1/2 (w/c0) Im(eps) |E|^2 — the 1/c0
# is eps0*Z0 (H is impedance-scaled in code units); see absorption_density.
C0 = 2.998e8
omega = 2 * np.pi * C0 / (600.0 * 1e-9)
eps_im = (N_ABS ** 2).imag  # scalar: isotropic
worst_e = 0.0
for fld in ("E_p", "E_s"):
    E = np.asarray(f[fld])
    A = 0.5 * (omega / C0) * eps_im * np.sum(np.abs(E) ** 2, axis=-1)
    A_int = float(np.trapezoid(A, zz)) * 1e-9  # nm -> m
    hfld = "H_" + fld.split("_")[1]
    F_in = flux(np.asarray(st.fields([-50.0])[fld])[0],
                np.asarray(st.fields([-50.0])[hfld])[0])
    F_out = flux(np.asarray(st.fields([350.0])[fld])[0],
                 np.asarray(st.fields([350.0])[hfld])[0])
    # (A_int + F_out) / F_in == 1 (energy balance, zero convention dependence)
    bal = (A_int + F_out) / F_in
    R = float(sol["R"][0, 0] if fld == "E_p" else sol["R"][1, 1])
    T = float(sol["T"][0, 0] if fld == "E_p" else sol["T"][1, 1])
    # A_int = (1-R-T) F0 with F_in = (1-R) F0
    worst_e = max(worst_e, abs(bal - 1.0),
                  abs(A_int / F_in - (1 - R - T) / (1 - R)))
    print(f"G8e {fld}: balance {bal:.9f}, A/F_in {A_int / F_in:.6f} vs "
          f"{(1 - R - T) / (1 - R):.6f}")
print(f"G8e worst: {worst_e:.3e}")
assert worst_e < 1e-6, "G8e failed"
print("G8e PASS")

# ── periods expansion contract + validation ──────────────────────────────────
CELL = [bl.Layer(np.diag([4.0] * 3).astype(complex), 70.0),
        bl.Layer(np.diag([2.25] * 3).astype(complex), 90.0)]
z = np.linspace(-50.0, 16 * 160.0 + 50.0, 60)
a = bl.BerremanStack(CELL, [600.0], [10.0], periods=10).fields(z)
b = bl.BerremanStack(CELL * 10, [600.0], [10.0]).fields(z)
dper = max(float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k]))))
           for k in ("E_p", "H_p", "E_s", "H_s"))
print(f"periods expansion field agreement: {dper:.3e}")
assert dper < 1e-12
ROUGH_CELL = [bl.Layer(np.diag([4.0] * 3).astype(complex), 70.0, roughness=(4, 2.0)),
              bl.Layer(np.diag([2.25] * 3).astype(complex), 90.0)]
try:
    bl.BerremanStack(ROUGH_CELL, [600.0], [10.0]).fields([10.0])
    raise SystemExit("MISSING-ERROR: rough fields accepted")
except ValueError:
    pass
try:
    bl.BerremanStack(SLAB, [600.0], [10.0]).fields([])
    raise SystemExit("MISSING-ERROR: empty z accepted")
except ValueError:
    pass
try:
    bl.BerremanStack(SLAB, [600.0], [10.0]).fields([10.0], method=7)
    raise SystemExit("MISSING-ERROR: bad method accepted")
except ValueError:
    pass
print("periods + validation PASS")
