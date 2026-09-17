# navette-berreman — 4×4 Berreman / Müller solver (Rust + PyO3)

A Rust port of the `pyllama` / `berreman_mueller` 4×4 Berreman–Müller algorithm
for optical birefringent multilayer systems, styled after the existing
`navette-smatrix` crate. The physics core is pure Rust (only `num-complex`, with
a self-contained complex eigensolver — no LAPACK); an optional `python` feature
builds a PyO3 extension module (`navette._berreman`) for use from NumPy.

## Architecture (same split as upstream `navette`)

Every numeric kernel lives in Rust; the Python layer owns only validation,
marshalling and result shaping — the same contract as upstream `navette`'s
`smatrix.py` ("owns no physics"):

| kernel | Rust home | Python role |
|---|---|---|
| Berreman matrices, eigensolver, partial waves | `berreman.rs`, `cmatrix.rs` | — |
| Redheffer SM / TM / EM propagation, periodic power | `transfer.rs`, `expm.rs` | — |
| Route-1 roughness form factors, Route-2 graded profile (erf) | `roughness.rs` | `_norm_rough` validation only |
| internal fields E(z)/H(z), absorption | `fields.rs` | z-array marshalling |
| rotations (Rodrigues/Euler/quat), twist schedule | `rotations.rs` | arg validation |
| materials (23 models + 7 EMA), ε↔n̂ | `materials.rs` (upstream crate) | spec dispatch/defaults (upstream-idiomatic) |
| Mueller suite (DI/D/P/CD/Cloude) | `mueller.rs` | broadcasting loop over flat-16 rows |
| Pasteur mapping κ→(ρ,ρ′) | `berreman::pasteur_tensors` | warning regime + Layer build |
| Rayon (λ, θ) sweeps, NaN policy | `pybind.rs` | dict assembly, squeezing |

Stack *construction* helpers (`twisted_stack`, `grade_interface`,
`graded_stack`, `cholesteric_stack`, `chiral_layer`) stay in Python as
`Layer`-list builders, but their math (rotation schedule, Gaussian-CDF mixing,
Pasteur tensor mapping) is called from Rust — pinned exact by G13f (0.0 vs the
numpy reference).

## What it computes

For a stack of (generally anisotropic, optionally magneto-optic) layers between
two isotropic half-spaces, over a grid of wavelengths × incidence angles:

* complex Jones reflection / transmission matrices in the linear (p, s) basis
  and the circular (L, R) basis (pyllama convention),
* power reflectance / transmittance `R`, `T`,
* 4×4 Müller matrices for reflection and transmission,

via the **scattering-matrix** (Redheffer), **transfer-matrix**, or direct
**exponential-matrix** (`method="exponential"`/`"em"`, == pyllama EM) method —
both validated to agree to machine precision.

## Build

```bash
# pure-Rust core + standalone validation (no Python needed)
cargo test --release
cargo run --release --example validate          # prints JSON for all 5 cases

# the Python extension module (via maturin, using pyproject.toml)
maturin develop --release          # build + install into the active venv
# or build a wheel:
maturin build --release            # -> target/wheels/navette_berreman-*.whl
```

The compiled extension is injected into the `navette` namespace as
`navette._berreman`; `navette/berreman.py` is the thin hand-written wrapper.
After install: `from navette import berreman as bl`.

### Toolchain / version pins

Built and validated with **rustc 1.98.1** (the same stable the upstream
`navette` 0.7.0 wheel was compiled with), PyO3 **0.29.2**, numpy crate **0.29**,
NumPy 2.x, maturin 1.14 (pinned `>=1.5,<2.0` like upstream), edition 2024,
`abi3-py312` (forward-compatible wheels from 3.12 up). Policy mirrors upstream
`navette`, with one addition:

* **No rust-toolchain.toml, no rustc ceiling** — the crate rides stable, same
  as upstream's release CI (`dtolnay/rust-toolchain@stable`). Latest stable is
  validated by every run of the gate suite.
* **MSRV floor declared** (`rust-version = "1.85"` in Cargo.toml): edition 2024
  needs 1.85, which dominates pyo3/numpy 0.29's own 1.83. Upstream declares no
  MSRV; our floor documents the verified minimum instead.
* **`Cargo.lock` is committed** — for a cdylib shipped as wheels the lock is
  part of the build contract (upstream commits theirs in the workspace root);
  the full gate suite is the re-audit tripwire when it changes.
* Dependency policy: only `num-complex` (core), the upstream `navette` crate
  + `ndarray` 0.15 (materials adapters), and pyo3/numpy/rayon under the
  `python` feature. caret-pinned minors, exact resolution by lock.

## Python API

```python
import numpy as np
from navette import berreman as bl

# a 250 nm rotated-uniaxial slab on glass, swept over wavelengths & angles
eps = bl.rot_z(np.diag([2.25, 2.89, 2.25]).astype(complex), np.radians(35))
stack = bl.BerremanStack(
    layers=[bl.Layer(eps, thickness_nm=250.0)],
    wavelengths_nm=[450, 550, 650],
    angles=[0, 15, 30, 45],            # degrees (angles_in_radians=True to override)
    n_entry=1.0, n_exit=1.5,
    method="scattering",                # or "transfer"
)
out = stack.solve()
# out["R"], out["T"]                 -> [n_wl, n_angle, 2, 2] real (p,s)
# out["J_refl"], out["J_trans"]      -> [n_wl, n_angle, 2, 2] complex
# out["J_refl_circ"], out["J_trans_circ"]
# out["M_refl"], out["M_trans"]      -> [n_wl, n_angle, 4, 4] real
# singleton wavelength / angle axes are squeezed away
```

### Magneto-optic (full Berreman) layers

Supply per-layer magneto-electric (`rho`, `rhop`) and magnetic (`mu`) tensors to
switch the whole stack onto the full Berreman matrix:

```python
bl.Layer(eps, 240.0, rho=rho, rhop=rhop, mu=mu)   # 3×3 complex tensors
```

Helpers: `bl.mueller_from_jones(jones2x2)`, `bl.rot_z(tensor, angle_rad)`
(now Rust-backed), `bl.rot_axis(axis, theta, eps)` (== pyllama `rot_mat`),
`bl.rot_euler(rx, ry, rz, eps)`, `bl.rot_quat(w, x, y, z, eps)`
(scalar-first quat; live parity ~1e-15 in `tests/test_rotations.py`).

### Anisotropic exit half-space

Pass `exit_eps` (a `(3,3)` or `(n_wl,3,3)` tensor, + optional
`exit_rho`/`exit_rhop`/`exit_mu` for magneto-optic exits) to radiate into a
birefringent substrate — ahead of pyllama (scalar exits only), at parity
with Berreman4x4 (whose Jones reflection matches to 7e-16). Amplitude Jones
works in all three methods; transmitted *power* uses Poynting-flux ratios
over the exit eigenmodes (`power_method` reports `"flux"` vs
`"jones_factor"`, and transmitted `T` is diagonal-only: no (p,s)-projected
cross power in a mode-resolved basis). `exit_eps` and non-unity `n_exit` are
mutually exclusive. Like B44, there is no scalar power-correction factor for
anisotropic backs — `factor` stays informational.

```python
stack = bl.BerremanStack([bl.Layer(eps_lc, 500.0)], [550.0], [30.0],
                         n_entry=1.0, exit_eps=eps_substrate)
res = stack.solve()  # res["power_method"] == "flux"
```

### Periodic stacks (Bragg fast path)

Pass `periods=N` to repeat the layer list N times without expansion:
identical to `layers * N` (including roughness dressings), with L instead
of N·L eigendecompositions and O(log N) matrix combines. A 40-period DBR
sweep runs ~10× faster at 7e-14 agreement.

```python
dbr = bl.BerremanStack([bl.Layer(epsH, 70.0), bl.Layer(epsL, 90.0)],
                       wavelengths, angles, periods=40)
```

### Internal fields E(z)/H(z)

Full 6-vector per input polarization at arbitrary depths (0 = entry plane,
negatives = entry side). Propagation is always transfer-based (same P/Q as
TM); `periods` expands the cell; roughness needs graded expansion first.
Longitudinal note: Ez/Hz follow the local medium (normal-D continuity ⇒
Ez jumps across interfaces); z == total thickness returns exit-side values,
so absorption integrals must use interior-side limits.

```python
f = stack.fields(np.linspace(-100, 800, 901))
Ep, Hp = f["E_p"], f["H_p"]  # complex [n_wl, n_th, n_z, 3]
```

### Chiral media (Pasteur–Tellegen)

SI definition (e^{−iωt}): `D = εE + iξH`, `B = −iξE + μH` with `ξ = κ/c₀`
(κ dimensionless). Code units carry impedance-scaled H, and the engine's
slots couple opposite to that derivation (recorded TIME_CONVENTION =
e^{+iωt}, pinned operationally) — so the mapping is `rho = −iκI₃`,
`rhop = +iκI₃`, with eigen-indices `n ± κ` (`n + κ` on the RCP channel).
Warns for `|κ| ≥ 0.2n`; `κ² = n²` is singular (NaN + `n_failed`).
Normal-incidence Pasteur slab: transmission keeps the one-way chiral phase
(`t_R/L = t₀·e^{±ik₀κd}`), reflection is κ-independent (round-trip
cancellation) and flips helicity — validated analytically to ~1e-14 plus an
independent scipy-expm replication. Cholesteric Bragg stacks via
`cholesteric_stack` match live B44 essentially exactly. Live pyllama can
never oracle chiral stacks (its det typo scales every a_i by det² with
κ ≠ 0) — validation is analytic + B44-structural by design.

```python
slab = bl.chiral_layer(1.5, 0.05, 500.0)
chol = bl.cholesteric_stack(1.5, 1.7, 350.0, 5)  # 48 slices/pitch
```

**Coming from DBF/β or ORD literature?** Bridges (see `CHIRAL_BRIDGE.md`
for the derivations and gate values):

- `dbf_beta_to_kappa(beta, n, wavelength_nm)` — Drude–Born–Fedorov β
  (`D = ε(E+β∇×E)`) → our κ, reproducing the DBF circular birefringence
  `Δn = 2βk₀n²/(1−β²k₀²n²)` exactly; `kappa_to_dbf_beta` inverts.
- `condon_kappa(wavelengths_nm, R, lambda0_nm, gamma)` — Condon's
  single-oscillator chirality dispersion `κ(ω) = ωR/(ω₀²−ω²−iωΓ)`
  (Condon–Altar–Eyring 1937; Lindell 1994 form), returned in our
  normalization (`n± = n±κ`); feeds `kappa_table`/Table interpolation,
  γ>0 gives complex κ for the tensor path.

### Mueller-matrix post-processing

Elementwise over the solved `M_refl`/`M_trans` grids (`(...,4,4)` in,
metrics out; unphysical M00 ≤ 0 maps to NaN):

```python
res = stack.solve()
DI = bl.depolarization_index(res["M_trans"])  # 1 == non-depolarizing
D, P = bl.diattenuation(res["M_trans"]), bl.polarizance(res["M_trans"])
CD = bl.circular_dichroism(res["M_trans"])    # M03/M00, dimensionless
cc = bl.cloude(res["M_trans"])                 # {"lambda": (...,4), "entropy": (...)}
```

The compiled module exposes `solve_grid_simple`, `solve_grid_full`, and
`mueller_from_jones`; the wrapper marshals tensors into the flat `[n_wl ·
n_layers · 18]` (re, im)-pair layout these expect.

### Surface roughness

Two models, both ported from the `smatrix` engine, with the **same integer
codes**:

| code | model                        |
|------|------------------------------|
| 0    | none                         |
| 1    | uniform / box                |
| 2    | two-delta                    |
| 3    | Lorentzian / exponential     |
| 4    | Gaussian (Debye–Waller)      |
| 5    | Névot–Croce: `exp(−2·ka·kb·σ²)` on reflection, `exp(−(Δkz)²σ²/2)` (= Gaussian W₄) on transmission (deliberate fix, see below) |

**Route 1 — specular attenuation (Névot–Croce / W dressing).** A direct
generalization of the smatrix dressing to anisotropic partial waves: each
element of the interface scattering matrix coupling modes with z-wavevectors
`kz = k₀·q` is multiplied by the height-distribution form factor evaluated at
the wavevector transfer. In the isotropic limit the forward/backward modes are
degenerate, the factor is constant within each reflection/transmission block,
and it reduces **exactly** to the smatrix scalar dressing (`r₁₂·W(2kz₁σ)`,
`r₂₁·W(2kz₂σ)`, `t·W((kz₁−kz₂)σ)`; type 5 → `exp(−2kz₁kz₂σ²)` on reflection).
On **transmission** type 5 uses the Gaussian transfer factor instead (see the
second intentional deviation below). Roughness is a
property of an interface; `roughness=(type, σ_nm)` on a `Layer` dresses the
interface at its **front**, and `exit_roughness` dresses the final interface.
Route 1 is a coherent/specular model and is **scattering-method only**.

```python
stack = bl.BerremanStack(
    layers=[bl.Layer(eps1, 120.0, roughness=(bl.ROUGH_NEVOT_CROCE, 8.0)),  # entry|l0
            bl.Layer(eps2,  90.0, roughness=(bl.ROUGH_GAUSSIAN,     5.0))], # l0|l1
    wavelengths_nm=[550.0], angles=[0, 20, 40, 60],
    n_entry=1.0, n_exit=1.5, method="scattering",
    exit_roughness=(bl.ROUGH_GAUSSIAN, 12.0),                               # l1|exit
)
```

Codes are also exposed as `bl.ROUGH_NONE`, `…_UNIFORM`, `…_TWODELTA`,
`…_LORENTZIAN`, `…_GAUSSIAN`, `…_NEVOT_CROCE` (0–5).

> **Route-1 anisotropic factors are an ansatz.** No external
> anisotropic-roughness reference exists (neither Berreman4x4 nor pyllama
> implements roughness), so the per-mode generalization is validated only in
> (a) the isotropic limit — exact reduction to smatrix for codes 0–4 and to
> the *corrected* smatrix semantics for code 5 — and (b) the σ→0 limit —
> exact reduction to smooth for all codes. Reflection uses the NC cross
> factor per mode pair; transmission uses the Gaussian transfer factor of the
> mode-pair wavevector transfer. Treat large-σ anisotropic predictions as
> model-dependent and cross-check against Route 2 (graded sublayers, G13e).

**Route 2 — graded effective-medium sublayers.** Replaces a rough interface by
thin homogeneous sublayers whose permittivity **tensor** interpolates between
the two media along the Gaussian volume-fraction profile
`f(z)=½(1+erf(z/(σ√2)))`. It makes no perturbative assumption, handles tensors
directly, and works in **both** methods (each sublayer is an ordinary layer).

```python
graded = bl.graded_stack(base_layers, interface_sigma_nm=[8, 5, 12],
                         n_entry=1.0, n_exit=1.5, n_sublayers=9)
out = bl.BerremanStack(graded, [550.0], [30.0],
                       n_entry=1.0, n_exit=1.5, method="transfer").solve()
# or build a single transition region directly:
subs = bl.grade_interface(eps_a, eps_b, sigma_nm=8.0, n_sublayers=7)
```

Route-2 limitations (each surfaced, none silent):

* Transition sublayers are **non-magnetic** — `rho`/`rhop`/`mu` are dropped
  inside the graded region (`UserWarning` naming the affected layers).
* **Dispersive** `(n_wl,3,3)` eps + roughness is unsupported — `grade_interface`
  raises `ValueError`; build sublayers per wavelength instead.
* Layers thinner than the 3σ shave are **clamped** to 1e-9 nm (`UserWarning`;
  thicken the layer or reduce σ).

### Materials (dispersion → tensors)

`navette.berreman_materials` evaluates the 23 upstream dispersion models
(Konstant … Roughness, including the Cody/Tauc/UBF KK trio) through the
`navette` crates.io kernels — same spec vocabulary as upstream
`navette.materials.evaluate`, live parity **0.0** on all models
(`tests/test_materials.py`). Named `berreman_materials` (not `materials`)
because the upstream wheel already owns `navette.materials` in the merged
namespace. `evaluate_tensor` builds birefringent `diag(n̂²)` stacks
(isotropic / uniaxial / biaxial + optional `R eps Rᵀ` rotation) for
`bl.Layer`; EMA mixes per axis. One deliberate delta: malformed `Table`
input raises `ValueError` here instead of upstream's `PanicException`.

```python
from navette import berreman_materials as mm
bk7 = {"model": "Sellmeier", "params": {"B1": 1.03961212, ...}}
eps_wl = mm.evaluate_tensor(bk7, wavelengths)  # (n_wl,3,3) diag eps = n̂²
lyr = bl.Layer(eps_wl, 500.0)                  # dispersive Layer: (n_wl,3,3)
out = bl.BerremanStack([lyr], wavelengths, angles).solve()
```


## Conventions (faithful to pyllama)

* Field 4-vector ψ = [Eₓ, H_y, E_y, −Hₓ]; eigenvectors form the columns of `P`,
  propagation `Q = diag(exp(i·k₀·q·d))`.
* `Kx = n_entry·sin θ_in`; `R = |J_refl|²`, `T = (Kz_exit/Kz_entry)·|J_trans|²`.
* Circular basis: `J_refl_c = B⁻¹·J_refl·F`, `J_trans_c = F⁻¹·J_trans·F` with
  `F = [[1,1],[−i,i]]`, `B = [[1,1],[i,−i]]`.
* Isotropic layers are routed through the analytic half-space basis to avoid the
  degenerate-eigenvector instability a numerical eig would hit there.

## One intentional deviation from the literal source

`full_berreman.py` writes

```python
d = 1/(eps[2,2]*mu[2,2] - rho[2,2]*rhop[2,2])
a_i = numerator / d          # == numerator * (eps22·mu22 − rho22·rhop22)
```

Taken literally this multiplies by the determinant, and the resulting matrix
does **not** reduce to the reduced ("simple") Berreman matrix when `mu = I`,
`rho = rhop = 0` — contradicting the file's own docstring and the cited
Mazur / Azzam–Bashara derivation. The intended operation is **division by the
determinant**. This port divides by the determinant (`a_i = numerator / det`),
which reproduces the simple matrix exactly in that limit (verified to 1e-16) and
matches the corrected reference for genuine gyrotropic cases (worst error
1.7e-15); see the comment in `src/berreman.rs`.

Second, **type-5 (Névot–Croce) transmission**. The old upstream code applied
the reflection factor `f = exp(−2·kz₁·kz₂·σ²)` to transmission as well
(upstream `docs/code_review.md` §3.2: R+T = 0.925 at a single interface with
σ = 10 nm). Release 0.7.0 instead uses a *growing* transmission factor
`ga = exp(+((kz₁−kz₂)·σ)²/2)` — a deliberate perturbative R+T=1 model with a
tight validity budget (their docs: unphysical past σ ≈ 0.0159·λ/Δn, R+T up
to 49.4). This port uses the textbook decaying Gaussian transfer factor
`ga = exp(−(kz₁−kz₂)²σ²/2)` on transmission; reflection (`f`) is formula-
identical to upstream (proven at 1e-15 in `tests/navette_parity.rs`). The two
re-converge as σ→0. Pinned by the code-5 fork-guard in `test_roughness.py`
(transmission diverges by design) and the energy gate
`test_roughness_energy.py`.

## Validation

* `examples/validate.rs` + `ref/ref_pyllama.py` + `ref/diff_check.py`: 5 cases
  (isotropic, uniaxial, rotated-uniaxial oblique, two-layer, absorbing) × both
  methods vs a NumPy oracle — worst abs error **6.3e-15**; lossless `R+T = 1`.
* `test_e2e.py`: extension + wrapper over a wavelength×angle grid vs the oracle —
  worst **9.4e-15**; `Mueller(I) = I`.
* `test_full.py`: full magneto-optic path — exact reduction to the simple path,
  and a gyrotropic case vs the corrected reference at **1.7e-15**.
* `test_roughness.py`: Route-1 roughness vs the `smatrix` engine on an identical
  isotropic stack — codes 0–4 × four angles, R and T (s and p), worst
  **2.3e-14**; code 5 reflection matches on a reverberation-decoupled
  absorbing fixture while transmission diverges by design (fork-guard, see above).
* `test_roughness_energy.py` (G13a): specular energy conservation, all codes ×
  sigmas × angles on lossless stacks (R+T ≤ 1) plus an absorbing code-5 arm.
* `test_graded.py` (G13b–e): Route-2 σ=0 identity (exact), SM≡TM agreement,
  sublayer convergence (monotone), and Route-1 vs Route-2 cross-check.
  Route 1 reduces to smooth exactly at σ=0 on anisotropic layers.
