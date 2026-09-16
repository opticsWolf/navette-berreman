# navette-berreman — Rust Implementation Plan

**Goal:** close the feature gaps vs the two reference implementations
(`Berreman4x4/Berreman4x4` = Castany, `andrewsalij/BerreMueller` = Salij/pyllama)
with new **Rust code paths** in this crate, validated to the same
machine-precision gates the crate already holds (≈ 1e-15 vs both references).

**Status of validation (Sep 2025, see prior session):**
- navette vs *live* BerreMueller `StackModel` (5 canonical cases + twisted-nematic
  20-slice + 10× Bragg, SM + TM): worst ≈ **6.3e-15**.
- navette vs *live* Berreman4x4 (5 slabs × 3 angles × SM/TM, R/T power): worst ≈ **1.9e-15**.
- `mueller_from_jones` vs live: 8.9e-16. Bundled `ref/ref_pyllama.py` Δ vs live `_build_D`: 0.0.
- Full-path deviation is intentional and confirmed: live literal
  `calc_berreman_matrix(mu=I, rho=0)` misses the simple matrix by 0.85
  (the `1/d` typo); we divide by `det` and reduce exactly (0.0).

**Phases in this plan:**

| # | Feature | Upstream ref | Rust work | Size |
|---|---------|--------------|-----------|------|
| 0 | Baseline, gates, conventions | — | none (survey) | S |
| 1 | Anisotropic exit half-space | B44 anisotropic back; pyllama gap | `berreman.rs`, `transfer.rs`, `pybind.rs` | M |
| 2 | Internal fields E(z)/H(z) | `pyllama.get_in_plane_fields`, `field_plotting.py` | new `fields.rs` + bindings | L |
| 3 | EM exponential-matrix method | `Structure.build_exponential_matrix` | new `expm.rs` + `transfer.rs` | M |
| 4 | Periodic / Bragg fast path (`N_periods`) | `Structure(N_periods)`, `StackModel(N_per)`, B44 `RepeatedLayers` | `transfer.rs` (pow/combine-pow) | M |
| 5 | Mueller suite (Cloude, DI, P/D) | `mueller.py` | new `mueller.rs` + bindings | M |
| 6 | Rotations in Rust (`rotations.rs`) | `dielectric_tensor.py`, `rot_mat` | new `rotations.rs` + bindings | S |
| 7 | Validation harness + release | `examples/validate.rs`, `tests/` | tests, docs, gates | S |
| 8 | Roughness correctness (type-5 fix, Route-2 cover) | Navette `code_review.md` §3.2; smatrix `coherent_block.rs` | `roughness.rs`, `berreman.py`, `tests/` | S |
| 9 | Materials (dispersion → tensors, Rust + Python) | Navette `materials/` (Rust kernels + `materials/__init__.py`) | new `materials.rs` + bindings | M |
| 10 | Chiral media (Pasteur constitutive + cholesteric Bragg) | pyllama `full_berreman` (mapping); B44 `TwistedMaterial` | `berreman.py`, `tests/` | M |

> Scope note: twisted/cholesteric stack builders, dispersion interpolation,
> Ψ/Δ ellipsometry are **pure-Python** helpers on top of `BerremanStack` and are
> specified briefly in Phase 7 §C. Everything else below is Rust.
>
> Tooling remarks (all phases): use **coderadar** for code analysis — semantic
> structural queries over this repo and the reference checkouts (`/tmp/ghread/`)
> instead of re-deriving context with grep/Read loops; it can also apply the
> code edits specified in each phase. Use the **gossamer skill** for web
> research to clarify topics before encoding them — especially physics
> questions (constitutive conventions, dispersion models, tolerance
> precedents), as was done for the §10.7 DBF study. Prefer graph/skill lookup
> over guessing from training data whenever a convention or formula is at stake.

---

## Phase 0 — Baseline, gates, conventions (read-only)

### 0.1 Repository map (as of this writing)

```text
navette_berreman/
├── Cargo.toml               # lib name `_berreman`, cdylib+rlib; `python` feature = pyo3+numpy+rayon
├── pyproject.toml           # maturin, module-name = "navette._berreman", features=["python"]
├── navette/berreman.py      # hand wrapper: BerremanStack, Layer, graded_stack, rot_z, ...
├── src/
│   ├── lib.rs               # module decls + #[pymodule] _berreman
│   ├── cmatrix.rs           # C=Complex64, Mat2/Mat4, mul/inv, similarity-diag, eig4 (LeVerrier+DK+Newton)
│   ├── berreman.rs          # Δ simple/full, Wave, sort_partial_waves, layer_waves_*, halfspace_waves
│   ├── transfer.rs          # SLayer, transfer_* , scattering_*, fresnel_*, power, circ, mueller, solve_stack
│   ├── roughness.rs         # Route-1 W(q) factors, interface_factors
│   └── pybind.rs            # solve_grid_simple/full, mueller_from_jones_py, Rayon sweep
├── examples/validate.rs     # 5 canonical cases → JSON (SM+TM)
├── tests/{test_e2e,test_full,test_roughness}.py
└── ref/ref_pyllama.py       # bundled NumPy oracle (Δ bit-identical to live)
```

### 0.2 Build / test commands (canonical)

```bash
cargo test --release                    # pure-Rust unit tests (no Python)
cargo run --release --example validate  # JSON oracle dump
# Python extension (needs VIRTUAL_ENV set to a venv with numpy):
VIRTUAL_ENV=/path/to/venv maturin develop --release
PYTHONPATH=ref <venv-python> tests/test_e2e.py
PYTHONIOENCODING=utf-8 PYTHONPATH=ref:<berremueller>/src FULL_BERREMAN_DIR=<berremueller>/src/berremueller \
    <venv-python> tests/test_full.py
<venv-python> tests/test_roughness.py          # needs PyPI `navette` in the venv (§0.5)
<venv-python> tests/test_roughness_energy.py   # no upstream needed
<venv-python> tests/test_graded.py             # no upstream needed
```

### 0.5 Upstream dependencies (added P8 — read before touching)

Two upstream channels, different jobs:

- **Rust (crates.io): `navette = "0.7"` as a dev-dependency.** Compiles the
  upstream engine for `cargo test` only — the shipped `_berreman` extension
  is unaffected (dev-deps excluded from the cdylib build). Consumers:
  `tests/navette_parity.rs` calls `w_function_inner` / `nevot_croce_factors`
  directly (formula parity at 1e-15, §8.6). Re-audit the pinned findings on
  any upgrade — the tests fail loudly by design (fork-guard philosophy).
- **Python (PyPI `navette`, same venv):** provides `navette.smatrix` (G5 live
  arm) and `navette.materials` (G14a later). Namespace merge: this repo's
  `navette/` has no `__init__.py`, so it merges with the wheel *only if* the
  wheel's shadowing `__init__.py` is disabled — it is renamed to
  `__init__.py.disabled_by_navette_berreman_*` in site-packages (reversible;
  **re-apply after any wheel reinstall/upgrade**, `test_roughness.py` asserts
  the merge with a loud guard). Upstream is LGPL-3.0-or-later — consistent
  with the Phase 9 license decision (D9.1). Suspected upstream defects found
  during validation are filed separately in `NAVETTE_UPSTREAM_REVIEW.md`
  (never mixed into this plan's spec sections) and that file is kept
  current as work proceeds — new finding → new F-entry + Changelog line,
  same session, no exceptions.

### 0.3 Conventions that MUST NOT drift (gates depend on them)

- Field 4-vector `ψ = [Ex, Hy, Ey, −Hx]`; `P` columns are eigenvectors in
  sorted order `[t0, t1, r0, r1]`; `Q = diag(exp(i·k0·q·d))`.
- `Kx = n_entry.re · sin θ` (real part only); `k0 = 2π/λ[nm]`.
- Jones layout `[[r_pp, r_sp],[r_ps, r_ss]]`, i.e. `[0][1]` = s→p
  (live pyllama calls the same slot `r_s_to_p`; B44 labels it `r_ps` — same slot).
- Circular: `F = [[1,1],[−i,i]]`, `B = [[1,1],[i,−i]]`;
  `J_refl_c = B⁻¹·J·F`, `J_trans_c = F⁻¹·J·F`.
- Power: `R = |J|²`, `T = Re(factor)·|J|²`, `factor = Kz_exit/Kz_entry`.
- Core returns `Option<Mat4>` / `Option<SolveResult>` on singularity —
  **never panic in physics code**; `pybind` maps `None → NaN` row + `n_failed++`.
- Isotropic fast path: layers with `eps ∝ I` (tol 1e-12) use analytic
  `halfspace_waves`, never the numeric eig. Any new path must preserve this
  (bit-identity gate, §0.4).

### 0.4 Standing acceptance gates (every phase must keep these green)

| Gate | Command / check | Tolerance |
|------|-----------------|-----------|
| G1 Rust unit | `cargo test --release` (unit + `navette_parity`) | all pass |
| G2 canonical SM+TM | `validate` vs `ref_pyllama.test_cases()` | ≤ 1e-9 (observed 1e-15) |
| G3 e2e grid | `tests/test_e2e.py` | assert 1e-6 (observed 9.4e-15) |
| G4 full MO | `tests/test_full.py` | reduction 1e-12, MO 1e-6 (observed 0 / 1.7e-15) |
| G5 roughness | `tests/test_roughness.py` codes 0–4 (needs PyPI navette); code-5 R sanity + T fork-guard (§8.6) | 0–4: 1e-9 (obs. 2.2e-14); 5T diverges |
| G6 isotropic bit-identity | any new path with isotropic I/O ≡ old path | ≤ 1e-15, target 0.0 |
| G7 aniso exit | `tests/test_aniso_exit.py` (needs B44 sources); Rust G7a/G7c | Jr+R ≤ 1e-9 (obs. 7.3e-16); T-flux ≤ 1e-9 (obs. 1.4e-15); G7a 0.0; G7d bounds |
| G9 exponential | `tests/test_em.py` (needs BerreMueller sources for G9c) + G9d arm | EM≡TM ≤ 1e-12 (obs. 8.5e-15); live-EM ≤ 1e-9 (obs. 2.3e-15); B44 ≤ 1e-9 (obs. 1.4e-15) |
| G10 periodic | `tests/test_periodic.py` (needs BerreMueller sources for G10c) | expand≡period ≤ 1e-12 (obs. 5.6e-15); live N_per ≤ 1e-9 (obs. 2.2e-15); speedup ≥ 2x (obs. 9.8x) |
| G11 mueller suite | `tests/test_mueller_suite.py` (needs BerreMueller sources + sympy/pandas) | vs live ≤ 1e-12 (obs. 3.6e-15); slab DI=1 ≤ 1e-9 (obs. 2.2e-16) |
| G8 fields | `tests/test_fields.py` (needs BerreMueller sources for G8d) | continuity ≤ 1e-9 (obs. 4.9e-11); exit/flux ≤ 1e-9 (obs. 1.1e-16/2.7e-16); contrast ≤ 1e-6 (obs. 2.9e-7); live ≤ 1e-6 (obs. 3.9e-8); absorption ≤ 1e-6 (obs. 2.1e-7); periods 0.0 |
| G15 chiral (10a analytic, no external ref) | `tests/test_chiral.py` | G15a iso-floor ≤ 1e-9 (obs. 1.6e-10), aniso 0.0, singularity n_failed = 1; G15b trans 3.2e-14/refl 9.6e-15/forbidden 3.2e-14 (≤ 1e-12); G15c rotation 4.1e-14, antisym 1.6e-16, energy 1.7e-15; oblique odd/even 0.0, k→0 3.3e-10, Lorentz-transpose 1.2e-15; G15e 7.3e-9 |
| G15d cholesteric Bragg vs live B44 | `tests/test_cholesteric.py` + `ref/ref_out_cholesteric.json` (+ `ref/gen_cholesteric_ref.py`) | pos 0.0 (≤ 0.5%), height 1.7e-15 (≤ 3%), FWHM 4.8e-15 (≤ 5%); swap both sides; triple monotone, 48-vs-96 4.3e-4; JSON spot-check 1.7e-15 |
| G12 rotations | `tests/test_rotations.py` (needs BerreMueller sources) | live ≤ 1e-14 (obs. ~1e-15); rot_z ≤ 1e-15; group exact |
| G13 roughness-energy | `tests/test_roughness_energy.py` + `tests/test_graded.py` (no smatrix needed) | R+T bounds, Route-2 pins (see Phase 8) |
| G14 materials | `tests/test_materials.py` (needs upstream `navette.materials`) | Tier-1 ≤ 1e-15 target 0.0; tensor exact |
| G15 chiral | `tests/test_chiral.py` (analytic) + cholesteric vs live B44 | spectrum/handedness exact; slab ≤ 1e-12; Bragg pos ≤ 0.5% |

New phases add new gates G7+ (defined per phase). A phase is done only when
G1–G6 plus its own gates pass. Phase 8 adds G13 and re-scopes G5's code-5 arm
(type-5 transmission intentionally diverges from smatrix after the §8.3 fix).
---

## Phase 1 — Anisotropic exit half-space

### 1.1 Why / upstream reference

- **Berreman4x4 supports it today:** `Structure(front, layers, back)` explicitly
  documents *"back half-space (exit), may be anisotropic"*
  (`Berreman4x4/Berreman4x4.py`, class `Structure`, `getStructureMatrix` builds
  `T = Lf⁻¹ · P · Lb` with a general `Lb`). Its `getPowerTransmissionCorrection`
  returns `None` for an anisotropic back — i.e. amplitude Jones works, scalar
  power correction does not. We mirror that semantic.
- **BerreMueller/pyllama does NOT:** `Model(n_entry, n_exit, ...)` and
  `Structure.get_refl_trans` hard-assume scalar `Kz_exit/Kz_entry`. This phase
  puts navette *ahead* of pyllama and at parity with B44.
- Use case: LC cell radiating into a birefringent substrate, Wollaston-type
  exits, anisotropic sensor stacks.

### 1.2 Current code (what changes)

`src/transfer.rs::solve_stack` builds both half-spaces analytically:

```rust
let entry_w = halfspace_waves(geom.n_entry * geom.n_entry, kx);
let exit_w  = halfspace_waves(geom.n_exit  * geom.n_exit,  kx);
let entry = SLayer::half_space(&entry_w);
let exit  = SLayer::half_space(&exit_w);
```

and the power factor uses scalar `kz_exit/kz_entry`. Key insight: the
**transfer and scattering assemblers already accept a general exit `P`** —
`transfer_matrix` does `exit.P⁻¹ · T · entry.P`, `s_to_next` only shuffles
columns of `a.p`/`b.p`. So the amplitude path needs *no* new linear algebra,
only (a) a general exit eigenbasis, (b) a flux-based power for that case.

### 1.3 Design

```rust
// transfer.rs (new)
/// How the exit half-space is described.
#[derive(Clone)]
pub enum ExitSpec {
    /// Scalar index -> analytic isotropic basis (existing path, bit-identical).
    Isotropic(C),
    /// Full permittivity tensor (+ optional MO set) -> numeric eigenbasis.
    Anisotropic {
        eps: Tensor3,
        full: Option<(Tensor3, Tensor3, Tensor3)>, // (rho, rhop, mu)
    },
}
```

`Geometry` gains `pub exit: ExitSpec` and keeps `n_exit: C` **only** for the
isotropic variant (so old callers / pybind paths are untouched):

```rust
pub struct Geometry {
    pub wl_nm: f64,
    pub theta_in_rad: f64,
    pub n_entry: C,
    pub n_exit: C,              // used iff exit == Isotropic
    pub exit: ExitSpec,         // NEW (default: Isotropic(n_exit))
    pub exit_roughness: (i32, f64),
}
```

Exit construction in `solve_stack` — factor it into a helper so the
`clean` flag on a half-space can fail the solve (see §1.7):

```rust
// transfer.rs
use crate::berreman::LayerWaves;

/// Build the exit SLayer from an ExitSpec. Returns None when the exit
/// eigenbasis cannot be sorted cleanly (grazing/evanescent exit modes).
fn build_exit(exit: &ExitSpec, kx: f64) -> Option<SLayer> {
    let w: LayerWaves = match exit {
        ExitSpec::Isotropic(n) => halfspace_waves(*n * *n, kx),
        ExitSpec::Anisotropic { eps, full } => {
            // NOTE: a half-space needs sorted partial waves but NO propagation
            // (Q = 1 via SLayer::half_space). layer_waves_simple already
            // special-cases isotropic tensors back to the analytic basis, so
            // this is a strict generalization of the scalar path.
            match full {
                Some((rho, rhop, mu)) => layer_waves_full(eps, rho, rhop, mu, kx),
                None => layer_waves_simple(eps, kx),
            }
        }
    };
    if !w.clean {
        return None; // §1.7: unsortable exit basis fails the point (NaN row)
    }
    Some(SLayer::half_space(&w))
}
```

`Geometry` also gains a constructor so existing struct literals keep working
with a one-line change (`..Geometry::isotropic(...)` is avoided on purpose —
explicit fields stay greppable; instead update the 3 literal sites):

```rust
// transfer.rs
impl Geometry {
    /// Convenience: isotropic exit (today's behavior).
    pub fn isotropic(
        wl_nm: f64, theta_in_rad: f64,
        n_entry: C, n_exit: C,
        exit_roughness: (i32, f64),
    ) -> Geometry {
        Geometry {
            wl_nm, theta_in_rad, n_entry, n_exit,
            exit: ExitSpec::Isotropic(n_exit),
            exit_roughness,
        }
    }
}
```

Call sites to update (all become `Geometry::isotropic(...)` or add `exit:`):

- `transfer.rs::solve_stack` itself (rewritten per §1.4 below),
- `transfer.rs::tests::isotropic_slab_methods_agree_and_conserve_energy`,
- `examples/validate.rs` (2 `Geometry { ... }` literals),
- any Phase 2–4 code taking `&Geometry` (no change — field is internal).

New `solve_stack` head (replaces the isotropic-only block):

```rust
pub fn solve_stack(geom: &Geometry, layers: &[LayerSpec], method: Method) -> Option<SolveResult> {
    let k0 = 2.0 * std::f64::consts::PI / geom.wl_nm;
    let sin_in = geom.theta_in_rad.sin();
    let cos_in = geom.theta_in_rad.cos();
    let kx = geom.n_entry.re * sin_in;

    let kz_entry = geom.n_entry * c(cos_in, 0.0);
    // Scalar factor is only meaningful for isotropic exits; for anisotropic
    // exits it is still computed (harmless) but NOT used -- see power branch.
    let sin_out = (geom.n_entry / geom.n_exit) * c(sin_in, 0.0);
    let cos_out = (cone() - sin_out * sin_out).sqrt();
    let kz_exit = geom.n_exit * cos_out;
    let factor = (kz_exit / kz_entry).re;

    let entry_w = halfspace_waves(geom.n_entry * geom.n_entry, kx);
    if !entry_w.clean {
        return None;
    }
    let entry = SLayer::half_space(&entry_w);
    let exit = build_exit(&geom.exit, kx)?;   // NEW: anisotropic-capable

    // interior layers -- unchanged
    let mut slayers: Vec<SLayer> = Vec::with_capacity(layers.len());
    for ls in layers {
        let w = match &ls.full {
            Some((rho, rhop, mu)) => layer_waves_full(&ls.eps, rho, rhop, mu, kx),
            None => layer_waves_simple(&ls.eps, kx),
        };
        slayers.push(SLayer::from_waves(&w, k0, ls.thickness_nm));
    }
    // ... jones match unchanged, then power branch (§1.4) ...
}
```

`SLayer::half_space` already sets `q = [1;4]` and keeps `q_raw` — the
roughness dressing at the last interface therefore keeps working with the true
anisotropic `q_raw`. No change needed there.

### 1.4 Power for anisotropic exits (flux method)

The scalar `factor = (kz_exit/kz_entry).re` is meaningless with two different
transmitted eigenmodes. Use time-averaged Poynting flux per mode:

```rust
// berreman.rs (new helper, next to Wave::from_psi)
impl Wave {
    /// Time-averaged z-directed power flux of a unit-amplitude mode.
    /// Fields are in Berreman units (H impedance-scaled); the common scale
    /// cancels in ratios, so only Re(E × H*)_z matters.
    pub fn flux_z(ex: C, ey: C, hx: C, hy: C) -> f64 {
        0.5 * (ex * hy.conj() - ey * hx.conj()).re
    }
}
```

(`Wave::from_psi` already recovers `ez`, `hz`; expose `hx, hy` — currently
stored as `_hx/_hy`. Rename to `hx, hy` and fix the two use sites.)

Exit-side power algorithm — full implementation (new fns in `transfer.rs`;
amplitude helpers live in Phase 2's `fields.rs` and are imported here, so
**implement `fields::entry_amplitudes` + `fields::exit_amplitudes` first**):

```rust
// transfer.rs
use crate::fields::{entry_amplitudes, exit_amplitudes};

/// Which power formula produced SolveResult.t_power.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PowerMethod {
    /// T = Re(kz_exit/kz_entry)·|J|² (isotropic exits; bit-identical legacy).
    JonesFactor,
    /// Flux ratio over exit eigenmodes (anisotropic exits).
    Flux,
}

/// (Ex, Ey, Hx, Hy) of eigenmode column m of an SLayer P matrix.
/// ψ-column layout is [Ex, Hy, Ey, −Hx] — see berreman.rs::Wave::from_psi.
#[inline]
pub fn mode_eh4(p: &Mat4, m: usize) -> (C, C, C, C) {
    (p[0][m], p[2][m], -p[3][m], p[1][m])
}

/// Time-averaged z-flux of a superposition b0·mode0 + b1·mode1, with the
/// cross term (exit modes are generally NOT power-orthogonal):
///   F = ½·Re(Ex·Hy* − Ey·Hx*),  E = Σ bm·Em, H = Σ bm·Hm.
/// Building totals first (instead of F0/F1/Fx separately) is exact and short.
fn superposition_flux(p: &Mat4, b: &[C; 2]) -> f64 {
    let (ex0, ey0, hx0, hy0) = mode_eh4(p, 0);
    let (ex1, ey1, hx1, hy1) = mode_eh4(p, 1);
    let ex = b[0] * ex0 + b[1] * ex1;
    let ey = b[0] * ey0 + b[1] * ey1;
    let hx = b[0] * hx0 + b[1] * hx1;
    let hy = b[0] * hy0 + b[1] * hy1;
    0.5 * (ex * hy.conj() - ey * hx.conj()).re
}

/// Incident flux per canonical input: unit forward mode of the entry basis.
/// Columns 0 (p) and 1 (s) of entry.p at unit amplitude.
fn entry_flux(entry: &SLayer) -> Option<[f64; 2]> {
    let fp = superposition_flux(&entry.p, &[cone(), czero()]);
    let fs = superposition_flux(&entry.p, &[czero(), cone()]);
    if fp <= 1e-300 || fs <= 1e-300 {
        return None; // grazing/complex incidence: no well-defined incidence
    }
    Some([fp, fs])
}

/// Reflection power only (|J|²) — the T half of refl_trans_power is skipped
/// for anisotropic exits.
fn refl_power_only(j: &Jones) -> Mat2 {
    let mut r = [[czero(); 2]; 2];
    for i in 0..2 {
        for k in 0..2 {
            r[i][k] = c(j.refl[i][k].norm_sqr(), 0.0);
        }
    }
    r
}

/// Transmitted power per input polarization from exit-mode amplitudes.
/// `b_cols[j]` = [b0, b1] exit forward amplitudes for input pol j
/// (from fields::exit_amplitudes on the transfer matrix).
fn transmitted_power_flux(
    exit: &SLayer,
    b_cols: &[[C; 2]; 2],
    inc_flux: &[f64; 2],
) -> Option<Mat2> {
    let mut t = [[czero(); 2]; 2];
    for j in 0..2 {
        let f = superposition_flux(&exit.p, &b_cols[j]);
        t[j][j] = c(f / inc_flux[j], 0.0);
        // off-diagonals stay 0: with a mode-resolved exit basis there is no
        // (p,s)-projected cross power; document this shape choice in README.
    }
    Some(t)
}
```

Power branch at the end of `solve_stack` (replaces the unconditional
`refl_trans_power` call):

```rust
    let jones_circ = jones_to_circular(&jones);
    let (r_power, t_power, power_method) = match &geom.exit {
        ExitSpec::Isotropic(_) => {
            // Legacy path, untouched (G6 bit-identity).
            let (r, t) = refl_trans_power(&jones, factor);
            (r, t, PowerMethod::JonesFactor)
        }
        ExitSpec::Anisotropic(_) => {
            // Amplitude path needs the transfer matrix even for SM solves
            // (one extra sandwich per point; documented cost).
            let tm = transfer_matrix(&entry, &slayers, &exit)?;
            let a_p = exit_amplitudes(&tm, [cone(), czero()])?;
            let a_s = exit_amplitudes(&tm, [czero(), cone()])?;
            let inc = entry_flux(&entry)?;
            let b = [[a_p[0], a_p[1]], [a_s[0], a_s[1]]];
            let t = transmitted_power_flux(&exit, &b, &inc)?;
            (refl_power_only(&jones), t, PowerMethod::Flux)
        }
    };

    Some(SolveResult {
        jones,
        jones_circ,
        r_power,
        t_power,
        factor,
        power_method,   // NEW FIELD (see below)
    })
```

`SolveResult` gains exactly one field (keep `factor` for the isotropic case;
for `Flux` it still holds the scalar Snell ratio — documented as
*informational only* — OR set `f64::NAN`; decision: keep the computed value
and document, so `SolveResult` stays `Copy`-friendly and snapshots comparable):

```rust
pub struct SolveResult {
    pub jones: Jones,
    pub jones_circ: Jones,
    pub r_power: Mat2,
    pub t_power: Mat2,
    pub factor: f64,
    pub power_method: PowerMethod, // NEW
}
```

`Wave` field rename in `berreman.rs` (needed by flux-adjacent code and
Phase 2): `_hx/_hy` → `hx/hy` (public), `_sz` → `sz` (public, currently
unused — flux uses conjugated products instead; keep `sz` for diagnostics).
Fix the two construction/reader sites in `Wave::from_psi`, `cp_poynting` (none
read `_hx` today, so the rename is mechanical).

### 1.5 Python / pybind surface

pybind threading — `sweep` gains exit-tensor arrays. Empty slice = isotropic
(zero-copy legacy path); length `n_wl*18` = anisotropic. The `*_full` variant
additionally accepts per-wavelength exit MO tensors (all-or-none with
exit_eps):

```rust
// pybind.rs — signature deltas (show simple; full gains 4 more of the same)
#[pyfunction]
#[pyo3(signature = (wls, thetas, n_entry_re, n_entry_im, n_exit_re, n_exit_im,
                    eps, thicknesses, rough_types, rough_vals, method, exit_eps))]
pub fn solve_grid_simple(
    py: Python<'_>,
    /* ... unchanged ... */
    method: i32,
    exit_eps: PyReadonlyArray1<f64>,   // NEW: len 0 or n_wl*18
) -> PyResult<Py<PyDict>> {
    sweep(/* ..., */ method,
        exit_eps.as_slice()?, None, None, None)
}

#[pyfunction]
#[pyo3(signature = (wls, thetas, n_entry_re, n_entry_im, n_exit_re, n_exit_im,
                    eps, rho, rhop, mu, thicknesses, rough_types, rough_vals,
                    method, exit_eps, exit_rho, exit_rhop, exit_mu))]
pub fn solve_grid_full(/* ..., */
    exit_eps: PyReadonlyArray1<f64>,   // len 0 or n_wl*18
    exit_rho: PyReadonlyArray1<f64>,   // len 0 or n_wl*18
    exit_rhop: PyReadonlyArray1<f64>,  // len 0 or n_wl*18
    exit_mu: PyReadonlyArray1<f64>,    // len 0 or n_wl*18
) -> PyResult<Py<PyDict>> { /* ... */ }
```

Inside `sweep`, after the roughness-length validation:

```rust
    // NEW: decode exit spec. Length protocol (validated up front, before any
    // allocation, so Python gets ValueError not a panic):
    //   exit_eps len 0            -> isotropic for all wl
    //   exit_eps len n_wl*18      -> anisotropic (simple unless MO set given)
    //   exit_rho/rhop/mu: each len 0 or n_wl*18; all-or-none with exit_eps.
    let n_exit_want = n_wl * 18;
    for (name, s) in [("exit_eps", exit_eps.len()), ("exit_rho", exit_rho.len()),
                       ("exit_rhop", exit_rhop.len()), ("exit_mu", exit_mu.len())] {
        if s != 0 && s != n_exit_want {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} length {} != 0 or n_wl*18 = {}", s, n_exit_want)));
        }
    }
    let mo_given = !exit_rho.is_empty() || !exit_rhop.is_empty() || !exit_mu.is_empty();
    let mo_full = !exit_rho.is_empty() && !exit_rhop.is_empty() && !exit_mu.is_empty();
    if mo_given && !mo_full {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "exit_rho/rhop/mu must be given together (all len n_wl*18 or all empty)"));
    }
    if mo_full && exit_eps.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "exit MO tensors require exit_eps"));
    }
    // NOTE: sweep() is shared by both entry points; the SIMPLE caller passes
    // empty rho/rhop/mu slices for the stack (existing `None` args), so an MO
    // exit set on the simple path is a hard error (MO needs the full Δ).
    // Detect it: rho.is_none() && mo_full. The wrapper pre-validates too;
    // belt-and-braces here because FFI arity drift is silent (see Phase 4.3).
    // (sweep() signature gains `exit_eps/exit_rho/exit_rhop/exit_mu: &[f64]`
    // alongside the existing `rho/rhop/mu: Option<&[f64]>` stack slices.)
    if rho.is_none() && mo_full {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "exit MO tensors require the full solver (solve_grid_full)"));
    }
    let mo_slices: Option<(&[f64], &[f64], &[f64])> =
        if mo_full { Some((exit_rho, exit_rhop, exit_mu)) } else { None };
    // per-wavelength: build Vec<ExitSpec> of length n_wl
    let mut exits: Vec<ExitSpec> = Vec::with_capacity(n_wl);
    for w in 0..n_wl {
        if exit_eps.is_empty() {
            exits.push(ExitSpec::Isotropic(c(ne_re[w], ne_im[w])));
        } else {
            let eps_t = read_tensor(exit_eps, w * 18);
            let full = match mo_slices {
                Some((r, rp, m)) => Some((read_tensor(r, w*18), read_tensor(rp, w*18), read_tensor(m, w*18))),
                None => None,
            };
            exits.push(ExitSpec::Anisotropic { eps: eps_t, full });
        }
    }
    // ... in the Rayon closure: geom.exit = exits[w].clone() (ExitSpec: Clone).
```

`Geometry` construction per point becomes:

```rust
let geom = Geometry {
    wl_nm: wls[w],
    theta_in_rad: thetas[th],
    n_entry: c(ne_re[w], ne_im[w]),
    n_exit: c(nx_re[w], nx_im[w]),
    exit: exits[w].clone(),          // NEW
    exit_roughness,
};
```

Output dict gains `"power_method"` (int codes `0 = JonesFactor, 1 = Flux`):

```rust
    // in sweep(), next to n_failed:
    let pm: Vec<f64> = outs.iter().map(|o| if o.flux { 1.0 } else { 0.0 }).collect();
    // PointOut gains `flux: bool`; from_solve() takes power_method param.
    out.set_item("power_method", PyArray::from_vec(py, pm).reshape([n_wl, n_th])?)?;
```

Wrapper (`navette/berreman.py`) — full `__init__` / `solve` deltas:

```python
class BerremanStack:
    def __init__(self, layers, wavelengths_nm, angles, n_entry=1.0, n_exit=1.0,
                 method="scattering", angles_in_radians=False,
                 exit_roughness=None,
                 exit_eps=None, exit_rho=None, exit_rhop=None, exit_mu=None):  # NEW
        # ... existing body ...
        self.exit_eps = self._tensor_per_wl_opt(exit_eps, "exit_eps")    # (n_wl,3,3) or None
        self.exit_rho = self._tensor_per_wl_opt(exit_rho, "exit_rho")
        self.exit_rhop = self._tensor_per_wl_opt(exit_rhop, "exit_rhop")
        self.exit_mu = self._tensor_per_wl_opt(exit_mu, "exit_mu")
        mo_given = [x is not None for x in (exit_rho, exit_rhop, exit_mu)]
        if any(mo_given) and not all(mo_given):
            raise ValueError("exit_rho/rhop/mu must be given together")
        if exit_eps is None and any(mo_given):
            raise ValueError("exit MO tensors need exit_eps")

    def _tensor_per_wl_opt(self, t, name):
        """Like _tensor_per_wl but passes None through."""
        if t is None:
            return None
        return self._tensor_per_wl(np.asarray(t, dtype=complex), name)

    def _flatten_opt(self, per_wl):
        """Flatten an (n_wl,3,3) tensor field to [n_wl*18] (re,im) pairs."""
        out = np.empty(self.n_wl * 18, dtype=float)
        for w in range(self.n_wl):
            pair = np.empty(18, dtype=float)
            pair[0::2] = per_wl[w].real.reshape(9)
            pair[1::2] = per_wl[w].imag.reshape(9)
            out[w*18:(w+1)*18] = pair
        return out

    def solve(self):
        # ... existing thick/eps/rtypes marshalling ...
        flat = lambda a: self._flatten_opt(a) if a is not None else np.empty(0, dtype=float)
        e_eps, e_rho, e_rhop, e_mu = (flat(x) for x in
            (self.exit_eps, self.exit_rho, self.exit_rhop, self.exit_mu))
        if self._use_full:
            # ... existing mo marshalling ...
            raw = _rs_solve_full(*args, eps, rho, rhop, mu, thick,
                                  rtypes, rvals, self.method,
                                  e_eps, e_rho, e_rhop, e_mu)   # 4 NEW trailing args
        else:
            if e_rho.size or e_rhop.size or e_mu.size:
                raise ValueError("exit MO tensors force the full path; "
                                 "give any layer rho/rhop/mu or pass exit tensors only "
                                 "via the full entry point")
            raw = _rs_solve_simple(*args, eps, thick, rtypes, rvals,
                                    self.method, e_eps)          # 1 NEW trailing arg
```

`_assemble` maps the int codes (squeezing like the other fields):

```python
        pm = np.asarray(raw["power_method"])
        # squeeze singleton axes with the same rules, then:
        res["power_method"] = np.where(pm == 1, "flux", "jones_factor")
```

Back-compat note: existing positional callers pass `method` last today; the
new trailing params are keyword-or-positional additions — old positional calls
still bind correctly. Document `exit_eps` + `n_exit` mutual exclusion:
`n_exit` non-unity *and* `exit_eps` given → `ValueError` (avoid silent
precedence bugs; state this in README).

### 1.6 Tests & gates (new gate G7)

- **G7a bit-identity** (`transfer.rs::tests`, pure Rust):

```rust
#[test]
fn aniso_isotropic_exit_is_bit_identical() {
    // each exit tensor PAIRED with its matching scalar (same medium!) —
    // the first draft looped both tensors against n_exit = 1.5 (wrong).
    let exits = [
        (c(1.0, 0.0), diag_eps(c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0))),
        (c(1.5, 0.0), diag_eps(c(2.25, 0.0), c(2.25, 0.0), c(2.25, 0.0))),
    ];
    for (n_exit, eps) in exits {
        for method in [Method::Scattering, Method::Transfer] {
            let g_iso = Geometry::isotropic(550.0, 0.35, c(1.0, 0.0), n_exit, (0, 0.0));
            let mut g_an = Geometry::isotropic(550.0, 0.35, c(1.0, 0.0), n_exit, (0, 0.0));
            g_an.exit = ExitSpec::Anisotropic { eps, full: None };
            let layers = [LayerSpec {
                eps: diag_eps(c(2.25, 0.0), c(2.89, 0.0), c(2.25, 0.0)),
                thickness_nm: 250.0, full: None, front_roughness: (0, 0.0),
            }];
            let a = solve_stack(&g_iso, &layers, method).unwrap();
            let b = solve_stack(&g_an, &layers, method).unwrap();
            assert_eq!(b.power_method, PowerMethod::Flux); // aniso *route* ...
            // ...but on an isotropic tensor the basis shortcut must give
            // bit-identical numbers (this is the actual assertion):
            for i in 0..2 { for j in 0..2 {
                assert_eq!(a.jones.refl[i][j], b.jones.refl[i][j]);
                assert_eq!(a.r_power[i][j], b.r_power[i][j]);
            }}
        }
    }
}
```

  Expected `dR = dT = 0.0` via the `is_isotropic` shortcut. (If an
  implementation detail ever breaks exact equality, relax to ≤ 1e-15 but file
  it as a regression — equality is by construction, not luck.)
- **G7b vs B44 anisotropic back** (`tests/test_aniso_exit.py`): slab
  `rot_z(diag(2.25,2.89,2.25), 35°)` 250 nm exiting into half-space
  `rot_z(diag(2.1,2.7,2.1), 50°)`; angles 0/20/40° × SM/TM; Jones positionally
  + R power vs `Berreman4x4.Structure.getJones` ≤ 1e-9 (B44's Padé-7
  propagator is the limiting factor). AS BUILT (two corrections):
  (a) only Jr+R compare directly — B44's T_ti lives in its own
  exit-eigenvector normalization ("Ey = c1 + c2" rescale in
  `HalfSpace.getTransitionMatrix`), so raw Jt differs by a basis change
  (|dT| ~ 1.4 direct). Transmission is compared as POWER via B44-side flux
  (Lb/Lf-reconstructed ψ, F = ½Re(ψ0ψ3* − ψ1ψ2*) in B44's [Ex,Ey,Hx,Hy]
  basis — read off `buildDeltaMatrix`). OBSERVED: Jr+R 7.3e-16,
  T-flux 1.4e-15. The two engines' fully independent flux computations
  agreeing to 1.4e-15 validates our exit amplitudes end-to-end.
  (b) the draft G7a exit list paired a vacuum tensor against a 1.5 scalar
  (different media!) — as built, each tensor is paired with its matching
  scalar (1.0↔diag(1.0), 1.5↔diag(2.25)); bit-identity holds (0.0).
- **G7c flux bridge** (Rust test): isotropic exit, `transmitted_power_flux`
  (with `exit_amplitudes` from the TM matrix) vs `refl_trans_power` scalar T
  ≤ 1e-12 on all canonical cases — proves the flux machinery before trusting
  it on anisotropic exits.
- **G7d energy** (Python test): lossless anisotropic-exit stack, per-input
  `R_col + T_col` each in `[-1e-9, 1+1e-9]` (not exactly 1 — different exit
  DOS), no NaN, `n_failed == 0`, `power_method == "flux"` everywhere.

### 1.7 Risks

- Exit eigen-sort at grazing `Kx` (evanescent exit modes): `sort_partial_waves`
  fallback may trigger; treat `clean == false` on a half-space as solve
  failure (`None`) rather than garbage — matches existing philosophy.
- Cross-flux term sign errors: mitigate with G7c bridge + G7d bounds.
---

## Phase 2 — Internal fields E(z) / H(z) (new `src/fields.rs`)

### 2.1 Why / upstream reference

- Live pyllama: `Model.get_in_plane_fields(input_vector, x_array, direction, n_z)`,
  helper `propagate_eigenmodes(cur_e_vec, z_propagator_matrix, cur_layer)`
  (`BerreMueller/src/berremueller/pyllama.py` ≈ ll. 2039–2125); plotting in
  `field_plotting.py`. Berreman4x4: `Layer.getPropagationMatrix` per slice +
  `Evaluation` records (field reconstruction left to user scripts).
- navette has **no** near-field output — only far-field Jones/R/T/Mueller.
  This is the biggest physics gap for cavity/LC designers (standing-wave
  profiles, Purcell overlap integrals, absorption-density profiles).
- Depends on Phase 1's amplitude helper (§1.4 step 2) — implement that helper
  here, in `fields.rs`, and re-export for Phase 1.
- AS BUILT: the helper landed EARLY (Phase 1 created `fields.rs` with
  `entry_amplitudes` + `exit_amplitudes` for the Flux arm), so §2.3 was
  already done and G8a pre-closed by P1's bridge test. Phase 2 consumed
  P4's `build_head`/`build_slayers` (the §2.5-sketch `build_point` refactor
  target — dependency flip as the P4 notes predicted).

### 2.2 Design overview

New module `src/fields.rs`, registered in `lib.rs`:

```rust
// lib.rs
pub mod fields;
```

```rust
// fields.rs
use crate::berreman::Tensor3;
use crate::cmatrix::{c, cone, czero, mat4_inv, mat4_mul, Mat2, Mat4, C};
use crate::transfer::{LayerSpec, Method, SLayer, Geometry};
use num_complex::ComplexFloat;

/// Full 6-vector field at one depth for one input polarization.
#[derive(Clone, Copy)]
pub struct FieldVec {
    pub e: [C; 3],
    pub h: [C; 3],
}

/// One depth sample: global z (nm, 0 = entry plane) + fields per input pol.
/// `per_input[0]` = response to p-incidence `[1,0]`, `per_input[1]` to `[0,1]`.
pub struct FieldPoint {
    pub z_nm: f64,
    pub per_input: [FieldVec; 2],
}
```

### 2.3 Entry amplitude solve (shared with Phase 1)

Given the full transfer matrix `M` (entry-plane amplitudes → exit-plane
amplitudes) and incident `i = [i0, i1]`, the reflected pair `(a2, a3)` follows
from the outgoing condition `b2 = b3 = 0`:

```rust
/// Solve entry-plane amplitude vector a = [i0, i1, a2, a3] from transfer
/// matrix M and incident pair. Returns None if the 2x2 system is singular.
pub fn entry_amplitudes(tm: &Mat4, incident: [C; 2]) -> Option<[C; 4]> {
    // rows 2,3: tm[r][0]*i0 + tm[r][1]*i1 + tm[r][2]*a2 + tm[r][3]*a3 = 0
    let d = tm[2][2] * tm[3][3] - tm[2][3] * tm[3][2];
    if d.norm() < 1e-300 {
        return None;
    }
    let di = d.recip();
    let rhs2 = -(tm[2][0] * incident[0] + tm[2][1] * incident[1]);
    let rhs3 = -(tm[3][0] * incident[0] + tm[3][1] * incident[1]);
    let a2 = (rhs2 * tm[3][3] - rhs3 * tm[2][3]) * di;
    let a3 = (tm[2][2] * rhs3 - tm[3][2] * rhs2) * di;
    Some([incident[0], incident[1], a2, a3])
}
```

Check against `fresnel_from_transfer`: with `incident = [1,0]`,
`a2/a3` must equal `(r_pp, r_ps)` — add a unit test asserting exactly that
(bridges new code to the validated path).

> Scattering-method note: internal amplitudes are extracted via the **transfer**
> route even when the far field was solved by SM (same matrices, same `P/Q`).
> Document that `fields()` always propagates with transfer matrices; SM stays
> the far-field default. (Matches pyllama, whose `get_in_plane_fields` likewise
> propagates eigenmodes directly.)

### 2.4 Depth propagation

```rust
// fields.rs — frame cache (built once per (λ,θ), reused for all z + both pols)
use crate::cmatrix::mat4_identity;

/// Per-layer transfer matrices, inverses, prefix products, depth bookkeeping.
struct StackFrames {
    /// T_j = P_j · Q_j · P_j⁻¹ for each interior layer.
    layer_tm: Vec<Mat4>,
    /// P_j⁻¹ for each interior layer (local propagation without re-inversion).
    layer_pinv: Vec<Mat4>,
    /// Prefix products: pref[j] = T_{j-1}·…·T_0 (pref[0] = I), so that
    /// ψ at front of layer j = pref[j] · ψ0. Length n_layers + 1; pref[n]
    /// pushes to the exit plane (used for the b-consistency check in tests).
    pref: Vec<Mat4>,
    /// Front depth of layer j in nm (layer 0 starts at 0). Length n_layers.
    fronts: Vec<f64>,
    total: f64,
}

fn build_frames(slayers: &[SLayer], thick: &[f64]) -> Option<StackFrames> {
    let n = slayers.len();
    let mut layer_tm = Vec::with_capacity(n);
    let mut layer_pinv = Vec::with_capacity(n);
    let mut pref = Vec::with_capacity(n + 1);
    pref.push(mat4_identity());
    let mut fronts = Vec::with_capacity(n);
    let mut z = 0.0;
    for (sl, &d) in slayers.iter().zip(thick.iter()) {
        fronts.push(z);
        z += d;
        let pinv = mat4_inv(&sl.p)?;
        // T_j = P·diag(Q)·P⁻¹ == mat4_similarity_diag (same helper as TM path)
        let tj = crate::cmatrix::mat4_similarity_diag(&sl.p, &sl.q)?;
        let acc = mat4_mul(&tj, &pref[pref.len() - 1]);
        pref.push(acc);
        layer_tm.push(tj);
        layer_pinv.push(pinv);
    }
    Some(StackFrames { layer_tm, layer_pinv, pref, fronts, total: z })
}

/// ψ-column times eigen-coefficients: ψ = P · c (c = per-mode amplitudes).
fn apply_p(p: &Mat4, coeff: &[C; 4]) -> [C; 4] {
    let mut out = [czero(); 4];
    for i in 0..4 {
        let mut s = czero();
        for k in 0..4 {
            s += p[i][k] * coeff[k];
        }
        out[i] = s;
    }
    out
}

/// Local propagation inside one layer/half-space over distance d (nm):
/// c_k(d) = c_k(0)·exp(i·k0·q_raw[k]·d), then ψ = P·c.
/// This is EXACTLY propagation_diag + P-application, factored for reuse.
fn propagate_local(sl: &SLayer, coeff_front: &[C; 4], k0: f64, d: f64) -> [C; 4] {
    let f = c(0.0, k0 * d);
    let mut cd = [czero(); 4];
    for k in 0..4 {
        cd[k] = coeff_front[k] * (f * sl.q_raw[k]).exp();
    }
    apply_p(&sl.p, &cd)
}
```

Field at global depth `z` (full decision tree — entry, interior, exit):

```rust
/// Resolve which region `z` falls in. Returns an enum so the main loop stays
/// readable; half-spaces are handled with the same propagate_local machinery.
enum Region { Entry(f64), Layer(usize, f64), Exit(f64) }
// Entry(d): d = -z > 0 distance back from the entry plane.
// Layer(j, d): d = z - fronts[j], 0 <= d < thick[j].
// Exit(d): d = z - total >= 0 distance past the last interface.

fn locate(frames: &StackFrames, thick: &[f64], z: f64) -> Region {
    if z < 0.0 {
        return Region::Entry(-z);
    }
    if z >= frames.total {
        return Region::Exit(z - frames.total);
    }
    // linear scan is fine (n_layers small); binary search if profiling says so
    let mut j = 0;
    while j + 1 < frames.fronts.len() && frames.fronts[j + 1] <= z {
        j += 1;
    }
    Region::Layer(j, z - frames.fronts[j])
}
```

Unpack 6-vector (Berreman units — AS BUILT via `Wave::from_psi`, sync BY
CONSTRUCTION not by test; `Wave::_ez/_hz` renamed public `ez/hz` for this.
A `psi_unpack_order` test pins the E=[Ex,Ey,Ez]/H=[Hx,Hy,Hz] mapping once):

```rust
/// ψ = [Ex, Hy, Ey, −Hx] (+ layer eps, kx) -> full 6-vector (Berreman units).
/// Ez from ∇·D = 0 (no free charges); Hz = kx·Ey (ky = 0 plane of incidence).
pub fn psi_to_eh(psi: &[C; 4], eps: &Tensor3, kx: f64) -> FieldVec {
    let ex = psi[0];
    let hy = psi[1];
    let ey = psi[2];
    let hx = -psi[3];
    let inv22 = eps[2][2].recip();
    let kxc = c(kx, 0.0);
    let ez = -(eps[2][0] * inv22) * ex - (eps[2][1] * inv22) * ey - (kxc * inv22) * hy;
    let hz = kxc * ey;
    FieldVec { e: [ex, ey, ez], h: [hx, hy, hz] }
}
```

Half-space treatment (unified — no special-case plane-wave code):

- **Entry (`z < 0`):** coefficients `a` from `entry_amplitudes` (§2.3) are
  defined AT the entry plane. Propagate backwards: `propagate_local(&entry,
  &a, k0, -d)` with `d = -z`. The diagonal phases handle forward and backward
  modes symmetrically (negative distance), so incident + reflected interfere
  into the correct standing wave. `eps` for unpacking = entry `n²·I`.
- **Exit (`z ≥ total`):** coefficients `b = M·a` (i.e. `exit_amplitudes`)
  defined at the exit plane; propagate forward by `d = z − total` with the
  exit `SLayer`. `eps` = exit tensor (isotropic `n²·I` or Phase-1 anisotropic
  tensor — thread it through: `fields_at_depths` needs the exit eps, so take
  `exit_eps: &ExitEps` where `enum ExitEps { Iso(C), Aniso(Tensor3) }`, or
  simply reuse `geom.exit: &ExitSpec` — it already carries both. Use that.).
- **Interior:** `ψ_front = pref[j] · ψ0` with `ψ0 = entry.P · a`; local
  coefficients `c_front = P_j⁻¹ · ψ_front` (via cached `layer_pinv[j]`);
  `ψ(z) = propagate_local(&slayers[j], &c_front, k0, d)`; unpack with
  `layers[j].eps` (the per-wavelength tensor from `LayerSpec` — dispersive
  layers automatically correct since `pybind::sweep` resolves `LayerSpec`
  per λ; fields take the same `&[LayerSpec]` — no new marshalling).

MO layers: unpacking uses `eps` only (Ez from ∇·D = 0 with the electric
tensor; H from ψ directly) — valid with `rho/mu` present because ψ already
encodes the full solution. Document this explicitly.

### 2.5 Public Rust API

```rust
/// Per-point scratch: everything invariant over z for one (λ,θ).
struct PointFrames {
    entry: SLayer,
    exit: SLayer,
    slayers: Vec<SLayer>,
    frames: StackFrames,
    tm: Mat4,          // full transfer matrix (entry -> exit amplitudes)
    kx: f64,
    k0: f64,
}

/// Compute fields at `z_nm` (any order; clamped handling per locate()) for
/// both canonical input polarizations. Returns None if any inversion fails.
pub fn fields_at_depths(
    geom: &Geometry,
    layers: &[LayerSpec],
    z_nm: &[f64],
    _method: Method, // accepted for API symmetry; propagation is transfer-based (§2.3)
) -> Option<Vec<FieldPoint>> {
    let k0 = 2.0 * std::f64::consts::PI / geom.wl_nm;
    let kx = geom.n_entry.re * geom.theta_in_rad.sin();
    // NOTE: replicates the solve_stack head (entry/exit/slayers). Refactor
    // target: extract `fn build_point(geom, layers) -> Option<PointFrames>`
    // shared by solve_stack + fields_at_depths (do it in this phase; G1-G6
    // must stay green — the refactor is behavior-preserving by construction).
    let entry = SLayer::half_space(&halfspace_waves(geom.n_entry * geom.n_entry, kx));
    let exit = build_exit(&geom.exit, kx)?; // Phase-1 helper (crate::transfer — make pub(crate))
    let mut slayers = Vec::with_capacity(layers.len());
    let mut thick = Vec::with_capacity(layers.len());
    for ls in layers {
        let w = match &ls.full {
            Some((rho, rhop, mu)) => layer_waves_full(&ls.eps, rho, rhop, mu, kx),
            None => layer_waves_simple(&ls.eps, kx),
        };
        slayers.push(SLayer::from_waves(&w, k0, ls.thickness_nm));
        thick.push(ls.thickness_nm);
    }
    let frames = build_frames(&slayers, &thick)?;
    let tm = crate::transfer::transfer_matrix(&entry, &slayers, &exit)?;
    let psi0 = |a: &[C; 4]| apply_p(&entry.p, a); // entry-plane state

    let mut out = Vec::with_capacity(z_nm.len());
    for &z in z_nm {
        let mut per_input = [FieldVec { e: [czero(); 3], h: [czero(); 3] }; 2];
        for (ji, incident) in [[cone(), czero()], [czero(), cone()]].iter().enumerate() {
            let a = entry_amplitudes(&tm, *incident)?;
            let b = exit_amplitudes(&tm, *incident)?; // b = tm·a, b[2..]≈0
            let psi = match locate(&frames, &thick, z) {
                Region::Entry(d) => propagate_local(&entry, &a, k0, -d),
                Region::Layer(j, d) => {
                    let psi0v = psi0(&a);
                    // ψ_front = pref[j]·ψ0, then local coeffs via cached P⁻¹
                    let mut front = [czero(); 4];
                    for i in 0..4 {
                        let mut s = czero();
                        for k in 0..4 { s += frames.pref[j][i][k] * psi0v[k]; }
                        front[i] = s;
                    }
                    let mut cf = [czero(); 4];
                    for i in 0..4 {
                        let mut s = czero();
                        for k in 0..4 { s += frames.layer_pinv[j][i][k] * front[k]; }
                        cf[i] = s;
                    }
                    propagate_local(&slayers[j], &cf, k0, d)
                }
                Region::Exit(d) => propagate_local(&exit, &b, k0, d),
            };
            let eps_here: &Tensor3 = match locate(&frames, &thick, z) {
                Region::Entry(_) => &ENTRY_EPS_TMP, // see note below
                Region::Layer(j, _) => &layers[j].eps,
                Region::Exit(_) => &EXIT_EPS_TMP,
            };
            per_input[ji] = psi_to_eh(&psi, eps_here, kx);
        }
        out.push(FieldPoint { z_nm: z, per_input });
    }
    Some(out)
}
```

Implementation notes (do not skip):

- `ENTRY_EPS_TMP`/`EXIT_EPS_TMP` above are placeholders for real locals:
  build `entry_eps = n_entry²·I` once, and match `geom.exit` for the exit
  tensor (`Isotropic(n) → n²·I`, `Anisotropic{eps,..} → eps`). Calling
  `locate` twice per (z, pol) is wasteful — hoist one `locate` per z outside
  the pol loop in the final code (sketch duplicates it for readability).
- `exit_amplitudes` (in `fields.rs` next to `entry_amplitudes`):

```rust
/// Exit-plane amplitudes b = M·a (b[0..2] = forward modes, b[2..4] ≈ 0).
pub fn exit_amplitudes(tm: &Mat4, incident: [C; 2]) -> Option<[C; 4]> {
    let a = entry_amplitudes(tm, incident)?;
    let mut b = [czero(); 4];
    for i in 0..4 {
        let mut s = czero();
        for k in 0..4 {
            s += tm[i][k] * a[k];
        }
        b[i] = s;
    }
    Some(b)
}
```

AS BUILT (Phase 1, authoritative — the sketch above is tombstoned):
`exit_amplitudes` returns `[C; 2]` (forward pair ONLY). The Exit branch of
`fields_at_depths` pads `[b0, b1, 0, 0]` — zeroing, not propagating, the
~1e-16 backward residuals (a 1e-16 backward mode could be evanescent-growing
in +z; zero IS the outgoing boundary condition).

- Signature note: `_method` is intentionally unused in propagation (see §2.3);
  keep the slot so the Python `method=` kwarg stays meaningful (it still
  selects the far-field engine whose Jones must match the field asymptotes —
  asserted in G8c).
- Periods (P4-postdates-draft decision): the WRAPPER expands the cell
  (`self.layers * self.periods`) and calls the Rust engine as a plain stack —
  z spans the unfolded 0..N·cell thickness, expansion agreement 0.0 (G8
  periods arm). No powering in Rust: prefix products need the unfolded
  sequence anyway, and expansion keeps one code path (verified, not trusted).
- Roughness: `fields()` REJECTS Route-1 specs with ValueError → graded_stack.
  Route-1 dressing is a far-field power redistribution with no local-field
  meaning; the graded expansion is the physically meaningful field picture.
  (Draft was silent here — decided at implementation.)
- Absorption-density helper (for G8e), same module (WITH the /c0 correction
  below — the sketch's plain ½·ω·s is tombstoned):

```rust
/// Time-averaged absorbed power density at a field point (W/m³ up to the
/// Berreman-unit scale — ratios/integrals only):
///   A = ½·ω·Σ_ij Im(ε_ij)·Re(E_i·conj(E_j)),  ω = 2πc/λ.
/// For diagonal lossless eps this is exactly 0; integrated over z it must
/// equal 1 − R − T per input pol (G8e). Off-diagonal Im parts included via
/// the full double sum (correct for gyrotropic media).
pub fn absorption_density(f: &FieldVec, eps: &Tensor3, wl_nm: f64) -> f64 {
    let omega = 2.0 * std::f64::consts::PI * 2.998e8 / (wl_nm * 1e-9);
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += eps[i][j].im * (f.e[i] * f.e[j].conj()).re;
        }
    }
    0.5 * omega * s
}
```

DRAFT CORRECTION (implementation finding — the sketch above is off by c0):
code-unit flux uses impedance-scaled H (H_berr = Z0·H_SI), so densities must
be Z0× SI: A = ½·(ω/c0)·ΣIm(ε)Re(E·conj E) (ε0·Z0 = 1/c0). As built in
`fields.rs::absorption_density` with `const C0`. G8e caught the sketch:
2.86e8 vs 1.0; corrected balances 1 ± 2.2e-7 — the prefactor is PINNED by the
independent R/T conservation identity, not fitted.

### 2.6 pybind + Python surface

pybind design — mirror `sweep` but with a depth axis. Parallelize over
`(w, th)` points (each computes all `n_z` depths; prefix products reused —
cache-friendly). Per-point payload is fixed-size only if `n_z` is capped;
instead have each Rayon task return a `Vec<f64>` flat buffer and concatenate
in order (tasks are indexed, so ordering is deterministic):

```rust
// pybind.rs
use crate::fields::fields_at_depths;

/// Shared fields sweep. `have_full` selects LayerSpec.full decoding, exactly
/// like sweep()'s rho/rhop/mu Option handling. Returns dict with:
///   "E_p","H_p","E_s","H_s": [n_wl, n_th, n_z, 3] complex (as re/im pairs
///       "E_p_re", "E_p_im", ... — 8 arrays, same put-pattern as sweep),
///   "z_nm": [n_z] float echo, "n_failed": int.
#[allow(clippy::too_many_arguments)]
fn fields_sweep(
    py: Python<'_>,
    wls: &[f64], thetas: &[f64],
    ne_re: &[f64], ne_im: &[f64], nx_re: &[f64], nx_im: &[f64],
    eps: &[f64], rho: Option<&[f64]>, rhop: Option<&[f64]>, mu: Option<&[f64]>,
    thick: &[f64], z_nm: &[f64],
    exit_eps: &[f64], method_code: i32,
) -> PyResult<Py<PyDict>> {
    let n_wl = wls.len(); let n_th = thetas.len();
    let n_layers = thick.len(); let n_z = z_nm.len();
    if n_z == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("z_nm must be non-empty"));
    }
    let method = if method_code == 1 { Method::Transfer } else { Method::Scattering };
    // Reuse sweep()'s stack pre-build (refactor: extract
    // `fn build_stacks(...) -> Vec<Vec<LayerSpec>>` shared by both sweeps)
    // plus per-wavelength ExitSpec vec (Phase-1 decoding, shared helper
    // `fn build_exits(...) -> PyResult<Vec<ExitSpec>>`).
    // ...
    let total = n_wl * n_th;
    // Each task -> flat [n_z * 3 * 2(pol) * 2(E/H) * 2(re/im)] = n_z*24 f64
    // + ok flag. Collect indexed, then scatter into 8 output buffers.
    let outs: Vec<Option<Vec<f64>>> = py.allow_threads(|| {
        (0..total).into_par_iter().map(|k| {
            let w = k / n_th; let th = k % n_th;
            let geom = Geometry { /* ... as in sweep, + exit: exits[w].clone() */ };
            let pts = fields_at_depths(&geom, &stacks[w], z_nm, method)?;
            let mut buf = vec![0.0; n_z * 24];
            for (iz, p) in pts.iter().enumerate() {
                for (ji, fv) in p.per_input.iter().enumerate() {
                    for c in 0..3 {
                        // layout: (((pol*2 + eh) * n_z + iz) * 3 + c) * 2 + (re/im)
                        let base = ((ji * 2 * n_z + iz) * 3 + c) * 2;
                        // E=eh0, H=eh1 handled by two writes; sketch abbreviates
                        buf[base] = fv.e[c].re; buf[base + 1] = fv.e[c].im;
                    }
                }
                // (H written to second half of the per-task buffer; same pattern)
            }
            Some(buf)
        }).collect()
    });
    // scatter outs[k] into E_p_re [n_wl,n_th,n_z,3] etc. via put-pattern;
    // None -> NaN row + n_failed++ (same convention as sweep).
    // ...
}

#[pyfunction]
#[pyo3(name = "fields_grid_simple")]
#[pyo3(signature = (wls, thetas, n_entry_re, n_entry_im, n_exit_re, n_exit_im,
                    eps, thicknesses, z_nm, exit_eps, method))]
pub fn fields_grid_simple(/* typed arrays as in solve_grid_simple */) -> PyResult<Py<PyDict>> {
    fields_sweep(/* ..., */ None, None, None, /* ... */)
}
// + fields_grid_full with rho/rhop/mu AND exit MO arrays (mirror solve_grid_full).
```

Buffer-layout rule (write it in the doc comment AND the wrapper): within one
`(w, th)` task buffer, index `((pol * 2 + eh) * n_z + iz) * 3 + comp`, times 2
for re/im, where `pol: p=0,s=1`, `eh: E=0,H=1`. Keep a `#[cfg(test)]`
layout test in pybind? pybind is feature-gated; instead put a pure-Rust
layout unit test on a helper `fn field_buf_index(pol, eh, iz, comp, n_z)` in
`fields.rs` (testable without Python).

AS BUILT: exactly as sketched (`field_buf_index` + `buf_layout_bijective`
bijectivity test), PLUS the sketched `sweep()` internal decode was extracted
into shared `decode_stacks`/`decode_exits` helpers used by BOTH sweeps — the
full G1–G14 suite staying green is the behavior-preservation proof. The
sketched `_assemble` squeeze loop became `_squeeze(res)` shared by solve()
and fields() (z never squeezed: only the two leading axes are touched).

Wrapper (`navette/berreman.py`):

```python
class BerremanStack:
    def fields(self, z_nm, method=None):
        """Near fields at depths z_nm (nm, 0 = entry plane; negatives allowed).
        Returns dict E_p/H_p/E_s/H_s: complex [n_wl, n_th, n_z, 3] + "z_nm".
        method defaults to self.method (far-field engine for asymptote checks)."""
        z = np.atleast_1d(np.asarray(z_nm, dtype=float))
        thick = np.asarray([l.thickness_nm for l in self.layers], dtype=float)
        eps = self._flatten(lambda l: l.eps)
        e_eps = self._flatten_opt(self.exit_eps) if self.exit_eps is not None else np.empty(0)
        m = self._METHOD[(method or ("transfer" if self.method == 1 else "scattering")).lower()]
        args = (self.wl, self.theta, self.n_entry.real.copy(), self.n_entry.imag.copy(),
                self.n_exit.real.copy(), self.n_exit.imag.copy())
        if self._use_full:
            # ... marshal rho/rhop/mu + exit MO exactly like solve() ...
            raw = _rs_fields_full(*args, eps, rho, rhop, mu, thick, z, e_eps, ..., m)
        else:
            raw = _rs_fields_simple(*args, eps, thick, z, e_eps, m)
        def cx(re, im):
            return np.asarray(raw[re]) + 1j * np.asarray(raw[im])
        res = {"E_p": cx("E_p_re", "E_p_im"), "H_p": cx("H_p_re", "H_p_im"),
               "E_s": cx("E_s_re", "E_s_im"), "H_s": cx("H_s_re", "H_s_im"),
               "z_nm": np.asarray(raw["z_nm"]), "n_failed": int(raw["n_failed"])}
        # squeeze singleton wl/angle axes exactly like _assemble (share a helper:
        # refactor _assemble's squeeze loop into _squeeze(res) and reuse here;
        # do NOT squeeze z (n_z == 1 stays).
        return self._squeeze(res, keep_last_dims=2)
```

### 2.7 Tests & gates (new gate G8)

- **G8a amplitude bridge:** `entry_amplitudes(TM,[1,0])[2..] == J_refl[:,0]`
  (and `[0,1]` ↔ column 1) to 1e-15 on all canonical cases. OBSERVED: green
  (pre-closed by P1; P2 added the explicit vs-`solve_stack` test).
- **G8b interface continuity:** tangential `(Ex, Ey, Hx, Hy)` continuous across
  every interface to ≤ 1e-9 (twisted-nematic 20-slice case included).
  OBSERVED: 4.9e-11 (slab + twist20 + MO-full + aniso-exit, ±1e-9 nm straddle).
- **G8c far-field asymptote:** fields at `z → −∞` decompose to incident +
  `J_refl`; at `+∞` to `J_trans` (≤ 1e-9, lossless slab). AS BUILT (three
  basis-free checks, no P⁻¹ needed in Python): exit |Ex|,|Ey| constancy
  1.1e-16 (pure-forward proof); flux conservation 2.7e-16; entry fringe
  contrast == |r| 2.9e-7 (scan-resolution limited, budget 1e-6).
- **G8d vs live pyllama** `get_in_plane_fields` on 2 cases (after mapping
  conventions: live returns in-plane E_x,E_y per its `input_vector`; compare
  those components up to the documented global phase) ≤ 1e-6. OBSERVED:
  3.9e-8 DIRECT (no phase alignment needed; live is complex64, x_array=[0]).
  Live raises for N_periods > 1 — consistent with our wrapper-expansion choice.
- **G8e absorption density (optional):** `A(z) ∝ ω·Im(ε)|E|²` integrated over z
  equals `1 − R − T` per input pol (≤ 1e-6, absorbing case) — strong end-to-end
  physics check; implement `absorption_profile` helper in the same module.
  OBSERVED: 2.1e-7 (2001-pt trapezoid; self-normalized
  (A_int+F_out)/F_in == 1 AND A_int/F_in == (1−R−T)/(1−R) — zero convention
  dependence; the numpy profile independently re-implements the Rust formula,
  so agreement pins BOTH). NOT optional anymore — gating.
- **G8f expansion contract (new):** periods=10 fields == layers*10 fields
  0.0 exactly (same code path post-expansion); rough/empty-z/bad-method
  validation errors pinned.

### 2.8 Risks / effort

- Effort L (new module + bindings + 5 tests), no changes to validated paths.
- Evanescent-mode overflow in thick layers at large `d`: `exp(i·k0·q·d)` with
  `Im(q)·d ≫ 1` overflows to inf → propagate **backwards from the next
  interface** for decaying modes? v1: document the limit, clamp test depths;
  full fix (adjoint propagation) is a stated follow-up, not a blocker —
  far-field SM stays exact regardless.
---

## Phase 3 — EM exponential-matrix method (new `src/expm.rs`)

### 3.1 Why / upstream reference

Live pyllama offers **three** engines: `"SM"` scattering, `"TM"` transfer via
eigendecomposition, `"EM"` transfer via direct matrix exponential
(`Structure.build_exponential_matrix`, `_get_fresnel_EM`, `_get_refl_trans_EM`
in `BerreMueller/src/berremueller/pyllama.py` ≈ ll. 1197–1225, 1500–1530).
Crucially, `_get_fresnel_EM` uses **the identical extraction formula** as TM
(same `deno`, same `r_/t_` combinations) — only the layer matrix differs
(`expm(i·k0·d·Δ)` instead of `P·Q·P⁻¹`). Berreman4x4 is *entirely* built this
way (`HomogeneousLayer.getPropagationMatrix` → `hs_propagator_*` Padé/Taylor/
linear; §"linear" ≈ first-order, "Padé" default order 2, "Taylor" order 5).

Value for navette: (a) parity with both references' full method set,
(b) independent cross-check of our eigendecomposition (EM shares no code with
`eig4`), (c) better conditioning than TM for moderately thick homogeneous
layers when combined with scaling-and-squaring.

### 3.2 New module `src/expm.rs`

Pure-Rust complex 4×4 exponential by **scaling-and-squaring + Taylor**
(chosen over Padé so every coefficient is trivially auditable; upgrade path
noted in §3.5):

```rust
//! Complex 4x4 matrix exponential for the EM transfer path.
//! Method: scale A -> A/2^s so ||.||_1 <= 0.5, Taylor to fixed order,
//! then square s times. No LAPACK, no allocation beyond fixed arrays.

use crate::cmatrix::{mat4_identity, mat4_mul, mat4_zero, Mat4, C};
use num_complex::ComplexFloat;

/// 1-norm (max column absolute sum). Subordinate => ||AB|| <= ||A||·||B||.
pub fn mat4_norm1(a: &Mat4) -> f64 {
    let mut m = 0.0;
    for j in 0..4 {
        let mut s = 0.0;
        for i in 0..4 {
            s += a[i][j].norm();
        }
        m = m.max(s);
    }
    m
}

fn mat4_add(a: &Mat4, b: &Mat4) -> Mat4 { /* element-wise */ }
fn mat4_scale(a: &Mat4, s: C) -> Mat4 { /* element-wise */ }

/// exp(A). Accurate to ~1e-15 for ||A||_1 <= ~25 with TAYLOR_N = 16.
pub fn mat4_expm(a: &Mat4) -> Mat4 {
    let n = mat4_norm1(a);
    // scale: want ||As|| <= 0.5
    let mut s = 0u32;
    let mut scaled = *a;
    while mat4_norm1(&scaled) > 0.5 && s < 32 {
        scaled = mat4_scale(&scaled, C::new(0.5, 0.0));
        s += 1;
    }
    // Taylor: T = I + As + As^2/2! + ... (Horner on the scaled matrix)
    let mut term = mat4_identity();
    let mut sum = mat4_identity();
    for k in 1..=16u32 {
        term = mat4_mul(&term, &scaled);
        let inv = 1.0 / (1..=k).product::<u32>() as f64;
        sum = mat4_add(&sum, &mat4_scale(&term, C::new(inv, 0.0)));
    }
    // square back
    for _ in 0..s {
        sum = mat4_mul(&sum, &sum);
    }
    let _ = n;
    sum
}
```

Required `cmatrix.rs` additions (same `#[inline]` style as `mat2_add` neighbors;
place directly after `mat4_similarity_diag`):

```rust
// cmatrix.rs
/// Element-wise add.
#[inline]
pub fn mat4_add(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = mat4_zero();
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

/// Element-wise scale by a complex scalar.
#[inline]
pub fn mat4_scale(a: &Mat4, s: C) -> Mat4 {
    let mut out = mat4_zero();
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][j] * s;
        }
    }
    out
}

/// 1-norm (max column absolute sum); subordinate, so ||AB|| ≤ ||A||·||B||.
#[inline]
pub fn mat4_norm1(a: &Mat4) -> f64 {
    let mut m = 0.0;
    for j in 0..4 {
        let mut s = 0.0;
        for i in 0..4 {
            s += a[i][j].norm();
        }
        if s > m {
            m = s;
        }
    }
    m
}
```

`expm.rs` then imports `{mat4_add, mat4_identity, mat4_mul, mat4_norm1,
mat4_scale, mat4_zero, Mat4, C}` from `crate::cmatrix` and keeps only
`mat4_expm` + tests. `lib.rs` gains `pub mod expm;` (core, NOT feature-gated
— no pyo3 dependency, so `cargo test` without `--features python` covers it).

Horner-form alternative for the Taylor loop (fewer temporaries; equivalent
result — pick one, note the choice in a comment):

```rust
// S = I + As·(I + As/2·(I + As/3·(...))) — 16 muls, 16 adds, no factorials:
let mut s_sum = mat4_identity();
for k in (1..=16u32).rev() {
    s_sum = mat4_add(&mat4_identity(), &mat4_scale(&mat4_mul(&scaled, &s_sum), c(1.0 / k as f64, 0.0)));
}
```

Remainder estimate for the record: with `||As|| ≤ 0.5`, truncation after
`As^16/16!` is bounded by `||As||^17/17! · e^{||As||} ≈ 0.5^17/17!·1.65 ≈
2e-23`. Squaring amplifies relative error by ≈ `s·κ` — negligible for our
sizes. Gate G9a pins this empirically.

### 3.3 Wiring into `transfer.rs`

The Berreman matrices are already available (`berreman_simple` /
`berreman_full` return `Mat4` in the ψ basis — exactly the `Δ` that EM
exponentiates):

```rust
// transfer.rs
use crate::expm::mat4_expm;
use crate::berreman::{berreman_full, berreman_simple};

/// Exponential layer matrix E_j = exp(i·k0·d_j·Δ_j).
/// Needs eps (+ MO set) and kx, so it takes &LayerSpec, not &SLayer.
fn exp_layer(ls: &LayerSpec, kx: f64, k0: f64) -> Mat4 {
    let d = match &ls.full {
        Some((rho, rhop, mu)) => berreman_full(&ls.eps, rho, rhop, mu, kx),
        None => berreman_simple(&ls.eps, kx),
    };
    let h = c(0.0, k0 * ls.thickness_nm);
    let mut a = d;
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] *= h;
        }
    }
    mat4_expm(&a)
}

/// Full exponential matrix: exit.P⁻¹ · (E_{N-1}…E_0) · entry.P.
/// Mirrors transfer_matrix() line-for-line (same half-space sandwich).
pub fn exponential_matrix(
    entry: &SLayer, layers: &[LayerSpec], slayers: &[SLayer],
    exit: &SLayer, kx: f64, k0: f64,
) -> Option<Mat4> {
    let mut t = crate::cmatrix::mat4_identity();
    for (ls, _sl) in layers.iter().zip(slayers.iter()) {
        t = mat4_mul(&exp_layer(ls, kx, k0), &t);
    }
    let exit_pinv = mat4_inv(&exit.p)?;
    Some(mat4_mul(&mat4_mul(&exit_pinv, &t), &entry.p))
}
```

`solve_stack` match gains:

```rust
let jones = match method {
    Method::Scattering => { ... unchanged ... }
    Method::Transfer => {
        let tm = transfer_matrix(&entry, &slayers, &exit)?;
        fresnel_from_transfer(&tm)   // unchanged
    }
    Method::Exponential => {
        let em = exponential_matrix(&entry, layers, &slayers, &exit, kx, k0)?;
        fresnel_from_transfer(&em)   // SAME extraction (== live _get_fresnel_EM)
    }
};
```

`Method` enum delta (`transfer.rs` — note existing variants have no payloads,
so all `match method` sites fail to compile until updated: compiler-guided
migration; sites are `solve_stack` + `pybind::sweep`'s code mapping):

```rust
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Method {
    Scattering,
    Transfer,
    Exponential, // NEW: transfer via direct matrix exponential (== live "EM")
}
```

`solve_stack` integration — `kx`/`k0` are already in scope (computed at the
head), so `exponential_matrix` takes them as plain `f64` params:

```rust
    let jones = match method {
        Method::Scattering => {
            let mut rough: Vec<(i32, f64)> = Vec::with_capacity(layers.len() + 1);
            for ls in layers {
                rough.push(ls.front_roughness);
            }
            rough.push(geom.exit_roughness);
            let sm = scattering_matrix(&entry, &slayers, &exit, k0, &rough)?;
            fresnel_from_scattering(&sm)
        }
        Method::Transfer => {
            let tm = transfer_matrix(&entry, &slayers, &exit)?;
            fresnel_from_transfer(&tm)
        }
        Method::Exponential => {
            // Same half-space sandwich as TM; layer matrices are
            // exp(i·k0·d·Δ) instead of P·Q·P⁻¹ (== live _get_fresnel_EM
            // sharing _get_fresnel_TM's extraction).
            let em = exponential_matrix(&entry, layers, &exit, kx, k0)?;
            fresnel_from_transfer(&em)
        }
    };
```

Note the simplified `exponential_matrix` signature vs the first sketch —
`slayers` is NOT needed (only `LayerSpec` for Δ + thickness):

```rust
pub fn exponential_matrix(
    entry: &SLayer,
    layers: &[LayerSpec],
    exit: &SLayer,
    kx: f64,
    k0: f64,
) -> Option<Mat4> {
    let mut t = crate::cmatrix::mat4_identity();
    for ls in layers {
        t = mat4_mul(&exp_layer(ls, kx, k0), &t);
    }
    let exit_pinv = mat4_inv(&exit.p)?;
    Some(mat4_mul(&mat4_mul(&exit_pinv, &t), &entry.p))
}
```

`slayers` (eigen-basis) are still built in `solve_stack` — EM needs the
entry/exit `SLayer`s for the sandwich. Interior eig work IS duplicated for EM
today; acceptable v1 (document; lazy-skip as follow-up — the skip must key on
`method == Exponential && isotropic-exit`, else `q_raw`/dressing paths diverge).

Roughness + EM guard in `pybind::sweep` (extend the existing match — today it
maps `1 → Transfer` else `Scattering` with a roughness rejection on TM):

```rust
    let has_rough = rough_types.iter().any(|&t| t != 0);
    let method = match method_code {
        1 => {
            if has_rough {
                return Err(/* ... existing TM message ... */);
            }
            Method::Transfer
        }
        2 => {
            if has_rough {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "roughness (Route 1) is only supported with the scattering method; \
                     use method='scattering' or expand the interface into graded sublayers",
                ));
            }
            Method::Exponential
        }
        _ => Method::Scattering, // 0 and any unknown code stay SM (legacy)
    };
```

Wrapper `_METHOD` becomes
`{"scattering": 0, "sm": 0, "transfer": 1, "tm": 1, "exponential": 2,
"em": 2, "exp": 2}` — one-line change.

### 3.4 pybind + Python

- `method_code`: `0 = SM, 1 = TM, 2 = EM`. `_METHOD` map gains
  `{"exponential": 2, "em": 2, "exp": 2}`.
- No array-layout changes. Docstring: EM is the cross-check engine; for very
  thick stacks prefer SM (EM/TM both suffer growing-exponential
  ill-conditioning — same caveat live pyllama carries).

### 3.5 Tests & gates (new gate G9)

- **G9a expm unit** (`expm.rs::tests`, full code — uses only crate helpers):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmatrix::{c, cone, czero, mat4_identity, mat4_zero};

    fn max_diff(a: &Mat4, b: &Mat4) -> f64 {
        let mut m = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                m = m.max((a[i][j] - b[i][j]).norm());
            }
        }
        m
    }

    #[test]
    fn expm_zero_is_identity() {
        let z = mat4_zero();
        assert!(max_diff(&mat4_expm(&z), &mat4_identity()) < 1e-15);
    }

    #[test]
    fn expm_diag_and_inverse() {
        // diag(0.3, −0.2+0.1i, 0.05i, 0) vs analytic element-wise exp
        let d = [c(0.3, 0.0), c(-0.2, 0.1), c(0.0, 0.05), czero()];
        let mut a = mat4_zero();
        let mut expect = mat4_zero();
        for i in 0..4 {
            a[i][i] = d[i];
            expect[i][i] = d[i].exp();
        }
        assert!(max_diff(&mat4_expm(&a), &expect) < 1e-14);
        // exp(A)·exp(−A) = I
        let neg = mat4_scale(&a, c(-1.0, 0.0));
        let prod = mat4_mul(&mat4_expm(&a), &mat4_expm(&neg));
        assert!(max_diff(&prod, &mat4_identity()) < 1e-14);
    }

    #[test]
    fn expm_rotation_block() {
        // generator of a 0.7-rad rotation in the (0,1) plane, embedded in 4x4:
        // exp([[0,−θ],[θ,0]]) = [[cosθ,−sinθ],[sinθ,cosθ]]
        let th = 0.7;
        let mut a = mat4_zero();
        a[0][1] = c(-th, 0.0);
        a[1][0] = c(th, 0.0);
        let e = mat4_expm(&a);
        assert!((e[0][0] - c(th.cos(), 0.0)).norm() < 1e-14);
        assert!((e[0][1] + c(th.sin(), 0.0)).norm() < 1e-14);
        assert!((e[1][0] - c(th.sin(), 0.0)).norm() < 1e-14);
        assert!((e[1][1] - c(th.cos(), 0.0)).norm() < 1e-14);
        assert!((e[2][2] - cone()).norm() < 1e-15);
        assert!((e[3][3] - cone()).norm() < 1e-15);
    }

    #[test]
    fn expm_nilpotent_is_exact() {
        // strictly upper-triangular shift N (N⁴ = 0): exp(N) = I+N+N²/2+N³/6
        let mut n = mat4_zero();
        n[0][1] = c(0.5, -0.2); n[1][2] = c(0.3, 0.4); n[2][3] = c(-0.1, 0.6);
        n[0][2] = c(0.2, 0.1); n[1][3] = c(0.4, -0.3); n[0][3] = c(0.1, 0.1);
        let n2 = mat4_mul(&n, &n);
        let n3 = mat4_mul(&n2, &n);
        let expect = mat4_add(
            &mat4_add(&mat4_identity(), &n),
            &mat4_add(&mat4_scale(&n2, c(0.5, 0.0)), &mat4_scale(&n3, c(1.0 / 6.0, 0.0))),
        );
        assert!(max_diff(&mat4_expm(&n), &expect) < 1e-14);
    }

    #[test]
    fn expm_scaled_berreman_sized() {
        // realistic magnitude: ||i·k0·d·Δ|| ~ 20 (1000 nm slab at 550 nm).
        // Hand-built Δ-like matrix with 1-norm ≈ 2.2, scaled by 10.
        let mut d = mat4_zero();
        d[0][1] = c(0.9, 0.0); d[1][0] = c(2.1, 0.1); d[1][2] = c(0.4, 0.0);
        d[2][3] = cone(); d[3][1] = c(0.3, 0.0); d[3][2] = c(1.8, -0.2);
        let a = mat4_scale(&d, c(0.0, 10.0));
        // group property: exp(A) = exp(A/2)² to 1e-13
        let half = mat4_expm(&mat4_scale(&a, c(0.5, 0.0)));
        let sq = mat4_mul(&half, &half);
        assert!(max_diff(&mat4_expm(&a), &sq) < 1e-13);
    }
}
```
- **G9b EM ≡ TM** on all 5 canonical cases (simple path) + 1 gyrotropic
  case (full path): Jones ≤ 1e-12 (looser than SM/TM because the algorithms
  are independent — that is the point), R/T ≤ 1e-12. OBSERVED: 8.5e-15.
- **G9c EM vs live pyllama EM** (`method="EM"`) on iso + rotated slab ≤ 1e-9.
  OBSERVED: 2.3e-15. (Test note: single-point solves squeeze Jones to
  (2,2) — compare whole matrices, not `[0, 0]` scalars.)
- **G9d B44 cross-check — DONE in Phase 1** (was deferred for the G7b
  harness): B44 *is* an EM engine (Padé-7); `Method::Exponential` vs B44
  (Jr direct + T via B44 flux, same arms as G7b) ≤ 1e-9. OBSERVED: 1.4e-15.
  Triangle closed: SM/TM/EM + B44 agree. Arm lives in
  `tests/test_aniso_exit.py` (reuses `b44_flux_T`).

*Upgrade path (not v1): Padé(6,6)/`expm`-style reduction if profiles show EM
dominating runtime; Taylor-16 is ~16 matmuls + ≤6 squarings per layer.*
---

## Phase 4 — Periodic / Bragg fast path (`periods`)

### 4.1 Why / upstream reference

- Live pyllama: `Structure(..., N_periods=1)` (default 1) and every pyllama
  `*Model`
  carries `N_per` (pyllama's `StackModel`, `CholestericModel`, `MixedModel` —
  pyllama classes, not B44's; B44's twist class is `TwistedMaterial`);
  DBR/cavity apparatus (`create_DBR_cavity_params_list`,
  `DBR_simulation_lists_from_params`, `cavity_berreman_angle_sweep` in
  `berreman_mueller.py`) assumes cheap repetition.
- B44: `RepeatedLayers(Layer)` wrapper with identical semantics.
- navette today: users must Python-expand `layers * N` (N× marshalling, N×
  eig work, O(N) Redheffer combines). For a 40-period DBR grid sweep this
  dominates runtime despite Rayon.

### 4.2 Design

Add a repetition count threaded from wrapper → sweep → solve. Two execution
routes, chosen automatically:

```rust
// transfer.rs
/// Raise a transfer/exponential unit-cell matrix to the p-th power
/// (binary exponentiation, O(log p) matmuls).
pub fn mat4_pow(m: &Mat4, mut p: u32) -> Mat4 {
    let mut base = *m;
    let mut acc = crate::cmatrix::mat4_identity();
    while p > 0 {
        if p & 1 == 1 {
            acc = mat4_mul(&base, &acc);
        }
        base = mat4_mul(&base, &base);
        p >>= 1;
    }
    acc
}

/// Redheffer power S^⊗p for a unit-cell scattering matrix.
/// S^⊗2 = combine(S, S); binary exponentiation identical to mat4_pow
/// but with s_combine as the product. Valid because every cell is identical.
fn s_pow(s: &Mat4, mut p: u32) -> Mat4 {
    let mut base = *s;
    let mut acc = crate::cmatrix::mat4_identity(); // ⊗-identity: check §4.4!
    while p > 0 {
        if p & 1 == 1 {
            acc = s_combine(&base, &acc);
        }
        base = s_combine(&base, &base);
        p >>= 1;
    }
    acc
}
```

`solve_stack` currently takes `layers: &[LayerSpec]`. Refactor first so the
SM interior assembly is reusable (behavior-preserving; G1–G6 pin it), then add
the periodic entry point:

```rust
// transfer.rs — 1-keyword change + extraction
pub(crate) fn s_combine(ab: &Mat4, bc: &Mat4) -> Mat4 { /* body unchanged */ }

/// Interior scattering matrix of a bare layer sequence (no half-spaces):
/// combines s_to_next(l[k], l[k+1]) for k = n-2..0. This is EXACTLY the
/// `if n >= 2` block inside scattering_matrix(), extracted verbatim.
fn scattering_interior(
    slayers: &[SLayer],
    k0: f64,
    rough_interior: &[(i32, f64)], // len n-1: dressings of l[k]|l[k+1], k=0..n-2
) -> Option<Mat4> {
    let n = slayers.len();
    let mut s = crate::cmatrix::mat4_identity();
    if n >= 2 {
        for kl in (0..(n - 1)).rev() {
            let (rt, sg) = rough_interior[kl];
            let s_layer = s_to_next(&slayers[kl], &slayers[kl + 1], k0, rt, sg)?;
            s = s_combine(&s_layer, &s);
        }
    }
    Some(s)
}

/// Previous scattering_matrix() body becomes (verify by diff — same order):
// pub fn scattering_matrix(entry, layers, exit, k0, rough) {
//     let s_in = scattering_interior(layers, k0, &rough[1..n])?; // interfaces l[k]|l[k+1]
//     let s_entry = s_to_next(entry, &layers[0], k0, rough[0])?;
//     let s_exit = s_to_next(&layers[n-1], exit, k0, rough[n])?;
//     s = combine(s_in, s_exit); s = combine(s_entry, s); Some(s)
```

Roughness index bookkeeping (critical — get this wrong and G10b fails): in
`scattering_matrix`, `rough` has length `n+1` (`[entry|l0, l0|l1, …, l{n-1}|exit]`)
and the interior loop uses `rough[kl+1]`. So interior slice = `rough[1..n]`
(length `n-1`, where element `kl` dresses `l[kl]|l[kl+1]`). For a periodic cell
of length L with per-layer `front_roughness`, the cell's interior dressings are
`cell[k].front_roughness` for `k = 1..L` (interface `l[k-1]|l[k]` is stored as
the FRONT of layer k) — i.e. interior slice `[cell[1].front, …, cell[L-1].front]`
(length L−1 ✓). The cell-BOUNDARY dressing (last layer of one repetition |
first layer of the next) is `cell[0].front_roughness`, applied when combining
cells — but `s_pow` combines IDENTICAL dressed cells, so fold the boundary
into the cell matrix itself: build the cell interior over an AUGMENTED sequence
`[last_cell_layer, cell...]`? NO — simpler and exact: the repeated structure's
interface list is periodic, so every `l|L-1 → l|0` boundary carries
`cell[0].front_roughness`, and every internal `l[k-1] → l[k]` carries
`cell[k].front_roughness`. The unit cell for powering is therefore the cyclic
sequence with dressings `[cell[0].front (as its exit), cell[1..L].front
(interior)]`. Implementation: build the cell S-matrix over the L layers with
interior dressings `cell[1..L].front`, then dress the wrap interface by
combining with a zero-thickness copy? Cleanest correct construction: assemble
the interior SM of the EXPANDED single cell PLUS its trailing boundary as an
(L+1)-layer interior problem is wrong (duplicates a layer).

Correct minimal approach — power the *transfer* cell and sandwich once for
TM/EM (trivially exact), and for SM power the *dressed cell scattering block*
defined as follows. Let `C` = combine over k of `s_to_next` for the cyclic
order, which for Redheffer repetition means: the cell block `S_cell` maps
`(fwd into cell, bwd into cell-from-right)` → `(fwd out, bwd out)` INCLUDING
the trailing boundary dressing. Build it as:

```rust
/// CORRECTED at implementation (the draft over-counted wraps — see §4.4):
/// total SM interior = I ⊗ (W⊗I)^{N-1}, where I is the bare cell interior
/// and W⊗I the wrap-first cell (wrap = dressed last→first interface, folded
/// LEFT exactly like live's S_period = combine(wrap, interior)). Powering an
/// interior-first [I,W] block N times would append a spurious Nth wrap whose
/// output (first-layer modes) mismatches the exit sandwich (expects last).
/// So: NO cell_block helper — solve_stack_periodic builds `bare`, `wrap`,
/// `wcell = combine(wrap, bare)` inline and powers N−1 (see §4.2 as built).
```

ORDER NOTE (reviewer must check): `scattering_matrix` builds interior by
`for kl in (0..n-1).rev(): s = combine(s_layer(kl,kl+1), s)` — i.e. each new
LEFT block combines as `combine(new_left, accumulated_right)`. Folding `wrap`
on the right is `combine(accumulated, wrap)` ✓ consistent.

Single-layer cell edge case (L == 1): interior is identity, block = wrap =
`s_to_next(l0, l0)` dressed with `cell[0].front` — a pure dressed interface
repeated p times with the layer's own propagation inside `s_to_next` via
`a.q`. Verify against expanded in G10b (this case is covered there).

Main entry point:

```rust
pub fn solve_stack_periodic(
    geom: &Geometry,
    cell: &[LayerSpec],
    periods: u32,          // >= 1 (0 => None; pybind rejects earlier)
    method: Method,
) -> Option<SolveResult> {
    if periods == 0 || cell.is_empty() {
        return None;
    }
    if periods == 1 {
        return solve_stack(geom, cell, method); // zero delta (G6)
    }
    let k0 = 2.0 * std::f64::consts::PI / geom.wl_nm;
    let kx = geom.n_entry.re * geom.theta_in_rad.sin();
    // ... same kz/factor/entry/exit head as solve_stack (share via the
    // build_point() refactor from Phase 2 if landed, else duplicate 12 lines)
    // cell eigen-basis (L eigs, not N·L):
    let mut cell_sl = Vec::with_capacity(cell.len());
    for ls in cell { /* layer_waves_* + from_waves, as in solve_stack */ }

    let jones = match method {
        Method::Scattering => {
            // AS BUILT (draft corrected — it powered [I,W]^N with a spurious
            // Nth wrap; see cell_block tombstone + §4.4): total interior is
            // I ⊗ (W⊗I)^{N-1} — leading bare cell, wrap-first cell powered
            // N−1, exactly live's S_last_period ⊗ S_period^{N-1} structure.
            // Dressing counts == expanded: cell[0].front at entry + each of
            // the N−1 wraps (N uses); internal fronts N times each. ✓
            let interior: Vec<(i32, f64)> =
                (1..l).map(|k| cell[k].front_roughness).collect();
            let bare = scattering_interior(&cell_sl, k0, &interior)?;
            let (rt0, sg0) = cell[0].front_roughness;
            let wrap = s_to_next(&cell_sl[l - 1], &cell_sl[0], k0, rt0, sg0)?;
            let wcell = s_combine(&wrap, &bare); // wrap-first cell [W, I]
            let s = s_combine(&bare, &s_pow(&wcell, periods - 1));
            let s_entry = s_to_next(&entry, &cell_sl[0], k0, rt0, sg0)?;
            let (rtn, sgn) = geom.exit_roughness;
            let s_exit = s_to_next(&cell_sl[l - 1], &exit, k0, rtn, sgn)?;
            let s = s_combine(&s_combine(&s_entry, &s), &s_exit);
            fresnel_from_scattering(&s)
        }
        Method::Transfer => {
            let mut tcell = crate::cmatrix::mat4_identity();
            for sl in &cell_sl {
                tcell = mat4_mul(&mat4_similarity_diag(&sl.p, &sl.q)?, &tcell);
            }
            let t = mat4_mul(&mat4_pow(&tcell, periods), &mat4_identity());
            let exit_pinv = mat4_inv(&exit.p)?;
            let tm = mat4_mul(&mat4_mul(&exit_pinv, &t), &entry.p);
            fresnel_from_transfer(&tm)
        }
        Method::Exponential => {
            let mut ecell = crate::cmatrix::mat4_identity();
            for ls in cell {
                ecell = mat4_mul(&exp_layer(ls, kx, k0), &ecell)?; // exp_layer infallible; no ?
            }
            // ... same sandwich with mat4_pow(&ecell, periods)
        }
    };
    // ... same jones_circ + power branch as solve_stack (share via helper
    // `fn finish(jones, factor, ...) -> SolveResult` incl. Phase-1 Flux arm)
}
```

Refactor hygiene (AS BUILT): `build_head()` (k0/kx/factor/entry/exit),
`build_slayers()`, and `finish()` (jones → circ → power → SolveResult,
taking a precomputed `tm_full: Option<Mat4>` for the Flux arm) extracted and
shared by `solve_stack` + `solve_stack_periodic` — both grow the Phase-1
Flux arm identically by construction. `solve_stack`'s Transfer arm passes its
`tm` through (`tm_hint`, avoiding a rebuild); SM/EM arms pass None and the
caller fills it lazily only for anisotropic exits (zero cost on the legacy
path). The periodic path always has the powered sandwich ready, so it passes
`Some(tm_full)` unconditionally (~log N matmuls, negligible). Phase 2 reuses
`build_head()` for the fields engine (dependency flip noted — P2 consumes
P4's helper, not vice versa).
- **Roughness interaction:** Route-1 dressing is per-interface. Semantics with
  repetition: dress *every* cell boundary identically (physical superlattice).
  That breaks the identical-cell assumption only in the dressing factors —
  but the factors are identical per repetition, so the dressed cell matrix is
  still identical per repetition. ✅ Fast path stays valid: build the cell
  *with* its front dressings, then power. The final `cell|exit` dressing uses
  `geom.exit_roughness` once. The only subtlety is SM interior assembly order
  (`scattering_matrix` combines `kl = n−2..0` then entry/exit) — replicate it
  for the cell, then `s_pow`, then sandwich. Unit-test expanded-vs-periodic
  with roughness on (G10b).

`Geometry` vs parameter: put `periods: u32` on the solve-call parameter, **not**
in `Geometry` (geometry is (λ,θ)-point data; repetition is stack data).
`pybind::sweep` gains a `periods: u32` scalar argument (same for all grid
points — matches `N_periods` semantics).

### 4.3 pybind + Python

pybind threading — `periods` is a plain `u32` scalar (same for the whole grid,
like `method`). Validation FIRST (before any allocation):

```rust
// pybind.rs — in sweep(), right after the non-empty check:
fn decode_method_periods(method_code: i32, periods: u32) -> PyResult<Method> {
    if periods == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "periods must be >= 1 (periods=1 is a plain stack)",
        ));
    }
    Ok(match method_code {
        1 => Method::Transfer,
        2 => Method::Exponential,
        _ => Method::Scattering,
    })
}
// (merge with the Phase-3 roughness guard: validate rough×method here too,
// so all arg errors surface before the Rayon section.)
```

`sweep` gains `periods: u32`; the Rayon closure calls:

```rust
match solve_stack_periodic(&geom, &stacks[w], periods, method) {
    Some(s) => PointOut::from_solve(&s, power_is_flux),
    None => PointOut::nan(),
}
// periods == 1 delegates to solve_stack internally (G6: identical numbers).
```

`PointOut::from_solve` signature: it currently takes `&SolveResult`; the
`flux` flag comes from `s.power_method == PowerMethod::Flux` — read it off
the result itself (no extra param needed; correct the Phase-1 sketch which
suggested a param).

`solve_grid_simple/full` signatures gain trailing `periods` (after `exit_*`
from Phase 1; full order: `(..., method, exit_eps[, exit_rho, exit_rhop,
 exit_mu], periods)`). Update the `#[pyo3(signature = ...)]` lists AND the
wrapper call sites in the same commit (compiler will not catch arity drift
across the FFI — grep `solve_grid_simple\|solve_grid_full` in both files).

Wrapper (`navette/berreman.py`):

```python
class BerremanStack:
    def __init__(self, layers, wavelengths_nm, angles, n_entry=1.0, n_exit=1.0,
                 method="scattering", angles_in_radians=False,
                 exit_roughness=None, exit_eps=None, ..., periods=1):  # NEW
        # ... existing body ...
        if not isinstance(periods, (int, np.integer)) or periods < 1:
            raise ValueError("periods must be an integer >= 1")
        self.periods = int(periods)

    def solve(self):
        # ... existing marshalling ...
        # `layers` here are the UNIT CELL; Rust repeats them `periods` times.
        raw = _rs_solve_simple(*args, eps, thick, rtypes, rvals,
                                self.method, e_eps, self.periods)
```

Docs (README + docstring): roughness dressings repeat per cell — state
explicitly with an example (2-layer cell ×3 uses l0-front 4 times: entry + 3
wraps... precisely: entry|cell once + one wrap per repetition = periods+1
uses of cell[0].front; internal fronts used `periods` times each). This differs
from naive `layers * N` Python expansion ONLY in that naive expansion puts
`cell[0].front` at every wrap too — wait, it is IDENTICAL to naive expansion.
State that: `periods=N` ≡ `layers * N` elementwise, verified by G10b. (The
superiority claim vs "dress once" in the earlier sketch was confused; the
semantics equal full expansion — which is what B44 `RepeatedLayers` does.)

### 4.4 Correctness notes (read before coding)

- Redheffer ⊗-identity: `s_combine(I4, S) == S`? Verify algebraically from the
  block formula: with `ab = I`, `ab00 = I, ab01 = 0, ab10 = 0, ab11 = I` →
  `C = (I − 0)⁻¹ = I`, `ac00 = bc00·ab00 = bc00` ✓, `ac01 = bc01 + bc00·0·… =
  bc01` ✓, `ac10 = 0 + I·bc10·ab00 = bc10·ab00`… careful: `ac10 = ab10 +
  ab11·bc10·C·ab00 = bc10` ✓ (ab00 = I), `ac11 = inner·bc11` with
  `inner = I + bc10·0 = I` → `ac11 = ab11·bc11 = bc11` ✓. And
  `s_combine(S, I)`: `bc = I` → `C = (I − ab01·0)⁻¹ = I`,
  `ac00 = I·C·ab00 = ab00` ✓, `ac01 = 0 + I·C·ab01·I = ab01` ✓,
  `ac10 = ab10 + ab11·0·… = ab10` ✓, `ac11 = ab11·(I+0)·I = ab11` ✓.
  So `I4` is the two-sided identity — `s_pow` valid. **Still**: pin with a
  randomized unit test (`s_combine(I,S)==S==s_combine(S,I)`, G10a).
- `mat4_pow` order: our transfer convention multiplies left
  (`t = T_layer · t`), so cell power `T_cell^p` with left-multiplication in
  `mat4_pow` (`acc = base·acc`) preserves layer order. Pin vs expanded product
  (G10b).
- Overflow: same as TM for large p — SM route stays the safe default;
  document `periods` + TM/EM caveat next to the existing thick-stack caveat.

### 4.5 Tests & gates (new gate G10)

- **G10a algebra** (`transfer.rs::tests` — deterministic pseudo-random via a
  SplitMix64/xorshift inline, NO rand dependency):

```rust
fn xorshift(state: &mut u64) -> f64 {
    // deterministic [-1, 1) doubles; no new deps (MSRV-safe, audit-trivial)
    *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17;
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

fn rand_mat4(state: &mut u64) -> Mat4 {
    let mut m = crate::cmatrix::mat4_zero();
    for i in 0..4 { for j in 0..4 { m[i][j] = c(xorshift(state), xorshift(state)); } }
    m
}

#[test]
fn pow_algebra() {
    let mut st = 0x9E3779B97F4A7C15u64;
    for _ in 0..20 {
        let a = rand_mat4(&mut st);
        // mat4_pow vs naive loop, p = 0..7 (p=0 must be identity!)
        let mut naive = crate::cmatrix::mat4_identity();
        assert_eq!(mat4_pow(&a, 0), naive); // by construction; pins the edge
        for p in 1..8u32 {
            naive = mat4_mul(&a, &naive);
            let d = max_diff(&mat4_pow(&a, p), &naive);
            assert!(d < 1e-12, "pow {p} diff {d:e}");
        }
    }
}

#[test]
fn redheffer_identity_and_pow() {
    let mut st = 0x243F6A8885A308D3u64;
    for _ in 0..20 {
        // physically-shaped S (blocks < 1 in norm, like real scatterers)
        let s = mat4_scale(&rand_mat4(&mut st), c(0.1, 0.0));
        let i = crate::cmatrix::mat4_identity();
        assert!(max_diff(&s_combine(&i, &s), &s) < 1e-14);
        assert!(max_diff(&s_combine(&s, &i), &s) < 1e-14);
        // s_pow vs naive repeated combine, p = 1..6
        let mut naive = i;
        for p in 1..7u32 {
            naive = s_combine(&s, &naive);
            // NOTE order: repeated LEFT-combine matches s_pow's acc update
            // (s_pow does acc = combine(base, acc) — same association).
            let d = max_diff(&s_pow(&s, p), &naive);
            assert!(d < 1e-12, "spow {p} diff {d:e}");
        }
    }
}
```

  Association warning: Redheffer combination is associative (it is — it
  represents cascading two-ports), but VERIFY empirically: G10a also asserts
  `combine(combine(A,B),C) == combine(A,combine(B,C))` on 10 random triples
  ≤ 1e-12. If associativity failed, binary exponentiation would be invalid —
  this test is load-bearing, not decorative.
- **G10b expanded ≡ periodic:** Bragg 10× bilayer (stress-test stack) and
  twisted 20-slice cell × 3, smooth **and** rough-dressed, SM+TM+EM:
  R/T/Jones ≤ 1e-12. OBSERVED: 5.6e-15 (incl. rough SM; TM/EM smooth).
  (The draft's spurious-Nth-wrap bug was caught during pre-coding review of
  live's S_period ordering; G10b passed first run on the corrected build.)
- **G10c vs live `N_per`:** `StackModel(..., N_per=10)` on the Bragg stack,
  SM+TM ≤ 1e-9. OBSERVED: 2.2e-15. (Live SM powers wrap-first S_period^{N-1}
  after a leading bare interior — our construction mirrors it; live TM/EM
  power the bare cell N times, no wraps involved.)
- **G10d perf:** 40-period DBR 40wl×9ang sweep in `tests/test_periodic.py`:
  OBSERVED expanded 0.01s vs periodic 0.00s (×9.8), agreement 7.1e-14.
  Gate asserts periodic < expanded/2.
---

## Phase 5 — Mueller-matrix suite (new `src/mueller.rs`)

### 5.1 Why / upstream reference

- BerreMueller `mueller.py` (≈1337 lines) is a full Mueller calculus toolkit:
  `MUELLER_MATRIX[_STACK]` classes, `parallel_decompose_matrix_stack`,
  `cloude_decompose_matrix_stack`, `extract_cd_mm_stack` (circular dichroism),
  `polarizance_*` / `brown_params`, differential-matrix / Magnus machinery
  (`magnus_array`, `diff_mueller_params`), plus `berreman_mueller.py` helpers
  (`mueller_matrix_suite_from_refl_trans_amplitude_matrices_basis_ps`,
  `extract_mueller_matrix_factors`, `get_absorbance_sets`).
- navette today exposes exactly one op: `mueller_from_jones` (validated 8.9e-16).
  Users doing depolarization/CD analysis must round-trip through Python.
- v1 scope (deliberately narrow — scalar 4×4 post-processing, no new physics):
  **depolarization index, polarizance/diattenuation vectors, Cloude
  decomposition (eigenvalues + entropy), CD/CB extraction**. Differential/Magnus
  machinery is out of scope (needs z-resolved differential matrices — revisit
  after Phase 2 ships; note as follow-up).

### 5.2 Prerequisite spike (do first, ≤ 1 hour)

Read and transcribe the exact conventions (they differ between textbooks —
Ossikovski vs Chipman sign/normalization must match *live*, not memory):

1. `mueller.py::create_pauli_stack` — Pauli basis order/normalization used by
   `cloude_decompose_matrix_stack` (ll. 159–178, 534–574).
2. `mueller.py::parallel_decompose_matrix_stack` element ordering (ll. 507–518).
3. `berreman_mueller.py::mueller_from_jones_matrix` A-matrix (already mirrored
   in `transfer.rs::mueller_from_jones` — re-verify the `1/√2` factors).
4. `extract_cd_mm_stack` / `get_m03_raw` CD formula (ll. 583–593, `berreman_mueller` l.776).
5. `POLARIZANCE` / `brown_params` definitions if polarizance is wanted in v1
   (ll. 658–746) — else defer to v2 explicitly.

Record the transcribed formulas as doc-comments with line pointers, e.g.
`// = mueller.py::cloude_decompose_matrix_stack, coherency via create_pauli_stack`.

SPIKE OUTCOME (executed at implementation; all from live source):

1. Pauli = `create_pauli_stack()` default `"optics"`: [I, σz, σx, σy],
   UNNORMALIZED (mueller.py:159-178). Live coherency path (ll. 545-555):
   H = Σ_ij M_ij·kron(σ_i, conj(σ_j))/4, then C = L·H·L† with L = Λ/√2.
   Λ rows are pairwise orthogonal with norm √2 ⇒ L is unitary ⇒ eig(H) ==
   eig(C). Verified numerically: |λ(H) − λ(C)| ≤ 1.4e-15, Σλ = M00, Jones
   inputs → {M00,0,0,0}, Dirichlet mixture → 3 nonzero λ.
2. `parallel_decompose_matrix_stack` (ll. 507-518) = sym/antisym split —
   unused by every v1 metric. Noted, not ported.
3. A-matrix confirmed (ll. 156-160); ours already validated 8.9e-16.
4. CD: `get_m03_raw(t)` (berreman_mueller.py:776-782) satisfies
   get_m03_raw ≡ 2·M03 EXACTLY, same sign — shown analytically (with the
   A-matrix, M03 = Im(txx·conj(txy)) + Im(tyx·conj(tyy)) ∈ ℝ; live's raw
   formula is twice that sum) and numerically (5/5 random Jones, exact 2.0
   ratio). The /2 lives on the TEST side (`raw/(2·M00)`), never absorbed
   into our API (which returns dimensionless M03/M00, matching live's
   `norm_mueller_matrix_stack` convention, mueller.py:1154).
5. POLARIZANCE/Brown (ll. 658-746) = differential calculus over z-resolved
   (b_vec, d_vec) — DEFERRED to v2 (needs Phase 2 fields). v1 = plain
   normalized polarizance vector only.
6. No `depolarization_index_*`, `diattenuation_stack`, or eigenvalue-returning
   Cloude entry point exists in live. G11a pins: D/P magnitudes + components
   vs `get_purity_components_mueller_matrix_stack` (mueller.py:1197-1210,
   needs (4,4,X) stack) and `norm_mueller_matrix_stack`; λ vs live's OWN
   `create_pauli_stack` basis + numpy eigvalsh (== eig(live C) by (1));
   DI vs the Gil–Bernabeu textbook formula + the eigenvalue identity
   DI² = (4Σλ²/(Σλ)²−1)/3.
7. DRAFT BUG (in §5.3 as written): normalized Pauli basis with divisor /4
   gives Σλ = M00/2 (Tr(σ̃0)²/4 = 2/4), NOT M00. As built: UNNORMALIZED basis
   with /4 = live's H verbatim (G11b's Σλ==M00 assert would have failed
   loudly on the draft constant — the gate was correctly designed, the
   constant was not).
8. Importing live `mueller.py` needs the mpl `get_cmap` shim + sympy +
   pandas in the test env (all recorded in `tests/test_mueller_suite.py`).

### 5.3 New module `src/mueller.rs`

Operates on plain `[[f64; 4]; 4]` (outputs of `mueller_from_jones` are real;
take `.re` at the boundary — same as `pybind::PointOut` does today):

```rust
//! Mueller-matrix post-processing (depolarization / polarimetry metrics).
//! Scalar 4x4 real algebra over f64. Conventions transcribed from
//! BerreMueller mueller.py — see per-function line pointers from the §5.2 spike.
//!
//! Crate placement: `pub mod mueller;` in lib.rs (core, not feature-gated).

/// M[0][0] guard: every metric below normalizes by M00; M00 <= 0 => None
/// (unphysical / dark matrix — caller maps to NaN, same as n_failed rows).
fn m00(m: &[[f64; 4]; 4]) -> Option<f64> {
    let v = m[0][0];
    if v > 1e-300 { Some(v) } else { None }
}

/// Depolarization index (Gil–Bernabeu):
///   DI = sqrt( Σ_ij M_ij² − M00² ) / (√3 · M00).
/// DI == 1 ⟺ non-depolarizing (Mueller-Jones); DI == 0 for diag(1,0,0,0).
pub fn depolarization_index(m: &[[f64; 4]; 4]) -> Option<f64> {
    let m00 = m00(m)?;
    let mut s = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            s += m[i][j] * m[i][j];
        }
    }
    let v = (s - m00 * m00).max(0.0); // max() guards −1e-18 rounding on Jones inputs
    Some(v.sqrt() / (3.0_f64.sqrt() * m00))
}

/// Diattenuation vector D (first ROW, normalized): D_k = M[0][k+1]/M00.
pub fn diattenuation(m: &[[f64; 4]; 4]) -> Option<[f64; 3]> {
    let m00 = m00(m)?;
    Some([m[0][1] / m00, m[0][2] / m00, m[0][3] / m00])
}

/// Polarizance vector P (first COLUMN, normalized): P_k = M[k+1][0]/M00.
/// (If the §5.2 spike shows live POLARIZANCE carries length/convention extras
/// e.g. brown_params scaling, add them here and pin with G11a — default off.)
pub fn polarizance(m: &[[f64; 4]; 4]) -> Option<[f64; 3]> {
    let m00 = m00(m)?;
    Some([m[1][0] / m00, m[2][0] / m00, m[3][0] / m00])
}

/// Circular dichroism from the transmission Mueller matrix: CD = M03/M00
/// (== live berreman_mueller get_m03_raw convention — CONFIRM in spike;
/// some texts use −M03 or M30; G11a pins the sign against live).
pub fn circular_dichroism(m: &[[f64; 4]; 4]) -> Option<f64> {
    Some(m[0][3] / m00(m)?)
}

/// Cloude coherency eigenvalues λ0 ≥ λ1 ≥ λ2 ≥ λ3 (Σλ = M00) + entropy.
/// C = Σ_ij M_ij · B_ij, B_ij = (σ_i ⊗ σ_j*)/4 with Pauli basis below.
/// Eig via crate::cmatrix::eig4; |Im λ| ≤ 1e-9 asserted in tests (G11).
#[derive(Clone, Copy, Debug)]
pub struct Cloude {
    pub lambda: [f64; 4], // descending, Σ == M00 (normalization self-check)
    pub entropy: f64,     // S = −Σ p_k log4 p_k, p = λ/Σλ, 0·log0 := 0
}

use crate::cmatrix::{c, czero, eig4, C};

type Pauli = [[C; 2]; 2];

/// Pauli basis, written OUT explicitly — UNNORMALIZED, exactly live's
/// create_pauli_stack() default "optics" (see spike outcome §5.2.7: the draft's
/// normalized-basis-with-/4 gave Σλ = M00/2; unnormalized-with-/4 IS live's H
/// and gives Σλ == M00, asserted by G11b).
///   σ0 = I, σ1 = diag(1,−1), σ2 = [[0,1],[1,0]], σ3 = [[0,−i],[i,0]].
fn pauli_basis() -> [Pauli; 4] {
    let s = 1.0 / 2.0_f64.sqrt();
    let (s0, s1) = (c(s, 0.0), c(-s, 0.0));
    let i = c(0.0, s);
    [
        [[c(s, 0.0), czero()], [czero(), c(s, 0.0)]], // σ0
        [[c(s, 0.0), czero()], [czero(), s1]],        // σ1
        [[czero(), c(s, 0.0)], [c(s, 0.0), czero()]], // σ2
        [[czero(), -i], [i, czero()]],                // σ3: [[0,−i],[i,0]]/√2
    ]
}

/// Kronecker product with CONJUGATION on the second factor: K = A ⊗ conj(B),
/// 4x4, row-major block layout K[i*2+p][j*2+q] = A[i][j]·conj(B[p][q])
/// (matches the kron(J, conj(J)) layout in transfer.rs::mueller_from_jones).
fn kron_conj(a: &Pauli, b: &Pauli) -> [[C; 4]; 4] {
    let mut k = [[czero(); 4]; 4];
    for i in 0..2 {
        for j in 0..2 {
            for p in 0..2 {
                for q in 0..2 {
                    k[i * 2 + p][j * 2 + q] = a[i][j] * b[p][q].conj();
                }
            }
        }
    }
    k
}

pub fn cloude(m: &[[f64; 4]; 4]) -> Option<Cloude> {
    let m00v = m00(m)?;
    let sig = pauli_basis();
    // C = Σ_ij M_ij · (σ_i ⊗ σ_j*)/4
    let mut cm = [[czero(); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let k = kron_conj(&sig[i], &sig[j]);
            let w = c(m[i][j] / 4.0, 0.0);
            for r in 0..4 {
                for cc in 0..4 {
                    cm[r][cc] += w * k[r][cc];
                }
            }
        }
    }
    // AS BUILT: Hermitian eigenproblem → NEW eig_hermitian4 (cyclic Jacobi,
    // cmatrix.rs), NOT the general quartic eig4. Multiple roots are
    // ill-conditioned for polynomial methods ((λ−1/4)⁴ fuzz → complex pairs
    // imag ~1e-4); G11b's ideal depolarizer caught this at implementation —
    // Jacobi is degenerate-safe by construction. Values-only (Cloude needs no
    // vectors); input symmetrized, so no imag threshold arm remains.
    let mut lam = crate::cmatrix::eig_hermitian4(&hm);
    // sort descending (insertion sort, n=4)
    for n in 1..4 { // (lam already real — Jacobi returns f64 by construction)
        let x = lam[n];
        let mut k = n;
        while k > 0 && lam[k - 1] < x {
            lam[k] = lam[k - 1];
            k -= 1;
        }
        lam[k] = x;
    }
    // entropy, base-4 log; clamp tiny negatives from rounding to 0
    let sum: f64 = lam.iter().sum();
    if sum <= 1e-300 {
        return None;
    }
    let mut s = 0.0;
    for v in lam {
        let p = (v / sum).max(0.0);
        if p > 0.0 {
            s -= p * p.ln() / 4.0_f64.ln();
        }
    }
    Some(Cloude { lambda: lam, entropy: s })
}
```

Design notes (keep in final code as comments):

- `cloude` returning `None` on `im_max > 1e-9` vs clamping: returning None
  surfaces bad inputs (e.g. caller passed a non-Mueller matrix) instead of
  silently symmetrizing. Python maps None → NaN lambda + NaN entropy.
- Negative-but-tiny λ (−1e-17 from rounding on ideal Jones inputs): entropy
  clamps via `.max(0.0)`; `lambda` keeps raw values (tests assert `λ[1..] ≈ 0`
  with tolerance, NOT exact zeros).
- If the spike finds live uses a DIFFERENT Pauli order (e.g. optics ordering
  σ1↔σ3 swapped), only `pauli_basis()` changes — eigenvalues are invariant
  under simultaneous basis permutation, so G11a/b still pass; the ORDER of
  returned vector elements is basis-independent (sorted). State this — it
  defuses the "classic silent bug" risk considerably.

### 5.4 pybind + Python

Per-Jones-matrix batch op would complicate the sweep; instead expose the
metrics as **elementwise post-ops over the already-returned `M_refl/M_trans`**
in Python (zero Rust batching work, matches "thin wrapper" architecture):

pybind — scalar flat-16 functions (same `PyReadonlyArray1[f64]` idiom as
`mueller_from_jones_py`; `None→NaN` mapping explicit). Register all five in
`lib.rs::_berreman` next to `mueller_from_jones`:

```rust
// pybind.rs
use crate::mueller;

fn read_m16(re: &[f64]) -> PyResult<[[f64; 4]; 4]> {
    if re.len() != 16 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Mueller matrix must have 16 elements (row-major 4x4)",
        ));
    }
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = re[i * 4 + j];
        }
    }
    Ok(m)
}

/// Scalar depolarization index; NaN when M00 <= 0.
#[pyfunction]
#[pyo3(name = "depolarization_index")]
pub fn depolarization_index_py(
    _py: Python<'_>,
    m_flat: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let m = read_m16(m_flat.as_slice()?)?;
    Ok(mueller::depolarization_index(&m).unwrap_or(f64::NAN))
}

/// Scalar diattenuation (3,) / polarizance (3,) / CD — NaN triple/single.
#[pyfunction] #[pyo3(name = "diattenuation"]] // + polarizance, circular_dichroism: same shape
pub fn diattenuation_py(_py: Python<'_>, m_flat: PyReadonlyArray1<f64>) -> PyResult<Vec<f64>> {
    let m = read_m16(m_flat.as_slice()?)?;
    Ok(mueller::diattenuation(&m).map(|v| v.to_vec()).unwrap_or(vec![f64::NAN; 3]))
}

/// Cloude -> (lambda[4], entropy); NaNs when non-physical.
#[pyfunction]
#[pyo3(name = "cloude")]
pub fn cloude_py(
    py: Python<'_>,
    m_flat: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyAny>> {
    // Return a (lambda, entropy) tuple; wrapper splits into dict entries.
    let m = read_m16(m_flat.as_slice()?)?;
    let (lam, s) = match mueller::cloude(&m) {
        Some(cc) => (cc.lambda.to_vec(), cc.entropy),
        None => (vec![f64::NAN; 4], f64::NAN),
    };
    use pyo3::types::PyTuple;
    Ok(PyTuple::new(py, [lam.into_pyobject(py)?.into_any(), s.into_pyobject(py)?.into_any()])?.into())
}
```

Wrapper — grid-vectorized WITHOUT per-element Python overhead concerns
(`n_wl·n_th` scalar FFI calls are microseconds; but batch anyway for clean
code — reshape to `(-1, 16)`, one call per row, reshape back):

```python
from ._berreman import (
    depolarization_index as _rs_di,
    diattenuation as _rs_diat, polarizance as _rs_pol,
    circular_dichroism as _rs_cd, cloude as _rs_cloude,
)

def _apply_m16(fn, M, out_shape):
    """Apply a flat-16 Rust scalar fn over (...,4,4) -> out_shape."""
    M = np.asarray(M, dtype=float)
    if M.shape[-2:] != (4, 4):
        raise ValueError(f"expected (...,4,4), got {M.shape}")
    flat = M.reshape(-1, 16)
    rows = [np.asarray(fn(row.copy()), dtype=float).reshape(-1) for row in flat]
    return np.stack(rows).reshape(M.shape[:-2] + out_shape)

def depolarization_index(M):
    """DI over (...,4,4) Mueller matrices -> (...) float; 1 == non-depolarizing."""
    return _apply_m16(_rs_di, M, ())

def diattenuation(M):
    return _apply_m16(_rs_diat, M, (3,))

def polarizance(M):
    return _apply_m16(_rs_pol, M, (3,))

def circular_dichroism(M):
    return _apply_m16(_rs_cd, M, ())

def cloude(M):
    """-> dict(lambda: (...,4) descending, entropy: (...) base-4)."""
    M = np.asarray(M, dtype=float)
    flat = M.reshape(-1, 16)
    lam = np.empty((flat.shape[0], 4)); ent = np.empty(flat.shape[0])
    for k, row in enumerate(flat):
        l, s = _rs_cloude(row.copy())
        lam[k] = np.asarray(l, dtype=float); ent[k] = float(s)
    sh = M.shape[:-2]
    return {"lambda": lam.reshape(sh + (4,)), "entropy": ent.reshape(sh)}
```

### 5.5 Tests & gates (new gate G11)

- **G11a vs live `mueller.py`** (`tests/test_mueller_suite.py` — fixtures
  generated BY live, compared, never re-derived). AS BUILT (the draft's
  `*_stack` scalar names don't exist in live — spike §5.2.6; real pins below).
  OBSERVED worst: 3.6e-15 (budget 1e-12).

```python
import matplotlib.cm as _cm  # get_cmap shim (live mueller.py:746 needs it)
if not hasattr(_cm, "get_cmap"):
    _cm.get_cmap = lambda name=None: __import__("matplotlib").colormaps[name]
import numpy as np
from navette import berreman as bl
from berremueller import berreman_mueller as bm
from berremueller import mueller as live_mu

rng = np.random.default_rng(7)
worst = 0.0
# 20 random Mueller-Jones matrices
for _ in range(20):
    J = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    M = np.real(np.asarray(bm.mueller_from_jones_matrix(J)))
    stack = M.reshape(4, 4, 1)  # live purity/norm helpers need (4,4,X)
    pd, pp, _ = live_mu.get_purity_components_mueller_matrix_stack(stack)
    nM = np.asarray(live_mu.norm_mueller_matrix_stack(stack))[:, :, 0]
    d, p = bl.diattenuation(M), bl.polarizance(M)
    worst = max(worst, abs(norm(d) - pd[0]), abs(norm(p) - pp[0]),
                max(abs(d - nM[0, 1:])), max(abs(p - nM[1:, 0])))
    di_ref = sqrt(max(sum(M*M) - M[0,0]**2, 0)) / (sqrt(3)*M[0,0])  # textbook
    worst = max(worst, abs(bl.depolarization_index(M) - di_ref),
                abs(bl.depolarization_index(M) - 1.0))  # Jones => DI == 1
    raw = real(bm.get_m03_raw(J.reshape(2, 2, 1)))[0]  # == 2*M03 (spike §5.2.4)
    worst = max(worst, abs(bl.circular_dichroism(M) - raw / (2*M[0, 0])))
# 5 depolarizing fixtures: convex sums Σ w_k M(J_k), w from Dirichlet;
# λ reference = live's OWN create_pauli_stack basis + numpy eigvalsh
# (== eig of live's C by L-unitary similarity, spike §5.2.1)
for _ in range(5):
    ...
    c = bl.cloude(M)
    worst = max(worst, max(abs(c["lambda"] - lam_live)),
                abs(c["entropy"] - ent_live), abs(di**2 - di2))
print(f"mueller suite vs live worst: {worst:.3e}")
assert worst < 1e-12
```
- **G11b identities** (pure Rust, `mueller.rs::tests`):

```rust
#[test]
fn identities() {
    let eye = [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]];
    assert!((depolarization_index(&eye).unwrap() - 1.0).abs() < 1e-15);
    let cc = cloude(&eye).unwrap();
    assert!((cc.lambda[0] - 1.0).abs() < 1e-12);
    assert!(cc.lambda[1].abs() < 1e-12 && cc.lambda[2].abs() < 1e-12 && cc.lambda[3].abs() < 1e-12);
    assert!(cc.entropy.abs() < 1e-12);
    // Σλ == M00 normalization self-check (catches Pauli-scale errors):
    assert!((cc.lambda.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // ideal depolarizer diag(1,0,0,0): DI = 0, maximal entropy 1
    let mut dep = [[0.0; 4]; 4]; dep[0][0] = 1.0;
    assert!(depolarization_index(&dep).unwrap().abs() < 1e-15);
    let cd = cloude(&dep).unwrap();
    assert!((cd.entropy - 1.0).abs() < 1e-12);
    // guards: zero matrix -> None everywhere
    let z = [[0.0; 4]; 4];
    assert!(depolarization_index(&z).is_none() && cloude(&z).is_none());
}
```
- **G11c end-to-end:** `cloude`/`DI` of `stack.solve()["M_trans"]` on the
  lossless slab == non-depolarizing (DI = 1 ± 1e-9) — physics sanity that
  coherent Berreman outputs stay Mueller-Jones.

### 5.6 Risks

- Pauli-order mismatch is the classic silent bug: mitigated by the spike
  requirement + G11a fixtures generated *by live itself*. (Further defused as
  built: eigenvalues are invariant under simultaneous basis permutation and
  we return sorted values — order can only bite per-component vectors.)
- ~~`eig4` on Hermitian coherency~~ SUPERSEDED: the general quartic solver is
  the WRONG tool for a Hermitian eigenproblem — G11b's ideal depolarizer
  (4-fold λ = 1/4) failed at implementation with complex fuzz ~1e-4 ≫ any sane
  threshold. Replaced by `eig_hermitian4` (cyclic Jacobi, values-only,
  degenerate-safe); the §5.3 draft's `eig4` call + `im_max` arm are tombstoned
  above. Lesson: polynomial root-finding near multiple roots is
  ill-conditioned as ε^{1/4} — no threshold choice fixes it.
---

## Phase 6 — Tensor rotations in Rust (new `src/rotations.rs`)

### 6.1 Why / upstream reference

- Today: one Python helper `rot_z` (wraps `Rz·ε·Rzᵀ` in NumPy).
- Upstream: `pyllama.Layer.rotate_permittivity / rotate_tensor`
  (axis-parameterized), `rot_mat(axis, theta)` (arbitrary axis!),
  `dielectric_tensor.py::euler_rotation_matrix / quaternion_rotation_matrix /
  rotate_2D_tensor / rotate_vector`, `berreman_mueller.rotate_rank2_tensor*`.
  Cholesteric/twisted builders all funnel through these.
- Moving rotations into Rust gives the Python twisted/graded builders
  (Phase 7 §C) a single tested primitive and removes NumPy from tensor prep
  in hot loops (e.g. building 200-slice cholesterics per wavelength).

### 6.2 Module

```rust
//! Real 3x3 rotations applied to complex 3x3 tensors: R·ε·Rᵀ.
//! Port of pyllama rot_mat / dielectric_tensor euler+quaternion helpers.

use crate::berreman::Tensor3;
use crate::cmatrix::{c, C};

pub type Mat3R = [[f64; 3]; 3];

fn ident3() -> Mat3R {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn mat3_mul(a: &Mat3R, b: &Mat3R) -> Mat3R {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}

/// Rodrigues rotation about `axis` by `theta` (== pyllama rot_mat).
/// R = I + sinθ·K + (1−cosθ)·K², K = skew(axis). Axis normalized defensively.
///
/// RESOLVED (implementation): live rot_mat RAISES on zero axis
/// (pyllama.py:58-59), so the wrapper raises ValueError too; the Rust
/// identity fallback below is unreachable-via-API (pinned in unit test).
pub fn axis_angle(axis: [f64; 3], theta: f64) -> Mat3R {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if n < 1e-300 {
        return ident3();
    }
    let (x, y, z) = (axis[0] / n, axis[1] / n, axis[2] / n);
    let (s, c) = (theta.sin(), theta.cos());
    let k: Mat3R = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]];
    let k2 = mat3_mul(&k, &k);
    let mut r = ident3();
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] += s * k[i][j] + (1.0 - c) * k2[i][j];
        }
    }
    r
}

/// Euler rotation == dielectric_tensor.euler_rotation_matrix(rx, ry, rz):
/// R = Rx·Ry·Rz (live line 323; "z, then y, then x" = application order).
///
/// RESOLVED (implementation): the drafted Rz·Ry·Rx expansion was WRONG —
/// source-read before coding showed live computes np.dot(Rx, dot(Ry, Rz)).
/// Built as a visible mat3_mul chain so the order stays reviewable; G12a
/// confirms at 8.9e-16.
pub fn euler_zyx(rx: f64, ry: f64, rz: f64) -> Mat3R {
    let (sx, cx) = (rx.sin(), rx.cos());
    let (sy, cy) = (ry.sin(), ry.cos());
    let (sz, cz) = (rz.sin(), rz.cos());
    [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
}

/// Unit quaternion [w, x, y, z] -> rotation (== quaternion_rotation_matrix).
/// Non-unit input normalized; zero norm => identity.
pub fn quaternion(w: f64, x: f64, y: f64, z: f64) -> Mat3R {
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if n < 1e-300 {
        return ident3();
    }
    let (w, x, y, z) = (w / n, x / n, y / n, z / n);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// Transpose (= inverse for rotations); needed by G12c composition checks
/// and any R·ε·Rᵀ caller that holds R instead of pre-transposing.
#[allow(dead_code)] // used by tests + future twisted builders
pub fn transpose3(r: &Mat3R) -> Mat3R {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = r[j][i];
        }
    }
    o
}

/// Apply: out[i][j] = Σ_kl R[i][k] eps[k][l] R[j][l].
pub fn apply_rot(r: &Mat3R, eps: &Tensor3) -> Tensor3 {
    let mut out = [[c(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = c(0.0, 0.0);
            for k in 0..3 {
                for l in 0..3 {
                    s += c(r[i][k], 0.0) * eps[k][l] * c(r[j][l], 0.0);
                }
            }
            out[i][j] = s;
        }
    }
    out
}
```

pybind — one generic core + three thin constructors (flat-9 re/im tensor
layout identical to `eps` marshalling, so the wrapper reuses `_flatten_opt`
style helpers in reverse):

```rust
// pybind.rs
use crate::rotations::{self, Mat3R};

/// Apply rotation R to a 3x3 complex tensor given as flat re/im length-9
/// row-major; returns flat length-18? NO — keep it simple: re (9) + im (9)
/// as a (2, 9)? Existing convention is interleaved pairs; reuse read_tensor
/// on an 18-slice and return a Vec<f64> len 18 the same way.
fn rot_apply(r: &Mat3R, t_re: &[f64], t_im: &[f64]) -> PyResult<Vec<f64>> {
    if t_re.len() != 9 || t_im.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tensor re/im must each have length 9 (row-major 3x3)",
        ));
    }
    let mut t = [[crate::cmatrix::czero(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = crate::cmatrix::c(t_re[i * 3 + j], t_im[i * 3 + j]);
        }
    }
    let o = rotations::apply_rot(r, &t);
    let mut flat = vec![0.0; 18];
    for i in 0..3 {
        for j in 0..3 {
            flat[(i * 3 + j) * 2] = o[i][j].re;
            flat[(i * 3 + j) * 2 + 1] = o[i][j].im;
        }
    }
    Ok(flat)
}

#[pyfunction] #[pyo3(name = "rot_axis_angle", signature = (ax, ay, az, theta, t_re, t_im))]
pub fn rot_axis_angle_py(
    _py: Python<'_>, ax: f64, ay: f64, az: f64, theta: f64,
    t_re: PyReadonlyArray1<f64>, t_im: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    let r = rotations::axis_angle([ax, ay, az], theta);
    rot_apply(&r, t_re.as_slice()?, t_im.as_slice()?)
}
// + rot_euler_py(rx, ry, rz, t_re, t_im), rot_quaternion_py(w, x, y, z, t_re, t_im).
// Register all three in lib.rs::_berreman.
```

Wrapper (`navette/berreman.py`) — single private choke point, public thin API:

```python
from ._berreman import rot_axis_angle as _rs_rot_aa, rot_euler as _rs_rot_eu, rot_quaternion as _rs_rot_q

def _rot_apply(rs_fn, *angle_args, eps):
    e = np.asarray(eps, dtype=complex)
    if e.shape != (3, 3):
        raise ValueError(f"eps must be (3,3), got {e.shape}")
    flat = np.asarray(rs_fn(*angle_args, e.real.reshape(9).copy(), e.imag.reshape(9).copy()),
                      dtype=float).reshape(9, 2)
    return flat[:, 0].reshape(3, 3) + 1j * flat[:, 1].reshape(3, 3)

def rot_axis(axis, theta_rad, eps):
    """Rotate tensor about arbitrary axis (Rodrigues; == pyllama rot_mat)."""
    ax = np.atleast_1d(np.asarray(axis, dtype=float))
    if ax.shape != (3,):
        raise ValueError("axis must have 3 components")
    return _rot_apply(_rs_rot_aa, float(ax[0]), float(ax[1]), float(ax[2]),
                      float(theta_rad), eps=eps)

def rot_euler(rx_rad, ry_rad, rz_rad, eps):
    """Extrinsic Z-Y-X Euler (== dielectric_tensor.euler_rotation_matrix)."""
    return _rot_apply(_rs_rot_eu, float(rx_rad), float(ry_rad), float(rz_rad), eps=eps)

def rot_quat(w, x, y, z, eps):
    return _rot_apply(_rs_rot_q, float(w), float(x), float(y), float(z), eps=eps)

def rot_z(eps, angle_rad):
    """Kept name/signature; now delegates to the Rust axis-angle path."""
    return rot_axis([0.0, 0.0, 1.0], angle_rad, eps)
```

G12b parity procedure (AS BUILT — two corrections to the draft):
1. No identity-tensor probe: R·I·Rᵀ = I for every R, vacuous (drafted probe
   removed before first run). The pin is R·E·Rᵀ over random complex E
   (fixes R up to sign) + Rust det=+1 (kills −R).
2. rot_z delegation bound is ≤ 1e-15, not 0.0 (measured 4.4e-16): Rodrigues
   vs closed-form Rz differ by 1 ulp in op order. The live G12a arm is the
   real pin; G12b guards the delegation against regressions.
The OLD NumPy body is inlined in the test file; `berreman.py` carries only
the delegation.

### 6.3 Tests & gates (new gate G12)

- **G12a vs live** (`tests/test_rotations.py`):

```python
import numpy as np
from navette import berreman as bl
from berremueller import pyllama as live_pl
from berremueller import dielectric_tensor as live_dt

rng = np.random.default_rng(11)
worst = 0.0
for _ in range(20):
    ax = rng.normal(size=3); th = rng.uniform(-np.pi, np.pi)
    # rotation matrices: live rot_mat returns ndarray; compare elementwise
    R_live = np.asarray(live_pl.rot_mat(axis=ax, theta_rad=th), dtype=float)
    # Rust path via identity tensor probe: rot_axis(I) == R. agnostic probe:
    R_rs = np.asarray(bl.rot_axis(ax, th, np.eye(3)), dtype=float).real
    worst = max(worst, np.max(np.abs(R_rs - R_live)))
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    Rq_live = np.asarray(live_dt.quaternion_rotation_matrix(q), dtype=float)
    Rq_rs = np.asarray(bl.rot_quat(*q, np.eye(3)), dtype=float).real
    worst = max(worst, np.max(np.abs(Rq_rs - Rq_live)))
    rx, ry, rz = rng.uniform(-np.pi, np.pi, size=3)
    Re_live = np.asarray(live_dt.euler_rotation_matrix(rx, ry, rz), dtype=float)
    Re_rs = np.asarray(bl.rot_euler(rx, ry, rz, np.eye(3)), dtype=float).real
    worst = max(worst, np.max(np.abs(Re_rs - Re_live)))
    # rotated random COMPLEX tensors (catches R·ε·Rᵀ conjugation slips)
    E = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    worst = max(worst, np.max(np.abs(bl.rot_axis(ax, th, E) - R_live @ E @ R_live.T)))
print(f"rotations vs live worst: {worst:.3e}")
assert worst < 1e-14
```

  Note: `live_dt.euler_rotation_matrix` arg order `(r_x, r_y, r_z)` per its
  docstring — if G12a fails ONLY on Euler, the order hypothesis in §6.2 is
  wrong; fix `euler_zyx` multiplication order (try `Rx·Ry·Rz`) before touching
  anything else. The test failure itself diagnoses the bug — that is why the
  Euler case is isolated per-iteration (report per-family worst separately).
- **G12b `rot_z` parity:** old NumPy body (inlined copy in the test) vs new
  Rust-backed `rot_z` on `diag(2.25,2.89,2.25)` at 0/15/35/90° → ≤ 1e-15
  (observed 4.4e-16; 1-ulp op-order divergence, see above). Plus degenerate
  rejections (zero axis / zero-norm quat → ValueError, like live).
- **G12c group property** (pure Rust, `rotations.rs::tests`):

```rust
#[test]
fn rotation_group_properties() {
    // R·Rᵀ = I and det = +1 for all three constructors (proper rotations)
    let cases: Vec<Mat3R> = vec![
        axis_angle([1.0, 2.0, 3.0], 0.7),
        euler_zyx(0.3, -0.5, 1.1),
        quaternion(0.5, 0.5, 0.5, 0.5),
    ];
    for r in &cases {
        let rt = transpose3(r);
        let should_be_i = mat3_mul(r, &rt);
        for i in 0..3 { for j in 0..3 {
            let e = if i == j { 1.0 } else { 0.0 };
            assert!((should_be_i[i][j] - e).abs() < 1e-14);
        }}
        // det via scalar triple product of rows
        let det = r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
                - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
                + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]);
        assert!((det - 1.0).abs() < 1e-14);
    }
    // composition: axis_angle z-rot then x-rot == euler(rx, 0, rz)
    let a = axis_angle([0.0, 0.0, 1.0], 0.4);
    let b = axis_angle([1.0, 0.0, 0.0], 0.9);
    let comp = mat3_mul(&b, &a); // apply a first, then b
    let eu = euler_zyx(0.9, 0.0, 0.4);
    for i in 0..3 { for j in 0..3 { assert!((comp[i][j]-eu[i][j]).abs() < 1e-14); } }
}
```
---

## Phase 7 — Validation harness, Python helpers, release

### 7.1 Extended validation (Rust + Python)

**A. `examples/validate.rs` extensions** (one per phase, additive flags;
default invocation byte-identical to today so `ref/diff_check.py` keeps
passing unmodified):

```rust
// examples/validate.rs — arg parsing (hand-rolled, no new deps; keep MSRV):
struct Flags {
    methods: Vec<Method>,        // default: [Scattering, Transfer]
    periods: u32,                // default: 1  (Phase 4)
    exit_eps: Option<Tensor3>,   // default: None (Phase 1; parsed from 18 floats)
    z_fields: Vec<f64>,          // default: []  (Phase 2; depths in nm)
}

fn parse_args() -> Flags {
    let mut f = Flags { methods: vec![Method::Scattering, Method::Transfer],
                        periods: 1, exit_eps: None, z_fields: vec![] };
    let mut it = std::env::args().skip(1).peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--method" => {
                let m = it.next().expect("--method needs sm|tm|em");
                f.methods = m.split(',').map(|s| match s {
                    "sm" => Method::Scattering,
                    "tm" => Method::Transfer,
                    "em" => Method::Exponential,   // Phase 3 arm
                    _ => panic!("unknown method {s}"),
                }).collect();
            }
            "--periods" => { f.periods = it.next().expect("N").parse().expect("u32"); }
            "--aniso-exit" => {
                // 18 comma-separated floats: 9 (re,im) pairs row-major
                let s = it.next().expect("18 floats");
                let v: Vec<f64> = s.split(',').map(|x| x.parse().expect("f64")).collect();
                assert_eq!(v.len(), 18, "need 9 (re,im) pairs");
                let mut t = [[crate::cmatrix::czero(); 3]; 3];
                for i in 0..3 { for j in 0..3 {
                    t[i][j] = crate::cmatrix::c(v[(i*3+j)*2], v[(i*3+j)*2+1]);
                }}
                f.exit_eps = Some(t);
            }
            "--fields" => {
                f.z_fields = it.next().expect("z list").split(',')
                    .map(|x| x.parse().expect("f64")).collect();
            }
            _ => panic!("unknown flag {a}"),
        }
    }
    f
}
```

Emit rules: `--periods N` routes every case through `solve_stack_periodic`
(and tags output `"periods": N`); `--aniso-exit` swaps the exit to
`ExitSpec::Anisotropic` for all cases (tags `"exit": "aniso"`, power dict
gains `"power_method"`); `--fields` adds a `"fields"` key per case with
`psi_to_eh`-unpacked E/H per depth per input pol (reuse `emit()`-style
`cj()` formatting for complex). `ref/diff_check.py` gains `--em`,
`--periods`, `--aniso-exit`, `--fields` modes comparing against
live-generated `ref/ref_out_<mode>.json`, produced by the checked-in
`ref/gen_live_ref.py` (drives live `StackModel` + B44 directly; documents its
own venv requirements at the top; outputs are COMMITTED so CI needs no
network — regenerate only when references change).

**B. New Python tests** (mirror existing style: oracle compare + print worst):

| File | Covers | Gate |
|------|--------|------|
| `tests/test_aniso_exit.py` | G7b (B44 back: Jr+R direct, T via B44 flux) + G7d (energy) + G9d (EM vs B44) | G7 |
| `tests/test_fields.py` | G8a–e (amplitude bridge, continuity, asymptotes, live in-plane, absorption integral) | G8 |
| `tests/test_em.py` | G9a–c (expm unit in Rust, EM≡TM, live EM); G9d arm in test_aniso_exit.py | G9 |
| `tests/test_periods.py` | G10a–d (algebra, expanded≡periodic, live N_per, bench) | G10 |
| `tests/test_mueller_suite.py` | G11a–c (live metrics, identities, e2e DI) | G11 |
| `tests/test_rotations.py` | G12a–c (live parity, rot_z parity, group) | G12 |
| `tests/test_roughness_energy.py` | G13a (energy all codes × σ, no smatrix) | G13 |
| `tests/test_graded.py` | G13b–e (Route-2 pins + Route-1/Route-2 cross-check) | G13 |
| `tests/test_materials.py` | G14a–c (live Tier-1 parity, tensor identities, e2e) | G14 |
| `tests/test_materials.py` (KK arms) | Cody/Tauc/UBF live parity incl. in G14a (9b dissolved) | G14 |
| `tests/test_chiral.py` | G15a–d (Pasteur spectrum/slab/rotation + cholesteric Bragg) | G15 |

Note: `tests/test_roughness.py` keeps codes 0–4 parity vs smatrix (G5) but its
code-5 arm changes meaning after the §8.3 fix — see §8.6 (reflection still
matches; transmission now *differs by design*, asserted as a fork-guard).

All tolerance asserts follow the existing pattern (`worst < tol` + printed
`worst` value in the test log for trend-spotting). Worked example —
`tests/test_em.py` skeleton (copy-paste starter; same shape for the others):

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../ref")
"""EM engine (Method::Exponential) vs TM, live EM, and B44. Gates G9b-d."""
import numpy as np
from navette import berreman as bl
import ref_pyllama as ref

CASES = [(e, t, wl, deg) for (e, t, wl, deg) in [
    (np.diag([2.25]*3).astype(complex), 200.0, 550.0, 0.0),
    (bl.rot_z(np.diag([2.25, 2.89, 2.25]).astype(complex), np.radians(35.0)), 250.0, 633.0, 30.0),
]]

def upto_phase(a, b):
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    k = int(np.argmax(np.abs(b)))
    ph = (b[k] / a[k]); ph /= abs(ph)
    return np.max(np.abs(a * ph - b))

worst = 0.0
for eps, thick, wl, deg in CASES:
    r_em = bl.BerremanStack([bl.Layer(eps, thick)], [wl], [deg], method="exponential").solve()
    r_tm = bl.BerremanStack([bl.Layer(eps, thick)], [wl], [deg], method="transfer").solve()
    r = ref.solve([(eps, thick)], 1.0, 1.0, wl, np.radians(deg), method="TM")
    worst = max(worst, np.max(np.abs(r_em["R"] - r_tm["R"])),
                       np.max(np.abs(r_em["T"] - r_tm["T"])),
                       upto_phase(r_em["J_refl"], r_tm["J_refl"]))
    # G9c live-EM arm (import live StackModel with method="EM" here; ≤ 1e-9)
    # G9d B44 arm (Berreman4x4 getJones; ≤ 1e-9)
print(f"EM vs TM worst: {worst:.3e}")
assert worst < 1e-12, "G9b failed"
print("EM PASS")
```
(As built, `tests/test_em.py` extends this starter: all 5 canonical cases +
the gyrotropic full-path arm for G9b, and live-`Structure` construction with
normalized wavevectors for G9c.)

**C. Pure-Python helpers (no Rust; specified here for completeness)**

```python
# navette/berreman.py additions — thin builders over Layer/BerremanStack.
# All use the Rust-backed rot_axis (Phase 6); no new compiled code needed.

def twisted_stack(eps_uniaxial, total_nm, twist_rad, n_slices, grid="midpoint"):
    """Discretized in-plane twist (cf. B44 TwistedMaterial; pyllama's
    CholestericModel).
    eps_uniaxial: (3,3) tensor at zero twist; slice k carries
      rot_z(eps, twist*(k+0.5)/n) for grid="midpoint" (default, 2nd-order
      accurate in slice thickness) or twist*k/(n-1) for grid="edge".
    Returns list[Layer] with equal thicknesses summing to total_nm."""
    e0 = np.asarray(eps_uniaxial, dtype=complex)
    if e0.shape != (3, 3):
        raise ValueError(f"eps_uniaxial must be (3,3), got {e0.shape}")
    if n_slices < 1:
        raise ValueError("n_slices >= 1")
    dz = float(total_nm) / n_slices
    out = []
    for k in range(n_slices):
        frac = (k + 0.5) / n_slices if grid == "midpoint" else k / max(n_slices - 1, 1)
        out.append(Layer(rot_axis([0.0, 0.0, 1.0], twist_rad * frac, e0), dz))
    return out

def cholesteric_stack(n_o, n_e, pitch_nm, n_periods, n_per_pitch=20):
    """Full-pitch helix: uniaxial tensor rotating 2π per pitch (cf.
    cholesteric.py / CholestericModel). Director starts along x."""
    e0 = np.diag([n_e**2, n_o**2, n_o**2]).astype(complex)
    return twisted_stack(e0, pitch_nm * n_periods, 2 * np.pi * n_periods,
                         n_periods * n_per_pitch)

def interp_eps_stack(eps_set, wl_set, target_wl):
    """Per-element linear interpolation of a dispersive tensor set to
    target_wl (cf. extract_tensor_wl_interpolated / update_for_spectrum).
    eps_set: (n,3,3), wl_set: (n,) ascending; target outside range clamps
    (documented; live extrapolates — clamping is the safer default)."""
    E = np.asarray(eps_set, dtype=complex)
    W = np.atleast_1d(np.asarray(wl_set, dtype=float))
    if E.shape[0] != W.shape[0] or E.shape[1:] != (3, 3):
        raise ValueError(f"eps_set (n,3,3) vs wl_set (n,): {E.shape} vs {W.shape}")
    t = float(np.atleast_1d(np.asarray(target_wl, dtype=float))[0])
    if t <= W[0]:
        return E[0].copy()
    if t >= W[-1]:
        return E[-1].copy()
    k = int(np.searchsorted(W, t)) - 1  # W[k] <= t < W[k+1]
    f = (t - W[k]) / (W[k + 1] - W[k])
    return (1 - f) * E[k] + f * E[k + 1]

def ellipsometry(jones_refl):
    """Psi/Delta (radians) from a 2x2 Jones reflection matrix
    (cf. B44 DataList.getEllipsometryParameters).
    DEFINITION (documented, Fujiwara p.220 as in B44 getJones docstring):
      rho = r_pp / r_ss  (diagonal ratio; cross-pol ignored, warned if large),
      Psi = atan(|rho|), Delta = arg(rho).
    Returns (psi, delta); NaN pair when |r_ss| ~ 0 (Brewster-adjacent)."""
    import warnings
    J = np.asarray(jones_refl, dtype=complex).reshape(2, 2)
    if abs(J[1, 1]) < 1e-300:
        return (float("nan"), float("nan"))
    cross = max(abs(J[0, 1]), abs(J[1, 0])) / max(abs(J[0, 0]), abs(J[1, 1]), 1e-300)
    if cross > 1e-3:
        warnings.warn(f"ellipsometry ignores cross-pol (ratio {cross:.2e}); "
                      "Psi/Delta are diagonal-only", UserWarning)
    rho = J[0, 0] / J[1, 1]
    return (float(np.arctan(abs(rho))), float(np.angle(rho)))
```

Round-trip tests (10 lines each, in `tests/test_builders.py`):

```python
def test_twist_zero_is_slab():
    e0 = np.diag([2.89, 2.25, 2.25]).astype(complex)
    a = bl.BerremanStack(bl.twisted_stack(e0, 1000.0, 0.0, 8), [550.0], [20.0]).solve()
    b = bl.BerremanStack([bl.Layer(e0, 1000.0)], [550.0], [20.0]).solve()
    assert np.max(np.abs(a["R"] - b["R"])) == 0.0  # identical tensors -> identical solve

def test_interp_knots_and_mid(): ...   # interp at wl_set[k] == eps_set[k] exactly;
                                       # midpoint == mean (linear)
def test_ellipsometry_fresnel(): ...   # air/glass interface: Psi == atan(|rp/rs|)
                                       # vs analytic (tolerance 1e-12)
```

### 7.2 Docs & release checklist (per phase PR)

1. `README.md`: new API section + method table (`scattering|transfer|
   exponential`, `periods`, `exit_eps`, `fields()`, metrics) and the
   intentional-deviation note stays accurate (extend to anisotropic-exit power
   semantics: Jones-factor vs flux).
2. `Cargo.lock` pins (`rustc-hash 2.0.0`, `rayon 1.10.0/1.12.1`) — re-verify
   after adding modules (no new deps expected; `rotations`/`mueller`/`expm`/
   `fields` are `num-complex`-only like the rest of the core).
3. MSRV note: core stays buildable on rustc 1.75 (`cargo test` without
   `--release` on the pinned toolchain if available); pyo3 `allow_threads`
   idiom unchanged.
4. Rebuild + reinstall wheel (`maturin develop --release`), rerun **all**
   gates G1–G12, paste the gate table with observed values into the PR.
5. Version bump policy: minor per phase (0.2.0 …), patch for gate fixes.

### 7.3 Suggested implementation order & dependencies

```text
Phase 1 (aniso exit) ──┐
                       ├──> Phase 2 needs §2.3 amplitudes (write helper in P2, backport use in P1)
Phase 3 (EM) ──────────┤ independent; can precede P1 (only touches Method + transfer fns)
Phase 4 (periods) ─────┘ needs s_combine pub(crate) (1 line) — any time after P0
Phase 6 (rotations) ──── independent, smallest; good first PR to shake out CI
Phase 5 (mueller) ────── needs only the §5.2 spike; independent
Phase 2 (fields) ─────── largest; schedule after P1 (shares amplitude helper + exit-b handling)
Phase 7 ──────────────── continuous (each phase ships its tests + docs)
```

Recommended PR sequence: **P8 → P9 → P6 → P3 → P1 → P4 → P5 → P2 → P10**, validating after each.
(P8 first: bug fix with no dependencies. P9 second: independent of solver phases
(9b dissolved — KK trio ships inside P9). P10 last: 10a may front-run anytime
after P8 (independent); 10b needs the P7 twist helper.)

---

---

## Phase 8 — Roughness correctness (type-5 fix, Route-2 coverage, validation)

### 8.0 Priority: do this FIRST

Unlike Phases 1–7 (new features), this phase fixes a **confirmed physics bug**
in shipped code and closes the validation holes that pin the bug in place. It
touches only `src/roughness.rs`, the wrapper's roughness handling, docs, and
tests — no API breakage except the intended type-5 transmission change (§8.6).
Size S. Gate: new **G13** (plus a re-scoping of G5's code-5 arm).

### 8.1 Upstream finding (Navette `dev`, `docs/code_review.md` §3.2 🔴 CONFIRMED BUG)

Quoted from the upstream review (verified present in `dev @ 4b59cd7`, the
commit audited):

- Névot–Croce (type 5) applies the **reflection** factor `f = exp(−2·kz1·kz2·σ²)`
  to **transmission** as well: `(r12·f, r21·f, t12·f, t21·f)` at
  `rust/navette/src/smatrix/coherent_block.rs:128-134` (second copy
  `:319-325`), `rust/navette/src/smatrix/solver.rs:1686-1694`,
  `rust/navette/src/smatrix/needle_operator.rs:142-148`.
- Measured damage: **R+T = 0.925** (7.5% of energy destroyed) at a *single
  almost-invisible interface*, σ=10 nm; severity grows with σ and n-contrast.
- Prescribed fix: `t12·ga, t21·ga` with `ga = exp(−(kz1−kz2)²σ²/2)` — i.e. the
  Gaussian single-argument factor at the wavevector transfer (exactly
  `w_function_inner((kz1−kz2)·σ, 4)`).
- If upstream fixes it first, our G13a stays green regardless; only §8.6's
  fork-guard magnitudes need re-baselining (reflection factor is identical
  either way, so the G5 reflection arm never moves).

Related upstream notes incorporated ниже (spelled out to avoid re-derivation):

- 🟡 Model choice (`code_review.md:144`): types 1–4 apply **per-side** factors
  (`r12·W(2kz1σ)`, `r21·W(2kz2σ)`, `t·W((kz1−kz2)σ)` — the "graded transition
  layer" interpretation, R+T ≈ 0.995 at σ=10 nm) vs NC's *correlated*
  `exp(−2kz1kz2σ²)` for reflection. The two disagree substantially at large σ;
  upstream asks for a doc note — we add ours in §8.11.
- Test gap (`code_review.md:266`): parity tests compare against a reference
  *with the same bug*; no energy-conservation test exists. Our suite has the
  identical hole (test_roughness.py pins 2.3e-14 vs smatrix) — closed by G13a.
- Coverage gap (`code_review.md:450`): types 1–4 spot-checked only at normal
  incidence, single interface, vs analytic graded reference. Our 4-angle
  sweep is better, but still same-model parity — closed by G13e (Route-1 vs
  Route-2 on anisotropic stacks).

### 8.2 Root cause in our code

`src/roughness.rs::interface_factors`, type-5 arm. The per-channel
`(ka, kb)` selection was engineered so every channel collapses to `+kz1·kz2`
in the isotropic limit (`qa = qb = [+q,+q,−q,−q]`):

| channel (i,j) | selection | isotropic product | upstream |
|---|---|---|---|
| T fwd a→b (T,T) | `(kza[j], kzb[i])` | `(+kz1)(+kz2)` | `f` — bug-compatible ❌ |
| R front (F,T) | `(kza[j], kzb[j])`, j<2 | `(+kz1)(+kz2)` | `f` ✓ |
| R back (T,F) | `(kza[i], kzb[i])`, i<2 | `(+kz1)(+kz2)` | `f` ✓ |
| T bwd b→a (F,F) | `(kza[i], kzb[j])` | `(−kz1)(−kz2)` | `f` — bug-compatible ❌ |

Reflection rows are correct and stay untouched. Transmission rows must use
the Gaussian transfer factor (§8.3).

### 8.3 The fix (Rust, `src/roughness.rs`)

Replace the type-5 arm wholesale (reflection arms simplify to a `j<2` test —
verify equivalence against the table in §8.2 during review):

```rust
for i in 0..4 {
    for j in 0..4 {
        if rough_type == 5 {
            match (i < 2, j < 2) {
                (true, true) | (false, false) => {
                    // TRANSMISSION (forward a→b / backward b→a): Nevot–Croce
                    // theory gives the Gaussian transfer factor
                    //   ga = exp(−(Δkz)²σ²/2) = W_4((kz_in − kz_out)·σ),
                    // NOT the reflection cross factor. Upstream applies f
                    // to t (bug, §8.1); we diverge here by design (§8.6).
                    // Isotropic check: (T,T): (+kz1)−(+kz2); (F,F):
                    // (−kz2)−(−kz1) = same Δkz → ga on both. ✓
                    let kz_out = if i < 2 { kzb[i] } else { kza[i] };
                    let kz_in = if j < 2 { kza[j] } else { kzb[j] };
                    w[i][j] = w_function((kz_in - kz_out) * c(sigma, 0.0), 4);
                }
                _ => {
                    // REFLECTION (front a-side / back b-side): NC cross
                    // factor exp(−2·ka·kb·σ²), mode-pair selection collapses
                    // to +kz1·kz2 in the isotropic limit (unchanged).
                    // (F,T): j<2 → (kza[j], kzb[j]); (T,F): j≥2 → (kza[i], kzb[i]).
                    let (ka, kb) = if j < 2 { (kza[j], kzb[j]) } else { (kza[i], kzb[i]) };
                    w[i][j] = (-c(2.0, 0.0) * ka * kb * s2).exp();
                }
            }
        } else {
            // types 0–4 path unchanged
            let kz_out = if i < 2 { kzb[i] } else { kza[i] };
            let kz_in = if j < 2 { kza[j] } else { kzb[j] };
            w[i][j] = w_function((kz_in - kz_out) * c(sigma, 0.0), rough_type);
        }
    }
}
```

Module-docstring table update (same edit — the old row documents the bug as
intended behavior):

```text
//! | 5    | Névot–Croce (Gaussian cross)  | exp(−2·ka·kb·σ²) on REFLECTION; |
//! |      |                               | exp(−(Δkz)²σ²/2) (= W_4) on      |
//! |      |                               | TRANSMISSION (upstream smatrix  |
//! |      |                               | applies f to t as well — known  |
//! |      |                               | energy bug, see Phase 8 §8.1;   |
//! |      |                               | we diverge by design)          |
```

### 8.4 Anisotropic semantics (README ansatz statement — add verbatim)

> **Route-1 anisotropic factors are an ansatz.** No external
> anisotropic-roughness reference exists (neither Berreman4x4 nor pyllama
> implements roughness), so the per-mode generalization is validated only in
> (a) the isotropic limit — exact reduction to smatrix for codes 0–4 and to
> the *corrected* smatrix semantics for code 5 (§8.3) — and (b) the σ→0 limit
> — exact reduction to smooth for all codes. Reflection uses the NC cross
> factor per mode pair; transmission uses the Gaussian transfer factor of the
> mode-pair wavevector transfer. Treat large-σ anisotropic predictions as
> model-dependent and cross-check against Route 2 (graded sublayers, G13e).

### 8.5 Gate G13a — energy conservation (`tests/test_roughness_energy.py`, no smatrix)

Self-contained physical invariant. Correction (measured during implementation):
this gate does **not** catch the §8.1 bug by itself — the pre-fix build gives
R+T ≈ 0.755 on the three-interface fixture yet still satisfies the ≤ 1 bound
(the bug destroys energy, it never creates it). The bug-catcher is the §8.6
transmission fork-guard (pre/post-fix |ΔT| ≈ 0.11–0.22 on the G5 fixture);
G13a pins the never-create-energy invariant going forward (it would catch
sign/conjugation-class errors that push R+T above 1):

```python
import sys
sys.path.insert(0, "ref")  # like the other tests
"""Gate G13a: specular energy conservation for Route-1 roughness.
Lossless stack => per-input R+T within [-tol, 1+tol] for every code x sigma
x angle. specular-only model => never ABOVE 1 (diffuse loss is unmodeled)."""
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
                f"code {rtype} σ={sig} ang={ang}: R+T={tot} violates energy"
            worst_t = max(worst_t, np.max(np.abs(tot - np.minimum(tot, 1.0))))
            worst_over = max(worst_over, float(np.max(1.0 - tot)))
    worst_deficit[rtype] = worst_t
    print(f"code {rtype}: max specular deficit 1-(R+T) = {worst_t:.3e} (over σ×angle)")
print(f"WORST deficit overall: {worst_over:.3e}")
print("ENERGY PASS")
# Expected signature (documents the §8.1 model distinction, cf. review:144):
# codes 1-4 deficit ~5e-3 @ σ=10nm (graded-profile-like); fixed code 5 <= 1
# everywhere (pre-fix measured R+T ≈ 0.755 — still ≤ 1, so this gate alone
# does not catch the bug; see the §8.5 correction and the §8.6 fork-guard).
```

Absorbing arm (same file, second half): absorbing rotated-uniaxial slab with
code 5, σ=10 nm, 3 angles — assert per-input R+T ≤ 1+1e-9 (absorption only
removes energy; roughness must never create it).

### 8.6 G5 update — code-5 transmission intentionally diverges from smatrix

`tests/test_roughness.py` changes (same commit as §8.3 — never land the fix
without this, or CI goes red for the right reason with no record of why):

- Codes 0–4: parity vs smatrix ≤ 1e-9, unchanged (G5 intact).
- Code 5: split the arm. **Upstream 0.7.0 redefined the fork** (audited in
  the .crate source during implementation — this supersedes the §8.1
  bug-only picture): `nevot_croce_factors` returns `(f, ga)` with
  `ga = exp(+((kz1−kz2)·σ)²/2)` — a DELIBERATE growing transmission factor
  (`optics_core.rs` docstring: perturbative R+T=1 via reflected-loss /
  transmitted-gain cancellation, X-ray-contrast origin, σ budget
  ~0.0159·λ/Δn; their own docs show R+T up to 49.4). Against NC theory
  (transmission must decay under roughness decorrelation) this is unphysical
  outside its validity range; we keep the textbook decaying `W_4` and diverge
  by design — now with a precise, documented model difference instead of a
  presumed typo. Rust pin: `ga_up · ga_theory == 1` exactly
  (`tests/navette_parity.rs`; fails loudly if upstream ever fixes the sign).
  **Reflection: formula parity is proven at Rust level** (our R-block ==
  upstream `f` at 1e-15 on lossy kz) — this is the real proof the fix touched
  transmission only. The Python R arm (reverberation-decoupled fixture: one
  400 nm n = 2.0+1.5j film, residue ~1e-12) is system-level sanity at
  ≤ 1e-6 (measured 1.4e-8): stack R mixes T back in through reverb, and the
  fix shifts two-film R by ~4e-3 with a bit-identical R-block, so stack-level
  R can never be a 1e-9 gate. **Transmission asserts DIVERGENCE**: `|ΔT| >
  1e-6` at σ ≥ 5 nm on the standard lossless two-film fixture (measured vs
  live 0.7.0 wheel: min |ΔT| = 1.75e-2, four orders of margin; pre/post-fix
  |ΔT| ≈ 0.11–0.22 vs the pre-fix 2.3e-14 parity). Comment cites §8.1 +
  upstream `code_review.md` §3.2 + the 0.7.0 `optics_core.rs` docstring.
  OPEN (mechanism unidentified, evidence filed): against the 0.7.0 wheel a
  1.4e-8 R residual remains on lossy media scaling exactly as σ² — while
  crate-level formula parity is 1e-15 and codes 1/4 match the same fixture
  at 1e-14. Suspect: wheel-build vs .crate-source drift in NC handling, or
  solver-level NC treatment of lossy kz. Does not gate anything (100× margin
  in the sanity arm, exact pin in Rust).
- README "intentional deviations" section gains a second entry (after the
det/division note): type-5 transmission uses `ga`, upstream uses `f` (energy
  bug, measured 7.5% loss); reflection identical; re-converges as σ→0.

### 8.7 Route-2 coverage (G13b–d, `tests/test_graded.py`, no smatrix)

The README's Route-2 claims (σ=0 reduction, SM≡TM ~1e-16, sublayer
convergence) are currently untested. Pin them:

- **G13b σ=0 identity:** `graded_stack(layers, [0]*(n+1))` ≡ base `solve()`
  to exactly `0.0` (zero widths → no sublayers, zero shave → identical
  layers; any nonzero diff is a construction bug, not numerics).
- **G13c method agreement:** graded anisotropic stack (rotated-uniaxial
  σ=8 nm interface, `n_sublayers=9`) SM vs TM R/T/Jones ≤ 1e-12 (README
  claims 1e-16 — pin 1e-12, print observed; tighten only with data).
- **G13d sublayer convergence:** same stack at `n_sublayers` 67/107/171 —
  assert monotone and 107-vs-171 R/T ≤ 1e-6; print the triple. Correction
  (measured during implementation): convergence on this harsh fixture (high
  contrast, σ up to 12 nm vs 90 nm films) is slow — err only ~2.8× per 1.6×
  refinement (9v15 = 9.2e-5) — so the bound needs large counts. Solves stay
  milliseconds; the triple pins convergence, not production practice (~9).

```python
def solve_graded(sigmas, n_sub, method):
    eps1 = np.diag([2.25, 2.89, 2.25]).astype(complex)  # anisotropic fixture
    eps2 = np.diag([2.56]*3).astype(complex)
    base = [bl.Layer(eps1, 120.0), bl.Layer(eps2, 90.0)]
    g = bl.graded_stack(base, sigmas, n_entry=1.0, n_exit=1.5, n_sublayers=n_sub)
    return bl.BerremanStack(g, [550.0], [30.0], n_entry=1.0, n_exit=1.5,
                            method=method).solve()
# G13b: solve_graded([0,0,0], 9, m) vs base solve -> 0.0 for m in (sm, tm)
# G13c: solve_graded([8,5,12], 9, "scattering") vs (... , "transfer") -> <=1e-12
# G13d: R at n_sub 5/9/15 -> monotone + 9v15 <= 1e-6
```

### 8.8 Route-1 vs Route-2 cross-check (G13e, same file)

Only independent check available for anisotropic roughness (no external
reference implements it). Same physical interface, two approximations
(perturbative W-factors vs erf-graded sublayers) — expect convergence as
σ→0, NOT equality (state this in the test docstring so nobody tightens it
into a false failure):

```python
# G13e: rotated-uniaxial slab, single rough entry interface, code 4 (Gaussian
# W is the closest Route-1 analogue of the erf-graded Route-2 profile).
for sig in (2.0, 1.0, 0.5):
    r1 = bl.BerremanStack([bl.Layer(eps_aniso, 250.0, roughness=(4, sig))],
                          [550.0], [30.0], method="scattering").solve()
    g = bl.graded_stack([bl.Layer(eps_aniso, 250.0)], [sig, 0.0],
                        n_sublayers=15)
    r2 = bl.BerremanStack(g, [550.0], [30.0], method="scattering").solve()
    err = max(np.max(np.abs(r1["R"]-r2["R"])), np.max(np.abs(r1["T"]-r2["T"])))
# assert err(0.5) < err(1.0) < err(2.0) (converging) and err(2.0) <= 5e-2
# (absolute bound documents model spread, not solver error).
```

### 8.9 Wrapper validation (`navette/berreman.py`)

One normalizer fixes three warts (§8.9.1–3): unknown codes silent, negative σ
divergence from upstream, `(0,·)`/`(·,0)` blocking transfer. Behavior-preserving
(Rust already returns identity for both degenerate cases — normalization only
changes `rtypes` zeros-to-zeros and unblocks the method gate):

```python
_ROUGH_TYPES = frozenset((0, 1, 2, 3, 4, 5))

def _norm_rough(spec, where):
    """Validate + normalize one (type, sigma_nm) spec. None/degenerate → None."""
    if spec is None:
        return None
    t, s = int(spec[0]), float(spec[1])
    if t not in _ROUGH_TYPES:
        raise ValueError(f"{where}: unknown roughness type {t} (valid 0–5)")
    if not np.isfinite(s) or s < 0:
        raise ValueError(f"{where}: sigma_nm must be finite and >= 0, got {spec[1]}")
    if t == 0 or s == 0.0:
        return None  # smooth: Rust identity either way; unblocks transfer
    return (t, s)
```

Integration (in `BerremanStack.__init__` + `solve`, no mutation of user Layers):

```python
    # __init__: normalize per-layer fronts + exit (parallel arrays)
    self._rough_norm = [_norm_rough(l.roughness, f"layers[{i}].roughness")
                        for i, l in enumerate(self.layers)]
    self._exit_rough_norm = _norm_rough(exit_roughness, "exit_roughness")
    self._has_rough = self._exit_rough_norm is not None or \
        any(r is not None for r in self._rough_norm)
    # ... method==1 gate unchanged (now fires only on real dressing)
    # solve(): build rtypes/rvals from self._rough_norm/_exit_rough_norm
    # instead of l.roughness/exit_roughness (identical values otherwise).
```

### 8.10 `graded_stack` / `grade_interface` silent behaviors

- **Thin-layer clamp:** `t_new = max(thickness − shave, 1e-9)` fires silently
  when 3σ exceeds a layer. Add (same commit):

```python
    shave = 0.5 * width(j) + 0.5 * width(j + 1)
    if shave > 0 and lyr.thickness_nm - shave < 1e-9:
        warnings.warn(
            f"graded_stack: layer {j} thickness {lyr.thickness_nm}nm < "
            f"3σ shave {shave:.2f}nm; clamped to 1e-9nm "
            f"(thicken the layer or reduce sigma)", UserWarning)
    t_new = max(lyr.thickness_nm - shave, 1e-9)
```

- **MO tensors dropped in transitions:** `grade_interface` builds plain
  `Layer(eps_eff, dz)` — for MO stacks the graded region silently becomes
  non-magnetic. Add a head-of-function guard in `graded_stack` (not an
  error — error would break the common non-magnetic path; a warning):

```python
    if any(s and s > 0 for s in interface_sigma_nm):
        mo = [j for j, lyr in enumerate(layers)
              if lyr.rho is not None or lyr.rhop is not None or lyr.mu is not None]
        if mo:
            warnings.warn(
                f"graded_stack: transition sublayers are non-magnetic; MO tensors "
                f"on layers {mo} are dropped inside the graded region", UserWarning)
```

  (Full MO-aware grading — erf-interpolating rho/rhop/mu alongside eps — is a
  documented follow-up, not this phase; the warning is the deliverable.)
- **Dispersive + roughness:** `_as_tensor` rejects `(n_wl,3,3)` with a generic
  message. Wrap in `grade_interface`:

```python
    try:
        ea, eb = _as_tensor(eps_a), _as_tensor(eps_b)
    except ValueError:
        raise ValueError("grade_interface needs scalar/(3,)/(3,3) eps; dispersive "
                         "(n_wl,3,3) + roughness is unsupported — build sublayers "
                         "per wavelength instead") from None
```

### 8.11 Docs (README, same PR)

1. Code-5 table row → the corrected two-line form from §8.3 (transmission
   uses `ga`; reflection uses the cross factor).
2. Ansatz paragraph from §8.4, verbatim, in the Route-1 section.
3. Second "intentional deviation" entry (after the det/division note):
   type-5 transmission `ga` vs upstream `f`, with the 7.5% figure and the
   σ→0 re-convergence statement.
4. Route-2 limitations list: non-magnetic transitions (warns), no dispersive
   eps (errors with workaround), thin-layer clamp (warns) — each naming its
   warning/error so users can `pytest.warns` / grep logs.

### 8.12 Verified non-issues (recorded so nobody "fixes" them)

- `w_function` type-1 gate `val.abs() < 1e-9` ≡ upstream `val.norm() < 1e-9`:
  both are `|z|` (num-complex 0.4.6 `norm()` = `hypot`, `abs()` = `ComplexFloat`
  modulus) — verified in source, not assumed.
- Dressing-then-Q-phase order in `s_to_next`: equivalent to dressing the final
  S (both element-wise diagonal scalings) — comment already correct.
- Interface topology `entry|l0 ↔ smatrix[1]` (+1 offset, `[0]` ambient unused)
  matches upstream `rv_slice[i_next]` convention — pinned by G5.
- σ units nm on both sides (`kz [nm⁻¹] × σ [nm]` dimensionless) — no conversion
  missing.
- Upstream `SQRT3 = 1.73205080757` (11 digits, `optics_core.rs:18`): code-1 W
  carries a ~1e-13 constant-truncation floor vs full-precision evaluation
  (ours is more precise; proven in `tests/navette_parity.rs` with a scaled
  tolerance). Dominates the G5 2.2e-14 worst row. Do NOT "fix" our literal
  to match — the floor is theirs.
- Upstream 0.7.0 `ga` sign (growing transmission factor) is a documented
  model choice in their docstring (perturbative R+T=1 picture), not a typo
  to mirror — see the §8.6 rewrite. Our `ga_up · ga_theory == 1` pin encodes
  the exact relation.

### 8.13 Upstream-only capabilities (out of scope — listed, not planned)

- Roughness × incoherent/thick-layer paths (upstream Modes A/B/C, `tau`;
  berreman has no incoherence at all — revisit if a decoherence phase lands).
- Roughness inside needle/optimizer synthesis loops (no berreman counterpart).
- Looyenga/Bruggeman mixing for graded transitions (upstream has a 50:50
  Looyenga roughness-interface EMA; ours is linear-only — candidate
  `mixing=` extension for `grade_interface` as a later micro-phase).

---

## Phase 9 — Materials: dispersion models in Rust, tensors out

### 9.0 Decision: depend, don't port (supersedes the port table below)

During implementation the user directed use of `navette` from crates.io as
a real dependency — and audit showed the port plan was obsolete: every
kernel (including the KK trio slated for 9b) is `pub` and callable, with no
solver coupling. So `src/materials.rs` is a thin slice-friendly adapter
over `navette::materials::*` (same op order → parity by construction),
`navette = "0.7"` moved from dev- to `[dependencies]` (plus
`ndarray = "0.15"` to build the views), and micro-phase 9b is DISSOLVED
(Cody/Tauc/UBF work with zero extra effort — G14a pins all three at 0.0).
The original file-by-file port verdicts are kept below for the record but
marked superseded; §9.2's quirk catalog still documents what parity
depends on (now upstream's to keep, ours to pin).

Original port table (SUPERSEDED — kept for audit trail; `dev @ 4b59cd7`):

| upstream file | old verdict | what changed |
|---|---|---|
| `units.rs`, `common.rs`, `cauchy.rs`, `sellmeier.rs` | ✅ port Tier 1 | now called directly, no copy |
| `lorentz.rs`, `drude.rs` | ✅ port Tier 1 | now called directly |
| `forouhi_bloomer.rs` | ✅ port Tier 1 | now called directly |
| `ema.rs` analytic arms + Bruggeman | ✅ port Tier 1 | now called directly |
| `table.rs` + `kk::interp` | ✅ port Tier 1 | called directly, PLUS a validation delta (see below) |
| `grid.rs` | ✅ optional | ❌ not wrapped (no caller; segment-grid builder stays upstream-only) |
| `kk.rs`, `cody_lorentz.rs`, `tauc_lorentz.rs`, `ubf.rs` | ⏳ 9b (`realfft`) | ✅ included free via the dependency — 9b dissolved, no `realfft` line in our manifest |
| `mod.rs` `Model`/`Dispersion` | 🔀 adapt | ❌ not mirrored — dispatch lives Python-side (§9.5); no Rust enum |
| `materials/__init__.py` | ✅ port vocab | ✅ same (`MODELS` + keys + defaults verbatim) |
| binding-crate layout, fitting-loop seam | ❌ not ported | still not ported; our seam is `pybind.rs` (§9.6) |

**Validation delta (ours, filed as F4):** upstream `table_nk` uses
`assert!` on grid length/mismatch (panics across PyO3 as `PanicException`,
measured); our adapter validates first and returns `Err → ValueError`
(`src/materials.rs::table_validates_instead_of_panicking`). Only intentional
behavioral delta in the module.

**Deliberately new (upstream is scalar-only):** the tensor assembly layer —
every upstream kernel returns isotropic n̂; birefringent eps tensors (per-axis
models → `diag(n̂²)` → orientation) have no upstream counterpart and live
Python-side in `evaluate_tensor` (§9.5). Upstream also has no MO material
models — rho/rhop/mu stay user-supplied, out of scope.

### 9.1 Two decisions to ratify before coding

**D9.1 License.** DONE (implementation): `license = "LGPL-3.0-or-later"`
added to both `Cargo.toml` and `pyproject.toml`; SPDX headers on
`src/materials.rs` and `navette/berreman_materials.py`. Direct dependency
makes the derivative status unambiguous — no clean-room argument available
or wanted.

**D9.2 Dependencies.** SUPERSEDED: the depend decision pulls the upstream
tree (`ndarray` 0.15/0.17, `realfft`, `rayon`, `serde`, `rand`, …) into the
build — accepted deliberately (kernels + KK trio + future smatrix reuse
outweigh the weight; `cargo test --release --lib` builds in ~58 s).
Our manifest adds exactly `navette = "0.7"` + `ndarray = "0.15"` (view
construction); `realfft` arrives transitively, no direct line. MSRV note:
upstream requires rust 1.88 (edition 2024); ours builds with 1.98.1 —
depending across editions is fine, but a future upstream MSRV bump past our
toolchain fails the build loudly (acceptable, pinned by `Cargo.lock`-less
`"0.7"` req — consider exact pin if reproducibility bites).

### 9.2 Quirk-preservation list (parity-critical — review against these, not vibes)

Every item below is a measured upstream behavior that a naive reimplementation
would get wrong. The port must reproduce each exactly (all verified in source
this phase; paths are `rust/navette/src/materials/`):

- **Units:** nm in everywhere; Cauchy/Sellmeier work in µm (`λ×1e-3`, squared
  once); energies in eV via the exact `HC_EV_NM` above; Urbach uses λ in
  **metres** (`×1e-9`) with α₀ in **1/cm**.
- **Sellmeier:** `n² = 1 + Σ Bᵢλ²/(λ²−Cᵢ)`, third term **only if B3 ≠ 0**
  (branch, not tolerance); then `n = √n²`.
- **Urbach tail:** `k = α₀·exp((E−Eg)/Eu)·λ_m/4π` for **strict** `E < Eg`, else
  exactly 0 (`common.rs:urbach_k`).
- **Drude guards:** `E_eff = E + 1e-12` in the Drude denominator **and** in the
  Drude–Lorentz oscillator damping term (`-e*γ_l` uses `E_eff`, `drude.rs` —
  faithful quirk, do not "fix").
- **Lorentz:** `ε = ε∞ + Σ f·E0²/((E0²−E²) − i·E·Γ)`, `n̂ = √ε` (principal sqrt —
  same `num_complex::sqrt` both sides, so bit-parity holds).
- **Forouhi–Bloomer 2019:** `disc = 4C−B²`, `Q = √disc/2` else `1e-6` when
  `disc ≤ 1e-12`; denominator clamped **up to +1e-15** (`if |D|<1e-15 {D=1e-15}`
  — sign-destroying clamp, keep it); `k = 0` for `E < Eg`; metal prepends the
  free-electron row `(Eg=0)` **only if `A_fe > 0`**; term layout `(N,4) =
  (Eg,A,B,C)`; everything sums on top of `n_inf` (index, not permittivity).
- **Tauc–Lorentz:** ε₂ is the analytic Jellison–Modine form, but ε₁ comes from
  the **FFT-KK grid, not the closed-form atan/ln ε₁** (`tauc_lorentz.rs` docs)
  — hence Tier 2 even though ε₂ is elementary. Osc layout `(N,3) = (A,E0,C)`.
- **UBF:** kernel takes **β = 1/Eu** (converted Python-side, `_ubf_array`);
  log-term guards `x > 50 → x`, `x < −50 → 0`; γ fast paths `2 → base²`,
  `0.5 → √base`, `1 → base` (else `powf` — parity needs the branches);
  osc layout `(N,6) = (Eg,Ec,β,A,Γ,γ)`; Eu ≤ 0 rejected.
- **EMA:** mixers take **n̂ and square internally**, return **ε** (caller applies
  `eps_to_nk` — matches `_native.eps_to_nk(ema_*(…))` call order in
  `evaluate`); PowerLaw falls back to Lichtenecker for `|α| < 1e-6`;
  Bruggeman inits at the arithmetic mean, `tiny = 1e-15` complex guard,
stops on `|Δ|² < tol²`, serial op order preserved for bit-parity.
- **Table:** `np.interp` clamped semantics (out-of-range → edge values, **not**
  NaN/error); `k_vals=None → k=0`; `n_factor/k_factor` post-scale;
  non-`"linear"` interpolation types **refuse with an explicit error**
  ("resample the table offline") — port the refusal, it is API.
- **Osc validation:** `(N,width)` 2-D, N ≥ 1, exact widths (Lorentz/Drude/Tauc
  3, Cody/FB 4, UBF 6) — port `_as_osc_array` checks with the same messages.
- **Defaults:** `epsilon_inf/n_inf` default 1.0, MoriTanaka `L` default 1/3,
  PowerLaw `alpha` default 0.5, Konstant `k` default 0.0, DrudeLorentz accepts
  `gamma_drude` **or** `gamma`.

### 9.3 New: tensor assembly (no upstream counterpart) — Python-side

CHANGE from the drafted Rust `TensorSpec` enum (kept below for the record):
with dispatch Python-side, a Rust enum would duplicate the dispatcher for no
parity gain — the assembly is exact numpy arithmetic either way. So
`evaluate_tensor` lives in `navette/berreman_materials.py` (per-axis
`evaluate` → `diag(n̂²)` → optional `R eps Rᵀ`), and Rust provides only the
`eps_of_n` squaring helper to keep the op identical on both sides of the
binding. G14b pins the identities to 0.0.

```python
# navette/berreman_materials.py::evaluate_tensor (as built)
# spec: MaterialSpec | {"uniaxial": {"ordinary":.., "extraordinary":..}}
#       | {"biaxial": {"x":.., "y":.., "z":..}}  -> (n_wl,3,3) diag eps
```

<!-- Drafted Rust enum (NOT built — record only):
#[derive(Clone, Debug)]
pub enum AxisModel { /* resolved variants per MODEL */ }
#[derive(Clone, Debug)]
pub enum TensorSpec {
    Isotropic(AxisModel),
    Uniaxial { ordinary: AxisModel, extraordinary: AxisModel },
    Biaxial { x: AxisModel, y: AxisModel, z: AxisModel },
}
-->

Rules (as built): `Uniaxial{o == e}` ≡ `Isotropic` exactly (same kernel calls
— asserted 0.0 in G14b); orientation via `rotate=` (`R eps Rᵀ`, same
convention as `rot_z`) or downstream `rot_z` / Phase-6 `rotations.rs` —
materials never rotate internally; EMA composites stay scalar per axis
(per-axis mixing is the defined semantic); `Roughness(bottom, top)` maps to
`roughness_interface` per axis (and is recorded as the future `mixing=` option
for `grade_interface`, cf. §8.13).

### 9.4 Rust surface (`src/materials.rs`, as built — adapters, not kernels)

```rust
// Slice-friendly adapters; infallible kernels return Vec<C>, the three KK
// drivers + shape validations return Result<Vec<C>, String> (→ ValueError).
// Oscillator matrices ride flat row-major (&[f64] + width) — no ndarray at
// the boundary; views are built inside. EMA takes n̂ slices, returns ε
// (caller applies eps_to_nk — upstream order). tensor assembly is NOT here
// (§9.3); only the eps_of_n squaring helper is.
pub fn konstant_nk(wl: &[f64], n: f64, k: f64) -> Vec<C>;
pub fn table_nk(wl: &[f64], grid: &[f64], n: &[f64], k: Option<&[f64]>, nf: f64, kf: f64) -> Result<Vec<C>, String>;
pub fn cauchy_nk / cauchy_urbach_nk / sellmeier_nk / sellmeier_urbach_nk(..) -> Vec<C>;
pub fn lorentz_nk(wl: &[f64], osc_flat: &[f64], eps_inf: f64) -> Result<Vec<C>, String>; // width 3
pub fn drude_nk(..) -> Vec<C>;  // + drude_lorentz_nk (width 3)
pub fn cody_lorentz_nk(.., osc_flat width 4, ..) -> Result<Vec<C>, String>; // KK, upstream Result passthrough
pub fn fb_interband_nk / fb_metal_nk(.., ib_flat width 4, ..) -> Result<Vec<C>, String>;
pub fn tauc_lorentz_nk(.., width 3, ..) / ubf_nk(.., width 6, ..) -> Result<Vec<C>, String>;
pub fn ema_lichtenecker / looyenga / power_law / maxwell_garnett / mori_tanaka -> Vec<C>;
pub fn ema_bruggeman(n_i: &[C], n_h: &[C], f: f64, max_iter: usize, tol: f64) -> Vec<C>;
pub fn ema_roughness(n_bottom: &[C], n_top: &[C]) -> Vec<C>; // 50:50 Looyenga
pub fn eps_to_nk(eps: &[C]) -> Vec<C>;
pub fn eps_of_n(nk: &[C]) -> Vec<C>; // element-wise n̂² (tensor-diag op)
```

`#[cfg(test)]` (3 tests, green): table validates instead of panicking;
osc shapes rejected; Bruggeman f=0/1 endpoints bit-exact. Parallelism is
upstream's (`map_nk`/rayon inside the kernels) — no split logic of our own.
NOT wrapped (deliberate): `wiener_bounds` (tuple, no Python caller),
`grid.rs` builder, `Model`/`Dispersion`/`MixRule` (dispatch lives in §9.5).

### 9.5 Python API (`navette/berreman_materials.py` — same vocab as upstream)

MODULE NAME (implementation finding): `navette/materials.py` is
unreachable — the wheel's `navette/materials/` package shadows it in the
merged namespace (site-packages portion sorts first; verified live). Ours
is therefore `navette.berreman_materials` (same `MODELS` vocab + params,
plus the new tensor layer). Revisit only if the wheel is ever absent.

```python
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from navette import _berreman as _b

MODELS = ("Konstant", "Table", "Cauchy", "CauchyUrbach", "Sellmeier",
          "SellmeierUrbach", "Lorentz", "Drude", "DrudeLorentz",
          "CodyLorentz",                       # ← in scope from day one (9b dissolved)
          "ForouhiBloomerSingle", "ForouhiBloomerMulti",
          "ForouhiBloomerMetal", "ForouhiBloomerMetal2021",
          "TaucLorentz", "UBF",                # ← likewise
          "Bruggeman", "MaxwellGarnett", "Looyenga", "Lichtenecker",
          "MoriTanaka", "PowerLaw", "Roughness")

@dataclass
class MaterialSpec:
    model: str
    params: Dict[str, Any]          # SAME keys as upstream evaluate() (§9.2 defaults)

def evaluate(spec, wavelength_nm) -> np.ndarray:
    """Scalar n̂(wl), complex128 — parameter-for-parameter compatible with
    navette.materials.evaluate for the Tier-1 MODELS (G14a parity)."""
    ...   # req()/dispatch mirror of upstream __init__.py:136-257 + _eval_nested

def evaluate_tensor(spec, wavelength_nm) -> np.ndarray:
    """Birefringent constructor. spec: MaterialSpec (isotropic) or
    {"uniaxial": {"ordinary": spec, "extraordinary": spec}} /
    {"biaxial": {"x": spec, "y": spec, "z": spec}} (nested dicts or
    MaterialSpec). Returns (n_wl,3,3) complex128 diagonal eps = n̂², ready for
    bl.Layer / graded_stack. Optional rotate=(phi,theta,psi) applied via
    existing rot helpers (Phase 6 later)."""
```

Split of labor (matches upstream native/thin-Python shape): kernels in Rust
(§9.6 bindings, upstream delegates); dispatch/defaults/validation/nesting +
tensor assembly in Python. No `NotImplementedError` arms — all 23 MODELS
work (9b dissolved). Bruggeman exposes `max_iter` (default 100) / `tol`
(default 1e-9) beyond upstream's fixed call (superset, defaults match
`MixRule::from_name`).

### 9.6 Bindings (`pybind.rs`, 22 entries, as built)

One thin wrapper per kernel: `materials_konstant/table/cauchy(+_urbach)/
sellmeier(+_urbach)/lorentz/drude/drude_lorentz/cody_lorentz/fb_interband/
fb_metal/tauc_lorentz/ubf` (numpy in → complex128 out, `Err→ValueError`) +
7 EMA (`lichtenecker/looyenga/power_law/maxwell_garnett/mori_tanaka/
bruggeman/roughness` — return ε, Python applies `materials_eps_to_nk`,
upstream order) + `materials_eps_to_nk`. Osc/ib ride as 2-D float arrays
(flat + width inside); `k_vals`/`fe` validated (`fe` len 3). No GIL-release
( point-wise kernels already rayon-parallel upstream; wrapper overhead
negligible). No `materials_tensor_eps` — tensor assembly is Python-side
(§9.3 change).

### 9.7 Gates G14a–c (`tests/test_materials.py`; G14a needs the PyPI wheel)

OBSERVED (implementation, all green):

- **G14a live parity — ALL 23 MODELS × 200-pt grid (300–1200 nm): worst
  0.0 exactly** (same code path both sides — parity by construction, not
  by port fidelity). Bound stays `≤ 1e-15` (fails loudly on any upstream
  drift/upgrade). Upstream import is `navette.materials` (wheel); ours is
  `navette.berreman_materials` (§9.5 name finding).
- **G14b tensor identities (no upstream needed):** all pass — uniaxial o≡e
  ≡ isotropic `0.0`; Bruggeman `f=0/1` endpoints `0.0`; `PowerLaw(α=1e-9)`
  ≡ Lichtenecker `0.0`; Table clamped outside (edge values); Urbach `k=0`
  above gap `0.0`. (MG dilute-limit print dropped as built — covered by
  live MG parity instead.)
- **G14c end-to-end:** BK7-Sellmeier slab via `evaluate_tensor` → `Layer`
  → `BerremanStack` ≡ direct-eps stack `0.0`, `n_failed == 0`, energy sane;
  uniaxial e2e OK.
- Oracle style follows the repo pattern (`worst < tol` + printed worst).

### 9.8 Micro-phase 9b — DISSOLVED (KK trio ships in Phase 9)

No port, no `realfft` line, no `tests/test_materials_kk.py`: the dependency
gives `cody_lorentz_nk` / `tauc_lorentz_nk` / `ubf_nk` (upstream `Result`
passthrough → `ValueError` on grid-range breach) with G14a parity 0.0 on
all three. The `kk.rs` internals note (8192-pt grid, odd extension, 1/M
scaling, DC/Nyquist zeroing) is upstream's to maintain. This section kept
as a tombstone so nobody re-plans the port.

---

## Phase 10 — Chiral media (Pasteur constitutive + cholesteric Bragg)

### 10.0 Status: solver-ready, everything else missing

| piece | status |
|---|---|
| `berreman_full(eps, rho, rhop, mu)` arbitrary 3×3 bianisotropic | ✅ exists — a Pasteur medium *solves* today if hand-fed tensors |
| circular I/O basis F/B (frozen §0.3) | ✅ exists — the right observables for rotation/Bragg/CD |
| `twisted_stack` staircase helper (Phase 7, `grid="midpoint"`) | ⚠️ PLAN BUG — never built (Phase 7 shipped aniso-exit, no twist helper); BUILT in P10 (§10.5) |
| κ → (rho, rhop) mapping | ✅ pinned — RECORDED assignment `rho = −iκI, rhop = +iκI` (sign-flipped vs the e^{−iωt} derivation, §10.2) |
| constructors, dispersive κ | ✅ `chiral_layer`, `kappa_table`, `twisted_stack`, `cholesteric_stack` (Condon + β↔κ deferred, bar stated) |
| any chiral validation | ✅ analytic G15a–c + oblique + G15e, live-Bragg G15d |

References: B44 has **no** constitutive chirality (grep empty) but owns
structural twist (`TwistedMaterial`, B44.py:396); pyllama has no named chiral
model but its `full_berreman.calc_berreman_matrix` is the mapping source;
upstream navette has neither (isotropic engine). Primary parameterization:
**Pasteur–Tellegen** (maps directly onto (rho, rhop), no spatial dispersion).

### 10.1 Box: chiral validation is analytic-only (the det typo forbids oracles)

With `rho22·rhop22 = κ² ≠ 0`, our corrected `÷det` and live pyllama's
literal `×det` typo (deviation #1, §0.5) differ by **det² in every a_i** — live
`full_berreman` can *never* oracle a chiral stack (it cannot oracle achiral
full-matrix stacks either). The implementer must NOT "validate" against it:
any comparison will differ and that difference is expected (shape: a_i ratio =
det² = (εμ−κ²)², assertable as a cross-check in G15a, not a failure). Gates
below are analytic (10a) + independent-solver (10b, B44 has no typo exposure:
structural twist uses achiral slices through B44's own Δ path, already
triangle-validated).

### 10.2 Task 0: Pasteur–Tellegen → (rho, rhop) mapping (derive, then pin)

Pasteur–Tellegen, SI, e^{−iωt}: `D = εE + iξH`, `B = −iξE + μH`, `ξ = κ/c₀`
(κ dimensionless). Code units are impedance-normalized (`h = Z₀H` is the
code's H; `D̃ = D/ε₀`, `B̃ = c₀B` carry E-field dimensions):

```text
D̃ = ε_r·E + i·(ξ/ε₀Z₀)·h = ε_r·E + iκ·h      (c₀ε₀Z₀ = 1)
B̃ = μ_r·h − i·(c₀ξ)·E       = μ_r·h − iκ·E      (c₀μ₀ = Z₀)
```

The implemented Mazur/Azzam-Bashara form reads `D̃ = eps·E + rho·h`,
`B̃ = rhop·E + mu·h` (tensor-level, unaffected by the row sign flips already
in the matrix). **Ansatz: `rho = +iκ·I₃`, `rhop = −iκ·I₃`** (reciprocal
Pasteur pair, `rho = −rhopᵀ`). The ansatz assumes e^{−iωt}; the time
convention is currently unpinned by any live oracle (achiral R/T are invariant
under simultaneous conjugation), so pin it numerically — robust to the
derivation being conjugated:

1. **Spectrum:** `eig(berreman_full(n²I, +iκI, −iκI, I, Kx=0))` must equal
   `{±(n+κ), ±(n−κ)}` ≤ 1e-15. Any `{±(n±iκ)}` → stop, ansatz structurally
   wrong (would mean the code's rho slot carries a different normalization).
2. **Handedness (operational R/L definition):** the forward (`Im q ≥ 0`)
eigenvector with `q = n+κ` must satisfy `Ey/Ex = −i`, i.e. be proportional to
   frozen F-col0 `[1,−i]`. If it pairs with F-col1 instead → flip both signs
   (`rho = −iκI`, `rhop = +iκI`), record `TIME_CONVENTION = e^{+iωt}` in the
   module docstring, and proceed — every downstream formula is written in κ
   plus the recorded assignment, so nothing else moves.
3. **Consistency:** the recorded convention must match the MO-tensor
   convention already validated in `test_full.py` (same Δ, same slots —
   cross-check, flagged for reviewer sign-off, not an oracle).

Regime guard: `det = n²−κ² ≠ 0` required (singular → existing `None→NaN`
`n_failed` path, asserted in G15a); tests use κ ∈ {0.001, 0.01, 0.05, 0.1} at
n ≈ 1.5. Constructor warns for `|κ| ≥ 0.2·n` (natural-media regime; chiral
nihility κ→n is out of scope).

TASK-0 OUTCOME (as run — the ansatz sign did NOT survive contact):
`rho = +iκI` pairs q = n+κ with F-col1 (Ey/Ex = +i, LCP channel), so per the
step-2 protocol BOTH signs flipped: RECORDED mapping `rho = −iκI`,
`rhop = +iκI`, TIME_CONVENTION = e^{+iωt} (code slots couple opposite to the
e^{−iωt} SI derivation; achiral observables can't see this, chiral ones do).
This restores the standard n_R = n+κ on the frozen channel-0 = RCP basis
(pyllama `fresnel_to_fresnel_circ` docstring: J_circ[0,0] = RCP→RCP).
Pinned by `berreman::tests::pasteur_task0_normal_spectrum_and_handedness`
(spectrum {±(n±κ)} real — gate 1e-14 not 1e-15: eig4 absolute accuracy scales
with ‖Δ‖ ~ n², observed 5.6e-15 at n = 1.5; handedness ratios ≤ 1e-12) +
`pasteur_task0_oblique_structure` (Kx = 0.3: ±q pairing ≤ 1e-12, exact
spherical dispersion Kx²+q² = (n±κ)², transverse circularity E_v/E_u = ∓i in
(u = ŷ, v = k̂×u) ≤ 1e-9). MO-consistency (step 3): same Δ slots as the live-
validated MO path (test_full.py) — reviewer sign-off recorded here.

### 10.3 Constructors (Python-side numpy — no hot loop; Rust untouched)

```python
def chiral_layer(n, kappa, thickness_nm, mu=1.0):
    """Pasteur–Tellegen slab: eps=n²I, rho=+iκI, rhop=−iκI, mu=μI
    (signs per recorded TIME_CONVENTION, §10.2). Validates finite n>0,
    warns |κ|≥0.2n. Full docstring: SI definition, normalization chain,
    eigen-indices n±κ with R/L assignment."""
    ...  # rho/rhop built with 1j*kappa * np.eye(3); returns bl.Layer(...)

def kappa_table(wavelengths_nm, kappas):
    """Dispersive κ by reusing the Phase-9 Table model (no new formula).
    A named single-oscillator Condon ORD model is a STRETCH GOAL with a
    literature checkpoint — do NOT hardcode a Condon form without a cited
    source and reviewer sign-off (competing sign/factor conventions)."""
```

### 10.4 Gates G15a–c (`tests/test_chiral.py`, no external reference)

AS-BUILT CORRECTIONS (three draft premises fell — recorded, not hidden):

1. **G15b reference replaced.** The draft's "each helicity channel carries
   an ordinary single-slab Airy with its own index n±" is FALSE — disproven
   at 0.4 (co-channel vs Airy(n±) differs O(1)). The TRUE relations, exact to
   ~1e-14 across 4 thicknesses (3 d × 2 κ × 2 n × 2 io fixtures):
   `t_R = t0·e^{+iD}`, `t_L = t0·e^{−iD}` with `D = k₀κd` (transmission keeps
   the one-way chiral phase); `Jr[1,0] = Jr[0,1] = r0` (reflection flips
   helicity by J_z conservation and the forward/backward chiral phases cancel
   round-trip); all other circ entries ~0 (no helicity mixing in T, none
   preserved in R). `(r0, t0)` = the achiral slab (same geometry, simple
   path — itself live-validated), so all chirality sits in the analytic
   phase. Corroboration: (i) INDEPENDENT numpy/scipy-expm replication of the
   full-Δ chain reproduces the solver exactly (zero shared code); (ii)
   symmetry structure (diagonal-T, antidiagonal-R) group-forced, observed;
   (iii) per-channel energy; (iv) G15c rotation. Literature tertiary citation
   stays a review checkpoint (same bar as Condon). Observed: trans-phase
   3.2e-14, refl 9.6e-15 (≤ 1e-12); forbidden 3.2e-14 (draft said 1e-15 —
   fp-cancellation floor of the F/B sandwich, ≤ 1e-12).
2. **G15c bare-phase promoted to exact gate.** Under the true relations the
   Fresnel part (t0) is COMMON and cancels in t_R/t_L, so rotation = k₀κd
   exactly (printed 3.000000°/15.000000°/6.000000°/30.000000°); the draft's
   "do NOT gate against bare-propagation" assumed per-channel Fresnel
   phases (textbook-Airy thinking). Ratio form `ρ = t_R/t_L` vs `e^{2iD}`
   (no arg branch cuts): 4.1e-14 (≤ 1e-12). Antisymmetry |ρ(κ)·ρ(−κ)−1| =
   1.6e-16 (draft said exactly 0.0 — fp ratio noise; ≤ 1e-12). Energy
   |t±|²+|r|²−1 = 1.7e-15 (per-helicity pairing, not co-diagonal).
3. **G15a κ=0: degenerate floor, not 0.0.** Isotropic-full spectrum is
   exactly degenerate → cluster-arm repair leaves q error ~3e-11 scaling
   LINEARLY in k₀d (measured 3.3e-11/1.6e-10/3.7e-10 at d = 100/500/1000;
   pre-fix: 0.78 silent garbage). Gate ≤ 1e-9 documents the honest floor.
   Anisotropic κ=0: non-degenerate singleton path → exactly 0.0 ✓.
   Singularity κ²=n² → n_failed = 1 with NaN ✓ (needed the honest-NaN
   backstops of §10.8: previously NaN values with n_failed = 0).

- **Oblique (no closed form):** cross-pol κ-odd 0.0, co-pol κ-even 0.0
  (symmetric slab, entry = exit); κ = 1e-9 → achiral 3.3e-10 (floor);
  Lorentz reciprocity `J_trans(mirror) == J_trans(fwd).T` = 1.2e-15 where
  mirror = reversal + κ negation (mirrored helix = enantiomer) — the draft's
  "R,T equal on reversal" was wrong physics TWICE (enantiomer-flip AND
  powers transpose; reflection has no amplitude relation). Naive-reversal
  power difference 3.9e-4 pins non-vacuity. Energy per column ≤ 3e-15.
- **G15e (new — pins the §10.8 from_psi_full fix end-to-end):** lossy chiral
  slab (complex n + real κ, oblique so Ez ≠ 0) absorption balance
  (A_int+F_out)/F_in − 1 = 7.3e-9 (≤ 1e-6; 2002-pt trapezoid with the
  endpoint discipline below). Also caught a REAL physics subtlety, not a
  bug: z == total returns EXIT-side values and Ez jumps by eps_ratio
  (normal-D continuity — observed |Ez|² ratio 5.06 == |n_c²|²); absorption
  integrals must use interior-side limits (documented in `fields()`).

### 10.5 Track 10b: cholesteric circular Bragg vs live B44 (G15d)

- **Fixture:** uniaxial `no=1.5, ne=1.7` (B44 `UniaxialMaterial(1.5,1.7)`),
  pitch `p=350 nm`, `N=5` pitches, `d=N·p`, `angle=±2πN`, 48 slices/pitch
  with a 24/48/96 convergence triple (monotone, 48-vs-96 printed).
- **Sampling match (DRAFT INVERTED — corrected):** B44 `getSlices` returns
  endpoint edges `linspace(0,d,div+1)`, but `InhomogeneousLayer`'s default
  midpoint evaluation samples the tensor at slice MIDPOINTS `(k+0.5)·d/div`
  (`getSlicePropagator_mid`). Our `grid="midpoint"` (default) reproduces
  B44's sampling EXACTLY — verified 5.6e-17 on tensor samples (rotation sense
  + base tensor + sampling in one number). `cholesteric_stack` defaults to
  midpoint; endpoint kept for explicit edge-matched studies only.
- **Live side:** `TwistedMaterial(material, d_meters, angle, div)` — note
  **meters** (`d=1.75e-6`) — in a B44 `Structure` per Appendix B, values via
  the `gen_live_ref.py` pattern committed to `ref/ref_out_cholesteric.json`.
  (Correction recorded here: plan §§4/7 cite a `CholestericModel` — B44's
  class is `TwistedMaterial`, B44.py:396. Fix those two references in this PR.)
  B44 API notes (measured, not guessed): base material must carry the
  extraordinary axis IN-PLANE (`NonDispersiveMaterial(diag(ne²,no²,no²))` —
  `UniaxialNonDispersiveMaterial` puts it along z); no `circularJones`
  method exists (own F/B sandwich; layouts match per G7b); slice propagator
  midpoint + Padé-2 (default) suffices at 48/pitch; scan window widened to
  480–640 nm (the ~95 nm-wide stopband overflows 520–600).
- **Observable:** circular Jones `R_RR` across the stopband
  `λ ∈ [n_o·p, n_e·p]` ± margin: Bragg peak **position ≤ 0.5%**, **height
  ≤ 3%**, **FWHM ≤ 5%** (position is geometric → tight; height/width feel
  boundary discretization → looser). OBSERVED at matched discretization:
  dpos = 0.0, dhei = 1.7e-15, dfwhm = 4.8e-15 — the engines are transfer-
  matrix-identical (same math, equivalent Δ); our triple 24/48/96 monotone
  (0.870296 → 0.871817 → 0.872188, 48-vs-96 4.3e-4). **Handedness check:**
  `angle → −angle` swaps the peak R_RR → R_LL at 558.0 nm both sides,
  suppressed channel 0.0053 (< 5% of 0.8718) — asserts the swap, not one sign.
  JSON integrity spot-check (5 live λ vs committed): 1.7e-15.
- `pytest.importorskip("Berreman4x4")` pattern for the live arm (no-network
  CI runs the analytic G15a–c always, G15d against the committed JSON).

### 10.6 Bindings + docs (same PR)

No new Rust (the solver slot exists). Python: `chiral_layer`, `kappa_table`
in `berreman.py` beside `Layer` (+ re-export). README gains: SI
Pasteur–Tellegen definition, the normalization chain + recorded time
convention, the mapping box, κ-regime note, and the analytic-only validation
statement with the §10.1 reason. Docstrings carry the derivation references
(Lindell–Sihvola constitutive form; Mazur/Azzam-Bashara matrix source already
cited in `berreman.rs`).

AS BUILT: substantial new Rust after all (findings below — §10.8). No
`__init__.py` re-export exists (namespace package merged with the upstream
wheel) — module-level `bl.chiral_layer` etc. suffice; "re-export" N/A.
`kappa_table` is a validation wrapper returning `(wl, kappas)` for
Phase-9 Table interpolation by the caller. Condon ORD model + β↔κ helper:
DEFERRED with the bar stated (no citable mapping source at hand; the plan's
own "do NOT hardcode" rule forbids guessing) — open reviewer checkpoints,
not silent drops.

### 10.7 Why not Drude–Born–Fedorov (study note — justifies the §10.0 choice)

DBF (time-harmonic, e^{−iωt}): `D = ε(E + β∇×E)`, `B = μ(H + β∇×H)` —
chirality through the **curl**, one length-dimensioned parameter β. Pasteur–
Tellegen: `D = εE + iξH`, `B = −iξE + μH` — **local algebraic** coupling,
`ξ = κ/c₀`. Terminology (per recent literature): "chiral" = imaginary
(reciprocal) coupling, "Tellegen" = real-χ (nonreciprocal, axion-linked)
coupling; our (rho, rhop) slots can express both, plus Faraday MO which is a
third, distinct nonreciprocal form — keep the three labels separate in docs.

- **Equivalence scope:** interconvertible by field/parameter redefinition for
  single-frequency fields (Lindell–Sihvola transformation theory; Lakhtakia
  *Beltrami Fields in Chiral Media*). Off-resonance the choice is
  representation, not physics — nothing is lost by standardizing on κ.
- **Disqualifier 1 — time-domain ill-posedness:** Freymond & Picard,
  arXiv:1204.5350 ("The Elusive Drude-Born-Fedorov Model"): DBF is settled
  time-harmonically but "in the physically relevant time-dependent case the
  record is much less convincing" (solution theory needs extrapolation
  spaces). Pasteur–Tellegen evolves cleanly with standard energy expressions.
- **Disqualifier 2 — wrong resonant dispersion:** Cho, arXiv:1501.01078:
  DBF misses the linear k=0 crossing in the resonant chiral left-handed
  region vs first-principles macroscopic constitutives — "only ...
  phenomenology in off-resonant region".
- **Disqualifier 3 — solver fit:** curl coupling means higher spatial
  derivatives at interfaces (delicate BCs) and no direct 4×4/transfer-matrix
  form; Pasteur drops into (rho, rhop) with `n_{R,L} = n±κ` immediate.
- **DBF's remaining merit (acknowledge, don't implement):** β is the natural
  first-order spatial-dispersion term — what helix-composite homogenization
  yields — with a century of analytic results. Relevant only for time-domain
  work or β-from-geometry fitting; neither applies here.
- **Actionable residue:** metamaterial papers report **β**, so the docs must
  ship a weak-chirality β↔κ translation helper. Do NOT hardcode a closed
  form from memory — cite the mapping source at implementation time
  (reviewer checkpoint, same bar as the Condon stretch goal in §10.3).

Sources: 1204.5350 · 1501.01078 · 1605.06406 (bi-isotropic surface waves) ·
2406.10277 (Tellegen/axion link). Full study in session notes; this section is
the normative summary.

### 10.8 Implementation findings (three — all caught by P10's own gates)

**F1. `Wave::from_psi` ez/hz wrong for full-tensor layers (pre-existing
since P1).** `from_psi` reconstructs `Ez = −(εzxEx+εzyEy+kx·Hy)/εzz`,
`Hz = kx·Ey` — the SIMPLE constitutive. With (rho, rhop, mu) ≠ (0,0,I) the
longitudinal components need the full 2×2 elimination (Azzam-Bashara
weights, read off `berreman_full`'s b-matrix feedback:
`Ez = a1Ex+a2Ey+a4Hy−a3Hx`, `Hz = a5Ex+a6Ey+a8Hy−a7Hx`). Caught by the
Pasteur oblique-circularity probe (Ev/Eu off by 2.7e-4 — small because κ is
small, but STRUCTURAL). Tangential (Ex,Ey,Hx,Hy) come straight from ψ and
were always exact → Jones, fluxes, G8b continuity, G8e (isotropic) all
unaffected; only Ez/Hz (fields longitudinal output, `absorption_density`'s
|E|² on full-tensor layers, `Wave.sz`) were wrong. Fix: `from_psi_full` +
`sort_partial_waves_full` (sort heuristics see true fields) +
`psi_to_eh_full` (per-sample select from `layer.full` / exit-spec full);
simple path keeps `from_psi` BIT-IDENTICALLY (reduction ≤ 1e-15, not 0.0 —
a6 = kx·e22/e22 rounds 1 ulp — hence opt-in, not replacement). Pinned by
G15e (chiral absorption balance 7.3e-9, oblique so Ez ≠ 0) + the oblique
Task-0 test (which USES `from_psi_full` — the line pins the fix). P2's
"valid for MO too" comment corrected. Full suite green = zero behavior
change on existing paths.

**F2. `eig4` degenerate-cluster silent garbage (pre-existing engine
landmine, P10-blocking).** Clustered roots (width ≤ 1e-6·scale) go through
`null_space_basis` on the cluster MEAN — but quartic-root noise on multiple
roots is ~1e-8, so the mean is off-spectrum and a noise pivot slips past the
1e-9 RREF threshold → vectors with residual 2.2 (not eigenvectors at all).
Measured boundary (pre-fix): iso-full κ = 0 AND κ = 1e-9 AND aniso
δn ≤ 1e-6 (SIMPLE path!) all garbage; iso κ = 1e-6 marginal (8.5e-7). The
simple path's `is_isotropic` shortcut (tol 1e-12) only covers δn < 1e-12 —
weak birefringence 1e-12 < δn < 1e-6 was silent garbage on the SIMPLE path
too. Natural optical activity (κ ~ 1e-8) sits inside the landmine → P10
could not ship without this fix. Repair (cluster arm, mult ≥ 2 ONLY —
singletons run the ORIGINAL code verbatim, so all non-degenerate solves are
bit-identical BY CONSTRUCTION): verify candidates by Rayleigh residual
(root-noise-immune); repair failures with shift-invert from
member-orthogonalized starts (direction preserved inside a true multiple →
orthonormal by construction, no mixing cost); MGS true-clusters only;
SUBSTITUTE the Rayleigh quotient for the quartic root (recovers ~1e-10
Jones floor, linear in k₀d — the honest degenerate accuracy floor, G15a
§10.4); persistent failures ZERO-FILL the column (P singular → `mat4_inv`
→ None — honest, never silent). Boundary now (gated in
`full_degenerate_boundary_probe`): all ≤ 1e-9 (four at 0.0, κ = 1e-6 at
1.3e-11). Newton polish on the char poly was TRIED and REVERTED (p ≈ p' ≈
0 near multiple roots → 0/0-dominated 5e-9 wander); Vieta-exact q for the
double-double case documented as future work (`newton_polish` kept as
`#[allow(dead_code)]` documentation of the negative result).

**F3. Honest-NaN backstops (det-singular + NaN-through-Some).** κ² = n²
(det = 0) produced NaN VALUES with `n_failed = 0` (`from_solve` set ok:true
unconditionally — NaN through Some was never counted). Two one-line-class
fixes, full suite green (no false positives): `mat4_inv` pivot test is now
`!(best >= 1e-300)` (NaN → None; subnormal/zero unchanged) and
`PointOut::from_solve` sets `ok:false` on ANY non-finite Jones/power output
(defense in depth — catches NaN paths that never touch an Option, e.g.
fresnel 0/0 divisions). G15a singularity arm (n_failed = 1 + NaN) needed
both. Note EM is DEGREE-OF-FREEDOM-exempt: `expm(Δ)` needs no eig, so EM
solves degenerate layers correctly where TM/SM honestly return None —
uniform-None was rejected (uniformity over cleverness loses to correctness).

## Appendix A — Upstream reference index (stable pointers)

| Symbol | Location |
|--------|----------|
| `calc_berreman_matrix` (literal, typo) | `BerreMueller/src/berremueller/full_berreman.py` (whole file, 75 lines) |
| `Layer._build_D`, `_sort_p_q`, `_correct_p` | `.../pyllama.py` ll. 366–712 |
| `Structure.build_scattering/transfer/exponential_matrix` | `.../pyllama.py` ll. 1073–1225 |
| `_get_fresnel_SM/TM/EM`, `fresnel_to_fresnel_circ` | `.../pyllama.py` ll. 1455–1578 |
| `get_refl_trans`, `Model.get_refl_trans_coefs` | `.../pyllama.py` ll. 1297–1440, 1932–1987 |
| `get_in_plane_fields`, `propagate_eigenmodes` | `.../pyllama.py` ll. 2039–2125 |
| `StackModel` (+`N_per`), `CholestericModel`, `MixedModel` | `.../pyllama.py` ll. 2125–2515 |
| `mueller_from_jones_matrix` | `.../berreman_mueller.py` ll. 148–182 |
| DBR/cavity apparatus | `.../berreman_mueller.py` ll. 192–660 |
| `create_pauli_stack`, Cloude, polarizance | `.../mueller.py` ll. 159–191, 507–746 |
| `rot_mat`, Euler/quaternion, LD/LB | `.../dielectric_tensor.py`, `.../pyllama.py` ll. 40–69, 3016–3062 |
| `Structure` (aniso back), `getJones`, Padé/Taylor propagators | `Berreman4x4/Berreman4x4.py` ll. 467–562, 1111–1340 |
| `TwistedMaterial`, `RepeatedLayers`, `Evaluation` | `Berreman4x4/Berreman4x4.py` ll. 396–466, 1032–1110, 1334–1340 |

Live validation scripts from the Sep-2025 session (re-runnable):
`validate_live_berremueller.py`, `validate_berreman4x4.py`, `validate_stress.py`
(all drive the *installed* `navette` wheel + live sources).

## Appendix B — Architecture review: Rust-only kernels, thin Python (post-P10)

Mandate: hold the implementation to the module docstring's own claim —
"owns no physics: every number comes from the Rust sweep" — the same contract
upstream `navette` states for `smatrix.py`. Reviewed `navette/berreman.py`
(751 lines), `berreman_materials.py` (320) against `pybind.rs`/core.

**Verdict: four Python-side numeric kernels found and moved to Rust (this
appendix records what moved where and what re-pinned them); everything else
was verified thin.** The upstream-idiomatic line we drew: dispatch, defaults,
validation, warnings, `Layer`-list assembly and result shaping stay in Python
(exactly what upstream's `smatrix.py`/`materials` `__init__.py` do); tensor
*math* does not.

**K1 — Route-2 graded profile (grade_interface) → `roughness::graded_tensors`.**
The Gaussian-CDF volume fraction `f(z) = ½(1+erf(z/σ√2))` and the linear
tensor mixing were Python (`math.erf` + numpy). Moved wholesale; Python keeps
the shape validation and the `sigma ≤ 0 → []` semantic. Needed a
double-precision `erf` in Rust (std has none; new deps forbidden): power
series with a Neumaier-compensated sum for |x| ≤ 2.25 (the naive sum loses
~5e-14 to cancellation at the top of the range), modified-Lentz continued
fraction on Γ(½, x²) above (empirics: the raw convergent is UNNORMALIZED —
exactly √π·erfc, so the 1/√π factor is applied; `FRAC_1_SQRT_PI` is behind
unstable `more_float_constants` on our MSRV → local literal). Pinned by the
new `roughness::graded_tests` (baked 17-digit libm references, ≤ 4e-16 abs
in the series arm, odd symmetry exact, guards) + **G13f** in
tests/test_graded.py: Rust kernel vs numpy/math.erf reference = **0.0**.

**K2 — twist schedule (twisted_stack) → `rotations::twisted_tensors`.**
Per-slice `rot_z(eps, twist·frac)` loop with the fraction schedule was Python
(the rotation itself was already Rust — same code path, so output is
bit-identical; G15d re-ran unchanged). Schedule now an enum (`Midpoint`,
`Endpoint`) in Rust with the exact former formulas (`(k+½)/n`, `k/max(n−1,1)`
via `n.max(2)−1`); Python maps the grid string → code (0/1) and marshals
interleaved-18 → Layers. Pinned by `twist_tests::twisted_schedule_matches_rot_path`
(slice-by-slice equality with the direct rot path, endpoint/n=1 edges,
batch-helper agreement).

**K3 — Pasteur mapping (chiral_layer) → `berreman::pasteur_tensors`.**
`eps = n²I, rho = −iκI, rhop = +iκI, mu = μI` was materialized in Python while
the convention lived in Rust test fixtures — two sources of truth for THE
Phase-10a physics. Now a single Rust helper that both the Task-0 tests and the
Python wrapper call (tests refactored; they additionally assert the helper's
tensors match the fixtures exactly). Regime policy kept exact: |κ| ≥ 0.2n is a
Python WARNING; **det = n²−κ² = 0 is deliberately NOT an error here** — that
case flows to the solver's honest-NaN path because G15a is the regression gate
for the §10.8 NaN backstops (a first draft hard-errored at |κ| ≥ n and broke
that gate arm; caught on re-run). New hard errors are pure input validity only
(n ≤ 0 / non-finite, μ ≤ 0 / non-finite — μ validity is a mild tightening,
documented). Pinned by `pasteur_helper_tests::pasteur_tensors_mapping_and_guards`.

**K4 — batched orientation (evaluate_tensor rotate) → `rotations::apply_rot_batch`.**
The per-wavelength `R·ε·Rᵀ` was a numpy einsum while `apply_rot` already
existed in Rust. New binding `rot_apply_matrix` (real 3×3 R over n tensors);
`rotate` is now REAL-only — complex orientation matrices are rejected with a
clear ValueError (previously silently accepted by einsum; an unconstrained
complex similarity transform was never a supported semantic). Pinned by the
batch test in K2's suite; the diag(2.25,2.89) G12/G15d paths re-ran green.

**Verified thin (no action):** `solve()`/`fields()` (marshalling + duplicated
validation that mirrors Rust errors — kept as defense in depth), `_squeeze`
(ergonomics, never touches z), `_apply_m16`/`cloude` (broadcast loops over
per-row Rust scalars — could batch later, not physics), `mueller_from_jones`,
all rot_* (validation only), `kappa_table` (validation only),
`grade_interface`'s shave/warn bookkeeping in `graded_stack` (geometric
orchestration), `_ubf_array`/dispatch in `berreman_materials.py` (mirrors
upstream's own Python split verbatim), `power_method` string mapping and `cx`
complex assembly (presentation/marshalling).

**Two API tightenings recorded (behavioral, intentional):** (1) unknown
`method=` strings now raise `ValueError` with the valid list instead of a bare
`KeyError`; (2) `grade_interface` rejects non-finite/non-positive
`total_width_nm` (previously a negative width would have built negative-
thickness layers silently). No other behavior change: full Python suite re-run
— G5 2.18e-14, G7b/G9d/G7d, G8b–e, G10b–d, G11a/c, G12a/b, G13b–f, G14a–c,
G15a–e, G15d pos/height/FWHM, e2e, FULL-PATH, energy gates all green; Rust 33
lib tests (+4 new) + 2 integration.

**Left as future work (explicitly, not silently):** batch the Mueller
`_apply_m16`/`cloude` Python loops into one Rust call per array (perf only);
vectorize `_flatten` marshalling (perf only); `cholesteric_stack`'s diag
assembly stays Python (Layer-list assembly, upstream-idiomatic).

### B.1 Toolchain pinning audit (vs upstream navette @ 4b59cd7 / 0.7.0)

Question raised post-review: why was the toolchain not pinned against latest
versions / latest rust? Comparison (upstream facts live-read from the checkout
root, `release.yml`, and fetched crate manifests — not recalled):

| aspect | upstream navette | ours (before) | ours (now) |
|---|---|---|---|
| rustc pin | none — `dtolnay/rust-toolchain@stable` in CI | none, no CI | none (rides stable, like upstream); validated on 1.98.1 (same rustc as their published 0.7.0 wheel) |
| MSRV declaration | none (workspace Cargo.toml has no `rust-version`) | none — "MSRV-aware" asserted in plan prose only | `rust-version = "1.85"` (edition 2024 floor; dominates pyo3/numpy 0.29's 1.83) |
| edition | 2024 (their core crate) | 2021 | 2024 |
| pyo3/numpy | 0.28 workspace pins (abi3-py312) | 0.23 (Phase-0 vintage of the `_smatrix`-style port; never bumped) | 0.29.2/0.29 — one minor AHEAD of upstream's 0.28 (latest per `cargo search`; migration was exactly one rename: `allow_threads` → `detach`) |
| abi3 | yes (`abi3-py312`) | no (cp313-tagged .pyd) | `abi3-py312` |
| Cargo.lock | committed (workspace root) | **gitignored (library habit — wrong for a wheel-shipped cdylib)** | committed (76 crates; pyo3 0.29.2, ndarray 0.15.6 for the upstream-adapter surface) |
| maturin pin | `>=1.5,<2.0` | same | same (local 1.14.1; crates.io latest 1.15.0 — caret range covers it) |
| requires-python | `>=3.12` | same | same |
| release CI | 3-OS wheels + OIDC PyPI + crates.io | none | none (deferred; upstream's release.yml is the template) |

**Root cause of the drift:** the port pinned pyo3/numpy 0.23 when it was
written against the `_smatrix` 0.23-era API surface and never revisited; the
"MSRV-aware, no new deps" decision was enforced in practice but never written
into `Cargo.toml` (`rust-version`), and the lock was ignored by library
crate habit while upstream treats the lock as part of the shipped artifact.
Fix verified end-to-end: `cargo check/test` (33+2 green) + maturin rebuild +
full Python gate suite re-run green on the 0.29 stack (13/13 files, same
observed values: e.g. G5 2.176e-14, G13f 0.0, code-5 |ΔT| 1.75e-2 diverges).

## Appendix C — Per-phase acceptance checklist (copy into each PR)

```markdown
- [ ] G1 cargo test --release green (paste count)
- [ ] G2/G3/G4/G5 prior gates green (paste observed worst values)
- [ ] G6 isotropic bit-identity unaffected (new paths default-off / gated)
- [ ] New gate G<n>a.. green (paste observed values + tolerances)
- [ ] Live-reference script output pasted (BerreMueller and/or B44)
- [ ] README + wrapper docstrings updated
- [ ] No new crate dependencies (or justified + MSRV-checked)
- [ ] No panics in core paths (Option-propagation; pybind maps None -> NaN/n_failed)
```
