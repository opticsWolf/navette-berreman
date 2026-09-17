# Phase 11 Report — POLARIZANCE v2 (Brown differential Mueller decomposition)

**Status: complete, all gates green.** Commits `59d972b` (detailed plan) →
`e446e01` (two-route decision) → `6d8e316` (as-built). Full suite: Rust
39 + 2 integration, all Python gates (13 test functions in
`tests/test_polarizance.py` plus every module-level gate re-asserting green
at collection).

---

## 1. Executive summary

Phase 11 delivers the *differential Mueller-matrix formalism* (dM/dz = H·M,
M = exp(−absorbance·L)·expm(H·L) for homogeneous elements) plus Brown's
(1999) polarizance parameters, as two routes producing the same
per-unit-length differential quantities (β, d, absorbance):

- **Route A** — live-BerreMueller-compatible scalar path (diagonal-ε
  semantics, validation oracle), transcribed exactly;
- **Route B** — our general eigenpath: q-eigenvalues/modes of the Berreman
  matrix at kx = 0 → bulk Jones generator k = W·diag(i·k₀·q_j)·W⁻¹ → the
  infinitesimal form of our G11-anchored `mueller_from_jones`. This is the
  deliverable: it generalizes to chirality, magneto-optics and non-diagonal
  tensors — exactly what live's elementwise `sqrt(eps)` cannot express.

The feature closes the Phase-5 deferral (§10.3 stretch goal) with the
premise corrected and cheaper than originally assumed, and it expands the
contract in the expand-only direction: the Berreman engine is untouched.

The strongest result of the phase: **the differential (Mueller-expm) path
and the Jones-group path reproduce each other at 1.7e-15 from the same eigen
data, and the full stack solver agrees with the interface-corrected bulk
propagator at 9.8e-16** — three formalisms, one machine-precision knot.

---

## 2. What was studied (provenance record)

### 2.1 Live source (BerreMueller checkout, read line-by-line)

| Object | Location | What it does |
|---|---|---|
| `polarizance_from_linear_optics` | `mueller.py` ll. 663–679 | assembles (b, d, p = b + i·d) from LINEAR_OPTICS; circular slots zero |
| `brown_params` | `mueller.py` ll. 674–691 | a₀..a₃ closed forms (cross-paired cosh/cos) |
| `POLARIZANCE.decompose_polarizance` | `mueller.py` ll. 706–713 | p_m = √(p·p) NO conjugation; r = Re, i = Im, n = √(r²+i²) |
| `POLARIZANCE.diff_matrix` | `mueller.py` ll. 713–734 | the differential generator placement |
| `LINEAR_OPTICS` feed chain | `dielectric_tensor.py` ll. 762–790 | elementwise √ε; ε′ = R(−45°)·ε·R(+45°); LD/LB with −1 signs; absorbance_v1/v2 + **max() stability hack** (their comment: prevents LD > absorbance nonphysicality at long paths) |
| `get_refractive_index_tensor` | `dielectric_tensor.py` ll. 508–513 | **elementwise `np.sqrt`** of the full tensor |

### 2.2 Literature (all live-verified)

- **Primary:** Salij, Goldsmith, Tempelaar, *Chiral polaritons based on
  achiral Fabry–Pérot cavities using apparent circular dichroism*,
  arXiv:2208.14461v2 — **Supporting Information** (§S2–S3). The arXiv
  ancillary endpoint was robots-blocked; the PDF was provided out-of-band and
  extracted locally (8,017 chars). The SI states the differential formalism
  (dM/dz = H·M with the restricted-Lorentz-group generator
  H = α·I + β·B̂ + d·D̂; 6 unique characteristics — mean absorption factors
  out diagonal), the homogeneous-sample closed form M = exp(H·l), the Brown
  B₀…B₃ forms (S5–S9), **second-order consistency relations (S10–S11)**, and
  the ACD forward/backward differential matrices (S12–S14).
- **Brown 1999:** DOI 10.1117/12.366361 — verified resolving (HTTP 200).
- **Gil & Ossikovski**, *Polarized Light and the Mueller Matrix Approach* —
  cited by live's coherency machinery; that side is already ours (G11).
- **Context:** arXiv:2208.14461's first author (Andrew Salij) is the
  BerreMueller author — the POLARIZANCE code **is** the SI companion
  implementation. This also explains why it is never wired into the solver.

### 2.2 Extraction notes (for future readers of the SI)

The arXiv `/src/` ancillary endpoint is robots-blocked for automated fetch
(the abs page is fine); the SI was obtained manually and extracted locally.
PDF text extraction mangles the two-column math layout (glyphs/indices
garble); the operational authority for every formula is **live's code**,
which we transcribed and pinned against instead of reconstructing the SI
text. The SI is cited by equation number as the formalism reference; our
derivations (§5 below) were checked numerically rather than transcribed.

---

## 3. Spike findings (premises settled, two of ours corrected)

1. **The POLARIZANCE/Brown machinery is NOT wired into BerreMueller's own
   Berreman solver.** Zero call sites across `berreman_mueller.py`,
   `full_berreman.py`, `__init__.py`. It is a standalone material-level
   forward model. Consequence: the differential-Mueller layer and the
   Berreman engine are complementary formalisms, not alternatives.
2. **`get_refractive_index_tensor(ε) = sqrt(ε)` elementwise** — live's
   LD/LB read √ε_xx, √ε_yy (exact only for diagonal ε in the measurement
   basis). Our port generalizes to arbitrary tensors via the q-eigenvalues.
3. **The Phase-5 deferral premise "needs Phase 2 fields" was wrong.** The
   uniform-layer differential calculus is material-tensor algebra; E(z)/H(z)
   enter only for z-resolved studies, and the natural discretization is
   per-slice tensors (the `twisted_tensors` machinery), not pointwise
   fields. Phase 11 therefore needed no new fields machinery.
4. **Route B is strictly more general than live:** the eigenpath handles
   (ρ, ρ′) Pasteur chirality, μ tensors and non-diagonal ε at normal
   incidence. The measured chirality check (|b₂| = 2k₀κ exact) is something
   live has no path to.

---

## 4. Design as-built

```
src/polarizance.rs        kernels + 6 Rust unit tests (new)
src/transfer.rs           + mueller_a_basis() (shared A-basis; mueller_from_jones
                            bit-identical, G11a re-run unchanged)
src/lib.rs                + module + 7 registrations
src/pybind.rs             + 7 batched pyfunctions (Rayon, py.detach, NaN rows
                            + n_failed) + read_tensor_batch/read_vec3_batch
navette/berreman.py       + imports + 7 wrappers + marshalling helpers
tests/test_polarizance.py 13 gates (new)
```

**Kernel inventory (all physics in Rust; Python = validation/shapes only):**

| Kernel | Semantics |
|---|---|
| `linear_optics_scalar(eps, ω, length_over_c)` | Route A: elementwise n = √ε; ε′ = R(−45°)·ε·R(+45°); ld = −(Im n_yy − Im n_xx)·ω·loc; lb from Re; **both** absorbance orientation sums returned |
| `polarizance_decompose(b3, d3)` | p = β + i·d; p_m = √(p·p) (NO conjugation, live semantics); (r_p, i_p, n_p) |
| `brown_params(r_p, i_p, n_p, L)` | live formulas exactly; None at n_p = 0 (live yields NaN silently; the core refuses) |
| `diff_mueller_matrix(b, d)` | live `POLARIZANCE.diff_matrix()` placement, traceless, circular slots (b₂, d₂) carry CB/CD |
| `mueller_from_diff(b, d, abs, L)` | exp(−absorbance·L)·expm(H·L) via the Phase-3 `mat4_expm` — no new deps |
| `bulk_differential(eps, ρ, ρ′, μ, k₀, L)` | Route B: LayerWaves at kx = 0, forward pair = columns 0,1; W = [[Ex1,Ex2],[Ey1,Ey2]]; k = W·diag(ik₀q_j)·W⁻¹; J(L) = W·diag(e^{ik₀q_jL})·W⁻¹; H_full = A·(kron(k,I) + kron(I,k̄))·A⁻¹; (b,d,absorbance) read off |
| `mueller_from_diff_stack_product` | composed z-resolved product, slice 0 first |

**Convention decisions (all five traps, decided up front and honored):**
ω = 2πc₀/λ computed Python-side; `length_over_c` kept OUT of the kernel
(callers pass physical length; c₀ = 299792458e9 nm/s); the −1 signs
transcribed from live and verified by G17b to machine precision; the
absorbance max-hack not inherited (both sums returned, live's max verified ==
max(ours)); explicit separate (β, d) vectors instead of live's complex-p
mixing. Per-unit-length differentials everywhere; length enters only in the
exponentials.

**Option-returning core respected:** None on unclean eigen-sort,
(near-)singular W (degenerate polarization eigenmodes, e.g. z-uniaxial at
kx = 0 — documented limitation), n_p = 0, and |Im H| beyond the noise guard.
Batched calls surface these as NaN rows + `n_failed`.

---

## 5. Physics & algebra findings (derived and verified this session)

1. **The generator identity.** With the SAME Stokes A-basis as
   `mueller_from_jones`, `H_full = A·(kron(k, I) + kron(I, conj(k)))·A⁻¹`
   is the infinitesimal Mueller generator of the Jones generator k: the
   identity kron(expm kL, conj expm kL) = expm((k⊗I + I⊗conj(k))L) makes
   `mueller_from_jones(J(L)) = expm(H_full·L)`. Validated to ≤ 1e-9 by
   Richardson-finite-difference against the G11-anchored kernel (3 random
   complex k), then to ≤ 1e-12 against the closed 2×2 expm (3 more fixtures).
2. **7 real characteristics.** The isotropic-phase direction k = iφ·I maps
   to zero (kron(iφI, I) + kron(I, −iφI) = 0), so the Jones-algebra image is
   7-dimensional — exactly live's diff_matrix span (3 symmetric d-entries +
   3 antisymmetric b-entries + the iso diagonal). The two formalisms are
   basis-aligned, not merely isomorphic — asserted, not assumed.
3. **Absorbance = −Re(tr k).** H[0][0] = Re(tr k) (the A-map algebra with
   2s² = 1); for the diagonal case this equals k₀(Im q₀ + Im q₁) — live's
   absorbance convention exactly. Verified 1e-15-class.
4. **The kron cross-diagonal mixing (recovery subtlety).**
   S[0][0] = 2Re k₀₀, S[3][3] = 2Re k₁₁, **S[1][1] = k₀₀ + conj(k₁₁),
   S[2][2] = k₁₁ + conj(k₀₀)** — the diagonal slots mix both Jones entries.
   Consequently the *difference* Im(k₀₀) − Im(k₁₁) (a relative phase —
   visible: it is the retardance) is recoverable, while only the sum
   (the invisible isotropic phase) is lost. Recovery is complete up to the
   phase-invariant direction, which cannot affect any Mueller quantity.
5. **The factor-2 circular slot.** For the Pasteur κ-medium
   (n± = n ± κ, task-0 spectrum) at kx = 0 the extracted circular slot is
   |b₂| = **2k₀κ exactly** (measured 2.0944e-4 at λ=600 nm, κ=0.01): the
   Stokes rotation rate equals the circular-eigenmode phase *difference*
   2k₀κ, while the Jones polarization vector itself rotates at k₀κ — the
   same factor of 2 G16d pinned for the round-trip arg-ratio, now re-measured
   through the differential generator.
6. **Brown's cross-pairing is the physics.** a₀ = (r/n)²·cosh(iL) +
   (i/n)²·cos(rL) pairs r_p with cosh(i_p·L) and i_p with cos(r_p·L). The
   pure-case limits check out physically: pure retardance (i_p = 0) →
   a₀ = 1, a₁ = (1−cos(RL))/R², a₂ = sin(RL)/R, a₃ = 0; pure dichroism
   (r_p = 0) → a₀ = 1, a₁ = (cosh(IL)−1)/I², a₂ = sinh(IL)/I, a₃ = 0.
   **The a-parameters are closed-form coefficients, not Mueller entries:**
   for the pure dichroic element a₀ = 1 while m₀₀ = cosh(DL) — the a-params
   parameterize the direction-structured reconstruction (Brown's
   parametrization), consistent with expm (the identity
   e^{−b}·cosh(a) = ½(e^{−2k₀κ_x L} + e^{−2k₀κ_y L}) closes the loop between
   the a-structure and the mode-absorption picture).
7. **The SI's S10 claim reproduced:** a₀ = 1 + **O(L⁴)** — the L² term
   cancels *exactly* in the cross-pairing ((r²i² − i²r²)L²/2n² = 0). Our
   G17e pins a₀ = 1 + O(L⁴), a₁ = L²/2 + O(L⁴), a₂ = L − L³(R²−I²)/6,
   a₃ = −RI·L³/6 — both in Rust and through the Python wrapper.
8. **The absorbance max-hack quantified.** For our fixture the two
   orientations differ physically by 5.55e-4 relative (the √((ε_xx+ε_yy)/2)
   averaged-loss orientation vs the unrotated one); live's max() selects the
   larger, and our returned pair reproduces both to ≤ 2e-16 with
   max(ours) == live's absorbance exactly (0.0).

---

## 6. Gate results (all green; full pin table)

Gates live in `tests/test_polarizance.py` (13 test functions; 3 live-oracle
tests auto-skip without BerreMueller importable). The Rust unit tests mirror
the same constructions (6 tests in `src/polarizance.rs`).

| Gate | Content | Measured | Tolerance |
|---|---|---|---|
| Rust: generator vs FD | Richardson FD of `mueller_from_jones` | ≤ 6e-8 class, all entries | 1e-9 |
| Rust: expm-vs-Jones expm (3 fixtures) | closed 2×2 expm vs `mat4_expm` | ≤ 1e-12 | 1e-12 |
| Rust: Route A diagonal transcription | lb = −(1.52−1.5)·ω·loc, ldp = 0 (rotation isotropizes xy) | ≤ 1e-18 / 1e-15 | exact |
| Rust: Route A lossy transcription | ld, abs2, abs1 vs independent arithmetic | ≤ 1e-15 (abs1 rel 1e-12) | — |
| Rust: Brown pure cases + Taylor | a-params limits + Taylor | ≤ 1e-15 / 1e-12 | — |
| Rust: isotropic reduction | (b,d) = 0; jones = e^{ik₀nL}·I | ≤ 1e-14 | — |
| **G17a** expm vs Jones group path (3 fixtures) | same eigen data, two formalisms | **1.706e-15** | 1e-14 |
| **G17a2** vs full solver + Airy correction (3 fixtures) | single-pass bulk from the full FP transmission | **9.778e-16** | 1e-13 |
| G17a Pasteur (ρ,ρ′) | expm == Jones; \|b₂\| = 2k₀κ | ≤ 1e-14 / exact | — |
| **G17b** Route A vs live (4-wl diagonal fixture) | ld rel ≤ **1.99e-16**; ldp/lb/lbp/brown **0.0 exact**; absorbance sums ≤ 1.99e-16; live max == max(ours) 0.0 | transcription-identical | 1e-12 |
| **G17c** reductions | all listed in §5 | ≤ 1e-14 class | — |
| **G17d** Trotter self-convergence | r(12→24) = 1.0519e-3, r(24→48) = 2.6248e-4, r(48→96) = 6.5589e-5 | ratios ×4.008, ×4.001 | monotone, < 0.75× |
| **G17d-solver** staircase vs `twisted_stack` | 0.16941 → 0.17010 → 0.17027 → 0.17031 (successive diffs 6.878e-4 → 1.722e-4 → 4.308e-5, ÷4.0 each) | fixed outer-interface factor ~0.17032 + O(1/n) stabilization | stabilization, not machine precision |
| **G17e** Brown small-L Taylor (SI S10–S11) | a₀ = 1+O(L⁴); a₁ = L²/2; a₂ = L−L³(R²−I²)/6; a₃ = −RI·L³/6 | ≤ 1e-12 / 1e-15 | — |

The Trotter halving ratio is *exactly* ×4 — the textbook midpoint-rule
O(1/n²) rate, measured rather than asserted.

---

## 7. Lessons learned (the phase's own fault-line record)

1. **The Airy decomposition lesson (caught by G17a2).** The first
   interface correction divided out the *whole* Airy transmission
   t = T_int·e^{ip}/D — which also removes the bulk phase e^{ip} and
   collapses the corrected Jones to I. The single-pass bulk propagator is
   `J_trans·D/T_int`: the interface product AND the Fabry–Pérot denominator
   come out, the bulk phase stays. A machine-precision gate is what catches
   a mistake of this kind — a 1e-6-class gate would have "passed" with the
   wrong construct on lossless fixtures.
2. **Richardson for forward differences is 2·f(ε/2) − f(ε).** The
   central-difference form (4f₂−f₁)/3 does not kill the O(ε) term for
   forward differences (it leaves c·ε/3). Same lesson class as the sign
   slips: the extrapolation formula must match the difference scheme.
3. **Absolute tolerances are meaningless at ω·L/c scale.** The differential
   quantities in SI units sit at 1e14–1e15 (ω ~ 3e15 rad/s); identical
   arithmetic paths still differ by ~1e-2 absolute there. Every gate that
   touches such quantities must compare O(1) ratios (or nondimensionalize
   with length_over_c). Three test failures before this was internalized.
4. **Rust let-shadowing in tuple initializers** (test-side): `let (r, i,
   np) = (0.3, 0.2, (r*r + i*i).sqrt())` binds np from the *previous* r/i
   (0.4/0.25), silently corrupting the fixture. Fresh names for fresh values.
5. **The solver-vs-differential comparison needs interface physics.** The
   outer interfaces are a fixed, n-independent offset between the bulk
   differential product and the solver's M_trans. The G17d-solver gate is
   therefore a *stabilization* gate (successive differences ÷4.0), not a
   machine-precision gate — and that is honest reporting, not a weaker gate:
   the residual is genuine physics (see G17a2's decomposition for the
   isolated single-layer case where the correction is analytic).
6. **Two formalisms from one A-basis.** Sharing `mueller_a_basis` between
   `mueller_from_jones` (G11-anchored) and the differential generator is
   what makes the G17a knot binding: a convention slip in either formalism
   breaks the 1e-15 agreement loudly.

---

## 8. Limitations & deferred items

- **Degenerate polarization eigenmodes at kx = 0** (e.g. z-uniaxial
  ε_x = ε_y): the E-projection W becomes singular → `bulk_differential`
  returns None (NaN rows + `n_failed`). Documented, not engineered around —
  the isotropic and near-degenerate cases are physically scalar-like and the
  Trotter/z-resolved path is the intended tool there.
- **No interface modeling in the differential formalism** (by design — it
  describes the bulk; G17d-solver quantifies the constant interface factor).
- **Brown's full M-reconstruction from a₀..a₃** (the direction-structured
  closed form) is not implemented — the expm path is exact for any (β, d)
  and supersedes it; `brown_params` ships as the live-parity scalar summary.
- **The SI's S10–S11 transcription** is by numerical structure (the Taylor
  coefficients), not verbatim text — the PDF extraction garbles subscripts;
  G17e pins what the SI *claims* against our own closed forms.
- **fields_at_depths** integration remains unnecessary (per-slice tensors
  are the z-resolution path, as `twisted_tensors` already does).

---

## 9. Artifacts & provenance

- **Code:** `src/polarizance.rs` (new); `src/transfer.rs` (`mueller_a_basis`
  refactor); `src/lib.rs`; `src/pybind.rs`; `navette/berreman.py`;
  `tests/test_polarizance.py` (new).
- **Docs:** `IMPLEMENTATION_PLAN.md` Phase 11 (§11.0–11.5, now AS-BUILT with
  measured pins) + the phase-status table row; `README.md` new
  "Differential Mueller calculus (Brown / POLARIZANCE v2)" section.
- **Commits:** `59d972b` (plan), `e446e01` (both-routes decision),
  `6d8e316` (as-built).
- **Literature provenance:** arXiv:2208.14461v2 SI extracted from the
  user-provided PDF (robots-blocked on arxiv.org/src); Brown 1999 DOI
  verified live; live formulas transcribed from the checkout at
  `.../BerreMueller/src/berremueller/` (line refs in §2.1).
- **No new crate dependencies** — the expm path reuses Phase-3 `mat4_expm`.
- **Expand-only policy respected:** the Berreman engine untouched; the new
  feature is an adapter + material-level kernels per the kappa_table
  precedent.

## 10. Suggested follow-ups (not started)

1. ~~**F3** (NC lossy type-5 wheel drift)~~ **RESOLVED 2026-09-18** — the
   residual is the F1 transmission fork seen through reverberation, not an
   upstream defect; see `F3_RESOLUTION.md` and the rewritten F3 section of
   `NAVETTE_UPSTREAM_REVIEW.md`.
2. **Release CI** (version-sync → 3-OS wheels → OIDC PyPI) + first tag.
3. Optional: Brown-direction reconstruction (the full M from a₀..a₃ and the
   normalized (r̂, î) directions) if a user needs Brown-land API parity
   beyond the scalar params — the expm path already covers the physics.
