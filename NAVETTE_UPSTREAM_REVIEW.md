# Upstream review: `navette` 0.7.0 roughness code paths

Auditor's context: we maintain `navette-berreman`, an independent 4×4
Berreman/Müller solver validated against this crate's `smatrix` engine
(codes 0–4 agree at 2.2e-14 on identical stacks). These findings come from
that validation work. Offered upstream in good faith; severity is judged by
physics impact, not by code quality (the codebase is, notably, honest about
F1 in its own docstrings — the dispute is about the model, and the fix is
half a line).

- **Scope:** `navette` 0.7.0 from crates.io (source audit) + the PyPI wheel
  0.7.0 (live probing). Paths below are `src/smatrix/…` in the crate and
  `navette/…` in the wheel.
- **Method:** source read of the `.crate`, direct Rust linkage tests against
  `w_function_inner` / `nevot_croce_factors` (`tests/navette_parity.rs` in
  our repo), and stack-level probing through the wheel (`tests/test_roughness.py`).
- **Out of scope:** the `1/d` determinant typo in `andrewsalij/BerreMueller`
  `full_berreman.py` is a different project, not filed here.
- **Living document (standing rule):** this file is updated as validation
  proceeds — every suspected upstream defect gets an F-numbered entry with
  location, measured evidence, severity, and a suggested fix, plus a line in
  the Changelog below. Findings live here, never only in chat; spec impact
  (if any) is cross-referenced from `IMPLEMENTATION_PLAN.md`, never mixed in.

---

## F1 — Type-5 transmission factor has the wrong sign (model dispute, evidenced)

**Location:** `optics_core.rs:244-245` (docstring), `:305-313` (`nevot_croce_factors`),
applied at `coherent_block.rs:123-124`, `:329-331`, `solver.rs:2487-2488`;
model stance documented in wheel `structure/types.py:42-80`.

The code returns `(f, ga)` with `ga = exp(+((kz1−kz2)·σ)²/2)` and applies
`(r12·f, r21·f, t12·ga, t21·ga)`. Upstream's stated rationale (quoted from
`types.py`): *"transmission is enhanced … so that R_spec + T_spec = 1 to
first order in sigma²"* under a "laterally-uniform index grading" picture.

Why we dispute it:

1. **It contradicts Névot–Croce theory**, the model the code is named after.
   The textbook NC result (Névot & Croce, *Rev. Phys. Appl.* **15**, 761
   (1980)) damps transmission by `exp(−(Δkz)²σ²/2)` — roughness *decorrelates*
   the transmitted beam; it cannot enhance it. There is no grading picture in
   NC: the interface is rough, and the missing energy is diffuse scatter
   (Sinha–Sirota–Garoff–Stanley, *Phys. Rev. B* **38**, 2297 (1988)).
2. **It contradicts the crate's own types-1–4 arm.** Two lines below the NC
   branch, `coherent_block.rs:130` damps transmission with
   `w_function_inner((kz1−kz2)·σ, rtype)` — the decaying form — for every
   other model. Type 5 is the only arm that enhances transmission, so the
   inconsistency is internal, not just theoretical.
3. **First-order cancellation does not make it physical.** Upstream's own
   docs concede the direction is wrong (*"real roughness scatters energy out
   of the specular beam, so the physically correct result is R+T slightly
   below 1. R+T > 1 has no [physical reading]"*) and quote unphysical output
   (T = 1.0768, R+T up to 49.4 in `optics_core.rs` docs). A model whose error
   term has the wrong sign and grows without bound is not fixed by a validity
   budget — the budget concedes the usable domain excludes routine optical
   coatings (their table: 1% budget at σ ≈ 6.5 nm for Δn = 1.35 @ 550 nm).
4. **Independent evidence:** our solver implements the decaying NC factor and
   measures R+T ≤ 1 at *all* σ × angles × codes including absorbing media
   (`tests/test_roughness_energy.py`), while diverging from the wheel's type-5
   transmission by min |ΔT| = 1.75e-2 on a standard two-film fixture
   (`tests/test_roughness.py` fork-guard). At Rust level we pin the exact
   relation `ga_upstream · ga_theory == 1` (`tests/navette_parity.rs`).

**Suggested fix (one line):** negate the exponent —
`ga = (-(d*d) * 0.5).exp()` — making type 5 consistent with NC theory *and*
with the crate's own types-1–4 transmission arm. Fallback if the grading
picture must stay: gate type 5 with a warning when the injected energy
`((kz1−kz2)·σ)²/2` exceeds the caller's budget instead of documenting the
unphysical regime only. Either way, the `types.py` "constructed to conserve
specular energy" claim should go — first-order cancellation is not
conservation.

**Severity: medium-high.** Silent energy creation (T > 1, A < 0, unclamped)
in a default-facing roughness model, reachable at ordinary optical contrast.

---

## F2 — Truncated `SQRT3` literal caps code-1 precision (trivial fix)

**Location:** `optics_core.rs:18`: `pub const SQRT3: f64 = 1.73205080757;`
(11 digits; true value 1.7320508075688772…, relative error 7e-13).

**Measured impact:** type-1 form factor differs from full-precision
`sin(q√3)/(q√3)` at ~6e-14 (q = 0.3i; error scales as |q|·|W′|·δs, verified
to 2.5e-12 at q = 3.7+1.2i). This dominates stack-level code-1 parity floor
(~2e-14 on our G5 fixture — the other codes sit at 1e-16–1e-15).

**Suggested fix:** spell the full literal. Zero behavioral risk (monotone
accuracy gain; golden values shift by ≤ 1e-12, inside every published
tolerance in `validation/`).

**Severity: low.** Below all gate thresholds, but it is a free digit of
accuracy and a standing source of confusing parity reports for downstream
validators (like us).

---

## F3 — RESOLVED: the lossy type-5 reflection residual is the F1 fork seen
## through reverberation (2026-09-18)

**Mechanism identified and proven. The residual is not an upstream defect
and not a wheel/source drift — it is the expected consequence of our F1 fix
observed through the film's internal reverberation.**

Against any wheel (0.7.0 AND 0.7.7: both give 1.41e-8 on the standard
fixture at wl = 550 nm; a locally built 0.5.0-source engine gives 1.28e-7),
type-5 *reflection* on a lossy fixture (400 nm film, n = 2+1.5j) differs at
~1e-8, scaling exactly as σ².

**Why the original ruling-out was wrong:** the review dismissed
reverberation feedback with "round-trip residue ~1e-12 in power ⇒
ga-mediated feedback ~1e-16". That squares the fork twice. The fork enters
R *linearly* through the amplitude cross-term of the coherent sum:
the second-bounce amplitude is t21·t12·r₂·φ² ~ 1.4e-3 (power ~2e-6, NOT
1e-12 — the 1e-12 figure included both r-factors squared), its ga-vs-f
difference is (f−ga) ~ σ²·O(k²) ~ 5% at σ = 8 nm, and the cross-term with
the main reflected beam 2·Re(r₁·f·conj(δ₂-bounce)) lands at ~1e-8. Exactly
the observed magnitude and the exact σ² law.

**Evidence chain (all measured 2026-09-18):**

1. **Probe A (crate source, 0.7.7):** `tests/navette_parity.rs` re-run
   against crates.io navette 0.7.7 (Cargo.lock bumped): the published
   crate's `nevot_croce_factors`/`w_function_inner` still match our formula
   at the pinned 1e-15 on lossy kz. The 0.5.0→0.7.7 source diff shows the
   NC factors bit-identical (0.7.7 centralizes them into `optics_core`
   after the R1.1 desync fix; `forward_branch` is the old inline cos rule,
   asserted bit-identical by the crate's own test).
2. **Bare-interface probe:** the wheel's effective f on the lossy interface
   (two-layer stack, one dressed interface) reproduces |r·f|² to 1.7e-16 —
   identical on the 0.7.7 wheel and a 0.5.0-source build. The NC factor in
   the wheel is exact.
3. **Asymmetry probe:** dressing ONLY the exit interface agrees **exactly**
   (dR = 0.0, both polarizations, both angles) — a semi-infinite exit has no
   reverberation, so the fork cannot leak back. Dressing only the FRONT
   interface carries the full σ² residual (6.28e-8 at 0°, 400 nm film).
4. **FP-model magnitude:** a two-interface numpy Fabry–Pérot with (f, ga)-t
   vs (f, f) transmission dressing gives 1.35e-7 at the front-only fixture —
   the same order as the engines (the 2× gap is the admittance-normalization
   convention in the model, not the mechanism).
5. **Knockout — the reverberation kill:** scaling the film thickness,
   the engine residual collapses 6.284e-8 (400 nm) → 5.153e-13 (800 nm) →
   **0.0 exactly** (1600 nm), i.e. exponentially with e^{−4·Im β} — the
   reverberation channel. At the module's wl = 550 nm the pin measures
   2.55e-9 → 1.63e-14 → 0.0 (the absolute value is phase-dependent; the
   exponential envelope is what is pinned).

**Resolution:** expected divergence given the documented F1 fork. Our
transmission block keeps the energy-conserving ga (the Phase-8 fix);
upstream applies f to transmission as well (the F1 energy bug); any
internally reverberating structure mixes the two back into R at O(σ²).
The σ²-law constant (5.0e-11/nm² at the review's fixture, σ5-scaled with
σ6 fixed) is the fork-through-reverberation signature and is now pinned as
a regression gate (`tests/test_roughness.py` `_f3_thickness_pin`): the
residual must collapse with film thickness (400 nm ≤ 1e-7, 800 nm ≤ 1e-11,
1600 nm == 0.0). A wheel that starts matching us at 400 nm would mean the
fork closed (F1 adopted upstream or restored here) — the thickness pin and
the transmission fork-guard together make that loud.

**Upstream ask, revised:** no longer needed for F3 (the wheel is vindicated
and the fork is a documented model dispute under F1). The one re-export
(`nevot_croce_factors` in `navette._smatrix`) remains a nice-to-have for
downstream validators, not a defect.

**Severity: closed (informational).**

---

## Changelog

- 2026-09-16 (P8): file created with F1 (NC transmission sign), F2
  (truncated SQRT3), F3 (open lossy-reflection residual). Repro pins:
  `tests/navette_parity.rs`, `tests/test_roughness.py`,
  `tests/test_roughness_energy.py`.
- 2026-09-16 (P9): F4 (`table_nk` asserts → `PanicException` on malformed
  input; ours raises `ValueError`). Repro: `src/materials.rs` unit test +
  wheel probe in session notes.
- 2026-09-18 (F3): **F3 RESOLVED** — the lossy type-5 reflection residual
  is the F1 transmission fork (our ga vs upstream f-on-t) leaking back into
  R through internal reverberation; proven by the front/exit asymmetry
  (exit-only agrees at 0.0 exactly), the σ² law, and the reverberation kill
  (6.28e-8 → 5.15e-13 → 0.0 at 400/800/1600 nm). Wheel 0.7.7 re-audited:
  crate source still bit-identical to our NC formula (parity tests re-run
  green); wheel bare-interface f exact at 1.7e-16; locally built 0.5.0
  engine reproduces the residual (1.28e-7) — mechanism version-stable.
  Regression pin added (`_f3_thickness_pin` in `tests/test_roughness.py`).
- 2026-09-16 (P10): reviewed, nothing new. The det-typo's chiral consequence
  (live `full_berreman` scales every a_i by det² = (εμ−κ²)² with κ ≠ 0, so it
  can never oracle a chiral stack) is already covered by the §0.5/F1-adjacent
  record and IMPLEMENTATION_PLAN.md §10.1; no new upstream defect filed.

## F4 — `table_nk` panics across PyO3 on malformed input (low, robustness)

**Location:** crate `src/materials/table.rs` (`assert!(grid >= 2)`,
`assert_eq!(grid/n)`, `assert_eq!(grid/k)`); wheel `materials/__init__.py`
passes `n_data`/`k_data` through without length validation.

**Measured:** 1-point grid → `PanicException: table_nk needs at least 2
grid points, got 1`; mismatched k length → `PanicException: assertion
left == right failed: grid/k length mismatch`. A Rust panic crossing the
Python boundary (PyO3 converts to `PanicException`, but the unwinding is
still a library panic on user input — callers catching `ValueError` miss
it, and `except Exception` hygiene varies). Our adapter validates first and
raises `ValueError` (proven in `src/materials.rs::table_validates_instead_of_panicking`).

**Suggested fix:** `Result<Array1, String>` return (like the KK-trio
kernels already use) + Python-side length check in `evaluate`. Zero physics
impact; pure API hygiene.

**Severity: low.** Only reachable with malformed caller input, but panics
in libraries are never correct handling of user error.

## Reproducibility

- Crate source: `navette` 0.7.0 from crates.io (static CDN), rustc 1.98.1.
- Wheel: PyPI `navette` 0.7.0 + `scipy`, CPython 3.13 / win_amd64.
- Our pins: `tests/navette_parity.rs` (crate-level, `navette = "0.7"`
  dev-dependency), `tests/test_roughness.py` (wheel-level, codes 0–4 at
  2.2e-14 + code-5 split arms), `tests/test_roughness_energy.py` (R+T ≤ 1).
