# F3 Resolution Report — the lossy type-5 reflection residual

**Status: RESOLVED (2026-09-18) — not an upstream defect.** Commit `03a0c21`.
The residual is the **F1 transmission fork seen through reverberation**;
the wheel is vindicated and the original "mechanism open" ruling-out
contained a wrong reverberation estimate, corrected below.

---

## 1. The problem (as filed, 2026-09-16)

Against the PyPI wheel, Névot–Croce (type-5) *reflection* on a lossy
fixture — one 400 nm absorbing film, n = 2 + 1.5j, σ_front = 8 nm,
σ_exit = 12 nm, exit n = 1.5 — differed from our engine at max 1.4e-8,
scaling *exactly* as σ² (constant 5.0e-11/nm² at the σ5-sweep fixture).
What had been ruled out: the dressing formula (crate-level parity 1e-15),
the wavevectors (codes 1/4 match the same fixture at 1e-14), and
reverberation ("round-trip residue ~1e-12 in power ⇒ ga-mediated feedback
~1e-16, four orders short"). Suspects on file: (a) wheel-build vs
published-crate-source drift in the NC path; (b) solver-level NC handling
of complex kz outside the shared `(f, ga)` helper. Filed as an open
question, explicitly not an accusation.

## 2. Trigger and first check (the upgrade tripwire)

`uv pip install -U navette` moved the wheel 0.7.0 → **0.7.7** (numpy
2.5.3 alongside). Watch-item actions executed:

- the wheel `__init__.py` guard re-applied
  (`__init__.py.disabled_by_navette_berreman_G5_reapplied_0.7.7`);
- `Cargo.toml`'s floating `navette = "0.7"` pin tightened to `0.7.7`,
  `cargo update` applied (requires Rust 1.88 — fine on our @stable ride);
- `tests/navette_parity.rs` re-run against the 0.7.7 crate: green;
- the F3 repro arm re-run against the 0.7.7 wheel: residual **1.41e-08 —
  bit-for-bit the 0.7.0 value**. First finding: not a one-off build
  artifact; the mechanism is version-stable.

## 3. The evidence chain (five probes)

### Probe A — the published crate source is clean

The crates.io 0.7.7 source (already in cargo's registry cache) was diffed
against the 0.5.0 checkout and the parity test re-run at 0.7.7:

- `nevot_croce_factors` in 0.7.7 is **bit-identical** to ours:
  `f = exp(−2·kz1·kz2·σ²)`, `ga = exp(±(Δkz)²σ²/2)` (the sign question is
  F1's dispute, unchanged).
- 0.5.0 → 0.7.7 changed the NC path only by *centralizing* the factors
  into `optics_core` ("so all four interface builders stay bit-identical —
  see R1.1, where a fix landed in two of them and silently desynchronised
  the rest") and replacing the inline cos-branch rule with
  `forward_branch` — which the crate's own test asserts is the old inline
  rule bit-for-bit.
- Parity tests against the 0.7.7 crate source: green at the pinned 1e-15
  on lossy kz.

⟹ The published source is clean at both audited versions. The residual
cannot be wheel-vs-crate-source drift in the helper.

### Probe B — the wheel's effective NC factor is exact

A bare-interface fixture (two-layer stack: ambient 1.0 | film half-space
2+1.5j, one dressed interface, no propagation, no reverberation) compared
against the analytic `|r·f|²` computed from our formula:

| σ (nm) | wheel R | \|r·f\|² | diff |
|---|---|---|---|
| 8 | 0.274668460533 | 0.274668460533 | 1.11e-16 |
| 4 | 0.285266206749 | 0.285266206749 | 1.67e-16 |
| 2 | 0.287978927995 | 0.287978927995 | 1.67e-16 |
| 1 | 0.288661129459 | 0.288661129459 | 5.55e-17 |

Identical on the **0.7.7 wheel** and on a **locally built 0.5.0-source
engine** (built via `uv pip install` of the checkout into a scratch venv).
The NC factor inside both engines is exact. The residual is structural.

### Probe C — asymmetry: the residual lives only on the reverberating side

With the 400 nm film, dressing ONE interface at a time
(wl = 632.8 nm, normal incidence):

| case | engine dR |
|---|---|
| exit-only (σ_exit = 12) | **0.000e+00 exactly** (both polarizations, both angles) |
| front-only (σ_front = 8) | **6.284e-08** |
| both | 5.257e-08 (partial interference of the two σ²-effects) |

A semi-infinite exit has no reverberation — the fork cannot leak back into
R from the exit side. The front interface's transmission factors, however,
multiply every path that re-enters the ambient after bouncing inside the
film. This asymmetry is the mechanism's fingerprint.

### Probe D — magnitude: a two-interface Fabry–Pérot model

A numpy FP replication of the front-only fixture, differing ONLY in the
transmission dressing (ours `ga` vs upstream `f` on t12/t21), gives

- front-only, d = 400 nm: |R(ga-t) − R(f-t)| = **1.346e-07**
- both, d = 400 nm: **1.279e-07**
- exit-only: **0.000e+00**

Same order as the engines (the ~2× gap is the admittance-normalization
convention of the model — `y = 1/n` vs the engines' `y = n·cos` field/flow
mix — not the mechanism). The model also reproduces the σ² law
((f − ga) ∝ σ²) and the thickness behavior below.

### Probe E — the knockout: reverberation kill

Scaling the film thickness (the reverberation amplitude ∝ e^{−2·Im β},
so the feedback channel dies exponentially):

| film thickness | engine residual (front-only, σ = 8 nm) |
|---|---|
| 400 nm | 6.284e-08 |
| 800 nm | **5.153e-13** (÷1.2e5 — matches the FP model's 7.9e-13 prediction) |
| 1600 nm | **0.000e+00 exactly** |

At the test module's wl = 550 nm the same collapse measures 2.55e-9 →
1.63e-14 → 0.0 (the absolute value at a fixed thickness is phase-dependent
— 2·\|r₁f\|\|δ₂\|cos Δphase — the *exponential envelope* is the invariant,
and that is what is pinned).

## 4. The mechanism, stated precisely

The F1 fork: our transmission block applies the energy-conserving Gaussian
transfer factor `ga = exp(−(Δkz)²σ²/2)`; upstream applies the reflection
factor `f = exp(−2·kz1·kz2·σ²)` to transmission as well (the documented
energy bug — R + T < 1 at a single interface). Both are proven bit-correct
*within* their own formalisms (crate-level parity 1e-15; bare-interface
1.7e-16). In a **reverberating** structure the two disagree in R, because
the light transmitted through a dressed interface bounces inside the film
and comes back:

```
R = | r₁·f  +  t21·t12·r₂·φ² · (1 + r-loop + …) |²
      └─main─┘  └────── the reverberating tail ──────┘
```

The tail's *amplitude* is t21·t12·r₂·φ² ~ 1.4e-3 (its power ~2e-6 — the
review's "~1e-12 in power" figure was the full round-trip *including both
r-factors squared*, i.e. the wrong channel for this comparison). The ga-vs-f
difference on the tail's t-factors is (f − ga) ~ σ²·O(k²) ~ 5% at σ = 8 nm,
and the cross-term with the main reflected beam,

    2·Re( r₁·f · conj(δ_tail) ),   δ_tail ∝ (f − ga) ∝ σ²,

lands at ~1e-8 — exactly the observed residual and its exact σ² law. The
"four orders short" dismissal had squared the fork; it enters linearly.

The front/exit asymmetry follows: the front interface's transmission
factors gate everything that re-enters the ambient; the exit interface's
transmission factors lead into a semi-infinite medium and never return.

## 5. Resolution & aftermath

1. **F3 is not an upstream defect.** The wheel (0.7.0 and 0.7.7) computes
   its documented model correctly; the residual is the *expected*
   divergence given the documented F1 fork (our fix) plus reverberation.
2. **The upstream ask is downgraded.** Exposing `nevot_croce_factors` in
   `navette._smatrix` is a nice-to-have for downstream validators, not a
   defect report. F1 remains the (model-dispute) record it always was.
3. **Regression pin added** (`tests/test_roughness.py::_f3_thickness_pin`):
   the residual must collapse with film thickness — 400 nm ≤ 1e-7,
   800 nm ≤ 1e-11, 1600 nm == 0.0 (measured 2.55e-9 / 1.63e-14 / 0.0 at
   wl = 550 nm). Together with the transmission fork-guard (T diverges by
   design) this makes any future fork movement loud in both directions:
   if a wheel starts matching us at 400 nm, the fork closed; if our
   reverberation or ga changes, the pin moves.
4. **Wheel-upgrade tripwire recorded:** 0.7.7 re-audit clean (guard
   re-applied, crate bumped in `Cargo.toml`/`Cargo.lock`, parity green,
   the F3 residual value unchanged).

## 6. Files touched (commit `03a0c21`)

- `NAVETTE_UPSTREAM_REVIEW.md` — F3 section rewritten (OPEN → RESOLVED,
  evidence chain, corrected ruling-out), changelog entry.
- `tests/test_roughness.py` — F3 arm comment updated + `_f3_thickness_pin`
  regression gate.
- `IMPLEMENTATION_PLAN.md` — §8.6 "OPEN (mechanism unidentified)" →
  RESOLVED with the pointer.
- `Cargo.toml`, `Cargo.lock` — the 0.7.7 crate bump (tripwire).

## 7. Method note (for whoever reads the raw numbers later)

The absolute magnitude of the residual at a fixed thickness is
phase-dependent (the cross-term's cos factor): 6.3e-8 at wl = 632.8 nm vs
2.6e-9 at wl = 550 nm on the same fixture — both are the same mechanism;
only the exponential *envelope* (thickness scaling) and the σ² law are
phase-invariant, and only those are pinned. Hand-deriving the cross-term
magnitude without the phase is what led the original review to rule
reverberation out; the FP model (probe D) is the cheap way to keep the
order-of-magnitude honest.
