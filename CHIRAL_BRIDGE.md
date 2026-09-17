# Chiral parameter conventions — DBF β ↔ Pasteur κ ↔ Condon ORD

**Status: normative verification guide** for the Condon ORD model and the
β↔κ bridge (implemented `navette/berreman.py::dbf_beta_to_kappa`,
`kappa_to_dbf_beta`, `condon_kappa`; gates G16a–e in
`tests/test_chiral_models.py`). Companion to plan §10.7 (why not DBF) and
§10.9 (as-built). Written so a reviewer can re-derive every constant from
the cited sources without trusting this document — or our memory.

---

## 1. Where the tension comes from (scope answer)

- **Not missing functionality:** the chiral engine exists
  (`pasteur_tensors` → `chiral_layer`), pinned to 1e-14 (Task-0, G15a–e).
- **Not a wrong implementation:** our Pasteur mapping is the most tightly
  pinned physics in the repo.
- **It IS an alternative parameterization — in the source literature.**
  Metamaterial/homogenization papers report **β** (DBF, length-dimensioned);
  polarimetry/chemistry tabulates **ORD [α](λ)** in Condon-family models.
  Our engine speaks **κ** (Pasteur, dimensionless, `n± = n±κ`). To first
  order these describe the *same physics* (Lindell–Sihvola reparameterization;
  Cho confirms off-resonance). The risk is a user plugging a paper's β into
  a constructor expecting κ → silently wrong physics. Hence: **adapters
  only, engine frozen**. No DBF solver (three disqualifiers, plan §10.7).

## 2. Frozen conventions (Task-0 record — the anchors everything pins to)

| item | frozen value | pinned by |
|---|---|---|
| time convention | `e^{+iωt}` (Berreman) | solver-wide |
| (rho, rhop) slots | `rho = −iκI`, `rhop = +iκI` | `pasteur_tensors` + Task-0 tests |
| eigen-indices | `n± = n ± κ`, `n+κ` on channel-0 = RCP | Task-0 (1e-14) |
| trans relations | `t_R = t0·e^{+iD}`, `t_L = t0·e^{−iD}`, `D = k₀κd` | G15b (3.2e-14) |
| reflection | `Jr[1,0]=Jr[0,1]=r0`, cross-circular 0 | G15b |
| rotation | `arg(t_R)−arg(t_L) = +k₀κd` | G15c (4.1e-14, exact gate) |
| lossy chiral | `Im q > 0 ⇒ decay`; `κ` real-only in `pasteur_tensors` | G15e |

Empirical fact (G15b, measured): **our κ-medium's wave impedance is
helicity-blind** — `|t±| = |t0|`, `|r±| = |r0|` with `t0/r0` the *achiral*
slab's. This is a known property of the Pasteur/bi-isotropic
parameterization (intrinsic impedance independent of chirality).

## 3. Literature used (all live-fetched this session, not memory)

1. **K. Cho, "Dispersion Relation in Chiral Media: Credibility of
   Drude–Born–Fedorov equations", arXiv:1501.01078** (fetched PDF, full
   text). Gives: DBF definitions (his eqs (2)–(3)) — `D = ε(E + β∇×E)`,
   `B = μ(H + β∇×H)`, e^{−iωt}; the DBF↔ChC parameter map (his eq (9)):
   `ξ = η = ωεβ/c` (frequency-carrying ⇒ not equivalent across
   frequencies — his "contradiction"); first-principles comparison.
2. *Condon model*, Wikipedia article (raw wikitext with LaTeX intact).
   Constitutive form (Lindell bi-isotropic): `D = εE − iκ√(εμ)H`,
   `B = μH + iκ√(εμ)E`; **single-oscillator rotatory power**
   `κ(ω) = ωR/(ω₀² − ω² − iωΓ)` (the raw text's `ω²−ω²` denominator is a
   mangling; the multi-oscillator form and the ω₀/Γ/R bullet list confirm
   `ω₀² − ω² − iωΓ`); multi-oscillator `κ(ω) = Σ_b ωR_ba/(ω₀,ba² − ω² −
   iωΓ_ba)`; passivity `Im[κ]² < Im[ε]·Im[μ]·c₀²`; notes the ω-numerator ⇒
   no static chirality. Cites: **Condon, Altar & Eyring 1937** (primary;
   Rev. Mod. Phys. 9, 432 lineage), **Lindell–Sihvola–Tretyakov–Viitanen
   1994** (Artech House), **Akyurtlu & Werner 2004** (IEEE TAP 52, 2267 —
   the FDTD-standard form).
3. **S. Kim & K. Kim, arXiv:1605.06406** (bi-isotropic surface waves):
   context confirmation; uses the κ-form (chirality index) with the
   Lekner 1996 pointer.
4. **Freymond & Picard, arXiv:1204.5350** (previously read): DBF
   time-domain ill-posedness (§10.7 disqualifier 1).
5. Not fetched / not needed: Lakhtakia *Beltrami Fields in Chiral Media*
   (book), Lindell 1994 book (quoted second-hand via the Wikipedia article
   — flagged in §6 caveats).

## 4. Derivation — DBF eigen-indices (from Cho's cited eqs, no memory)

DBF, e^{−iωt}, absolute ε, μ (relative ε_r, μ_r, n = √(ε_rμ_r)):
`D = ε(E + β∇×E)`, `B = μ(H + β∇×H)`. Circular eigenwave of helicity s
(ẑ×v_s = −is·v_s, so ∇×E = s·k·E for a forward wave):

1. Maxwell: `k×E = ωB` → `B = −is(k/ω)E`; `k×H = −ωD` → `D = is(k/ω)H`.
2. Constitutive → `D = ε(1 + sβk)E`, `B = μ(1 + sβk)H`.
3. `H = B/(μ(1+sβk)) = −is(k/ω)E/(μ(1+sβk))`; substitute into `D`:
   `is(k/ω)H = (k/ω)²E/(μ(1+sβk)) = ε(1+sβk)E`.
4. ⇒ **`k_s = k₀·n/(1 − s·x)`**, `x := β·k₀·n` (exact, all orders).
   The four roots: `±k₀n/(1∓x)` — **asymmetric about n**.

Consequences (all algebra, checkable by hand):

- `Δn_DBF = n/(1−x) − n/(1+x) = 2xn/(1−x²)` — symmetric in x, exact.
- mean index `n̄ = n/(1−x²)` (O(x²) above n).
- **Roundtrip is helicity-blind:** the backward root of helicity s is
  `−k₀n/(1+sx)` (denominator flips!), so forward+backward per helicity
  accumulate `k₀n/(1−sx) + k₀n/(1+sx) = 2k₀n̄` — helicity-blind, and the
  forward deviation from the mean is exactly `±n·x/(1−x²) = ±κ_sym`.
  ⇒ a DBF slab factorizes exactly as `A_DBF · diag(e^{+ik₀κ_sym d},
  e^{−ik₀κ_sym d})` with `A_DBF` helicity-blind.
- **Impedance** (for the same eigenwave): `Z_s = E/H =
  is·Z₀√(μ_r/ε_r)·(1−x²)` — ALSO helicity-blind, common O(x²) factor.
  ⇒ DBF and Pasteur slabs agree in **transmission phase structure
  exactly** (given κ_sym) and differ in the common interface/roundtrip
  factor only at O(x²). (This supersedes an earlier working note that
  claimed an O(x) impedance split — that note fumbled the k substitution;
  the derivation above is the corrected, checked one.)

## 5. The two κ choices (why κ_sym is the default)

| choice | κ | Δn vs DBF | absolute n vs DBF |
|---|---|---|---|
| `κ_sym` (default) | `n·x/(1−x²)` | **EXACT** | O(x²) each (mean) |
| `κ_plus` (single-branch) | `n/(1−x) − n` | O(x³) off | `n₊` **EXACT**, `n₋` O(x²) |

ORD/CD depend only on Δn ⇒ **κ_sym** is the physically right default;
`κ_plus` is available for branch-matched studies. Roundtrip inverse
(closed form): `a := κ/n`, `x = (√(1+4a²) − 1)/(2a)`, `β = x/(k₀n)`.

## 6. The Condon normalization trap (documented, not hidden)

The Wikipedia/Lindell form couples `D` with `−iκ√(εμ)H` (absolute ε,μ ⇒
coefficient `κn/c₀`) and, with e^{−iωt}, produces eigen-indices
`n± = √(εμ) ± κ` — our slots carry the OPPOSITE sign (`rho = −iκ`) under
our e^{+iωt} convention, and the `√(εμ)` factor sits inside their κ
normalization. Resolution (no hardcoded factors from memory):

- **Our κ is *defined* by `n± = n±κ`** — Task-0-pinned, solver-verified.
  Any literature normalization difference lands in **R**, which is a
  per-material fit parameter; the Condon model's predictive content is the
  dispersion *shape* (pole at ω₀, ω-numerator, damping), not R's absolute
  scale.
- **Sign discipline:** R > 0 below resonance (ω < ω₀, i.e. λ > λ₀) ⇒
  κ > 0 ⇒ `arg(t_R) − arg(t_L) = +k₀κd` (G15c sense). G16d pins this
  through the solver.
- The absolute/relative √(εμ) ambiguity of the wiki snippet is thereby
  made irrelevant *for our API* (R absorbs it) — stated here so a reviewer
  sees it was confronted, not skipped.

## 7. What each gate proves (and why it is not circular)

| gate | assertion | why non-circular |
|---|---|---|
| G16a | solver `arg(Jt00/Jt11) == k₀·Δn_DBF·d` at ≤1e-13; RHS computed **directly from β** via the §4 formula | ties formula→κ_sym→full-4×4 chain; no κ on the RHS |
| G16a2 | our κ_sym slab vs hand-built DBF Airy reference `A_DBF·diag(e^{±ik₀κ_sym d})`: \|ΔJ\| = O(x²), constant pinned empirically | independent slab construction (per-channel scalar Airy with DBF's n_s, η_s) |
| G16a3 | discriminator: with `κ_plus` the arg-diff is off Δn_DBF by `2k₀d·n·x²/(1−x²)` (measured, O(1e-2)) — proves the κ_sym choice is *doing* something | both κ choices through the same solver |
| G16b | β→κ→β roundtrip ≤1e-14 rel (closed-form inverse, incl. κ→0, κ<0) | pure algebra |
| G16c | weak limit κ ≈ βk₀n², rel err ≤ x²·1.1 | matches the literature linear coefficient |
| G16d | Condon: γ=0 ⇒ real dtype; κ·λ const as λ≫λ₀ (static-chirality-free, the ω-numerator property stated in the source); R>0 below resonance ⇒ κ>0 ⇒ solver rotation in G15c sense | source-stated limits |
| G16e | Γ>0 ⇒ complex κ, Im κ>0 ⇒ R-channel damps: arg-ratio == 2k₀Re(κ)d, \|ratio\| == e^{−2k₀Im(κ)d} through the **tensor path** (complex ρ, the documented route for dispersive/lossy chirality) | end-to-end engine check of the §10.9 usage recipe |

## 8. Open caveats / reviewer checkpoints

1. **DBF k± formula not found verbatim** in a fetched source (Cho's PDF
   extraction truncates before the dispersion section). The §4 derivation
   from Cho's *cited* equations (2)–(3) is the substitute — transparent
   4-line algebra. If a verbatim source surfaces (Jaggard–Mickelson–
   Papazoglu 1989; Lindell 1994 ch. 3), record it here.
2. **Lindell 1994 book not read directly** — the bi-isotropic form is
   taken from the Wikipedia article quoting it; flag for the reviewer.
3. **R-unit convention**: documented as fit-parameter (§5 of condon_kappa
   docstring); if a user needs absolute R units (e.g. to compare with
   Condon's 1937 tabulations), that normalization bridge is still open.
4. **Passivity inequality NOT gated** — `Im[κ]² < Im[ε]Im[μ]c₀²` belongs
   to the source's parameterization; mapping it into our normalization
   needs the same unit bookkeeping as caveat 3. Docs-only.
5. **O(x²) residue constants** in G16a2 are empirical pins — re-measure
   if the solver's fp path changes (they are floors, not physics).

## 9. Implementation lessons caught by the gates (same discipline as §10.8)

1. **Scalar-Airy reference had two sign/factor slips** — the exit
   transmission is `2η_e/(η_e+η_l)` (not `2η_l/...`) and the roundtrip
   uses the front reflection *seen from inside* `(η_a−η_l)/(η_a+η_l)`.
   Sanity gate: Airy-vs-engine on achiral slabs = **9.875e-16**.
2. **n2 must be threaded through every call** — the (n2=1.5) config
   silently solved with n2=1.0 while the reference used 1.5 (residue
   0.203 = the 0.8/1.0 flux scale). Symptom: residue >> predicted floor.
3. **Rotation = half the arg-ratio** (`t± = t0·e^{±iD}` ⇒ arg-ratio =
   2k₀κd; the polarization rotates by k₀κd). G16d first failed at ratio
   exactly 2.0 — caught, fixed, now an exact 1e-12 gate.
4. **Roundtrip inverse cancellation** — `x = (√(1+4a²)−1)/(2a)` loses ~13
   digits at small a (G16b saw 2.8e-13); rationalized form
   `x = 2a/(1+√(1+4a²))` → **3.488e-16**.
5. **Static-limit gate must carry the algebra's own correction** —
   κ·λ is constant only to O((λ₀/λ)²); the naive 1e-6 bound was 15× too
   tight. Gate now uses 2(λ₀/λ_min)² plus a 1e-13 check of the leading
   form.

## 10. Changelog

- **This session:** literature haul (§3), derivations (§4), κ_sym
  decision (§5), normalization trap documented (§6), helpers implemented
  (`dbf_beta_to_kappa`, `kappa_to_dbf_beta`, `condon_kappa`), gates
  G16a–e written and green with the measured values above; lessons §9
  recorded. §10.7's "actionable residue" (β↔κ helper) and §10.3's
  "Condon stretch goal" are hereby CLOSED; plan §10.9 records as-built.
