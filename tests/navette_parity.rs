//! Rust-level parity vs the upstream `navette` crate (crates.io, 0.7.0).
//! Python-level companion: tests/test_roughness.py (G5), tests/test_graded.py.
//!
//! Audit findings pinned here (Phase 8; verified against the 0.7.0 sources in
//! the pre-downloaded .crate, NOT assumed):
//!  * `w_function_inner` (codes 0-4) is formula-identical to ours — asserted.
//!  * `nevot_croce_factors` returns `(f, ga)` with
//!    `ga = exp(+(Δkz)²σ²/2)` — SIGN-FLIPPED vs Névot–Croce theory and vs
//!    upstream's own types-1..4 arm (`W_4`, decaying). Upstream applies
//!    `(r·f, t·ga)`, so their code-5 transmission is unphysical (their own
//!    docs: R+T up to 49.4). We assert our transmission block against W_4
//!    (theory), and assert upstream ga differs the documented way — this
//!    test FAILS LOUDLY if a future upstream release fixes the sign, which
//!    is the intended re-audit tripwire (then revisit §8.6 fork-guard).

use _berreman::cmatrix::{c, cone, C};
use _berreman::roughness::{interface_factors, w_function};
use navette::smatrix::optics_core::{nevot_croce_factors, w_function_inner};

fn close(a: C, b: C, tol: f64) -> bool {
    (a - b).norm() <= tol
}

/// §8.12 claim, now executable: our W(q) == upstream `w_function_inner` for
/// codes 0-4, over real, imaginary, complex, tiny (gate-straddling) and
/// large q. The type-1 `|·|<1e-9` gate spelling differs (`abs` vs `norm`)
/// but both are the modulus — straddling values prove it.
#[test]
fn w_parity_with_upstream() {
    let qs = [
        c(0.0, 0.0),
        c(1e-12, 0.0),
        c(5e-10, 0.0), // straddles 1e-9/sqrt(3) type-1 gate
        c(2e-9, 0.0),
        c(0.0, 0.3),
        c(0.46, -0.21),
        c(3.7, 1.2),
        c(20.0, -15.0),
    ];
    for t in 0..5 {
        // Code 1 is constant-limited: upstream hardcodes SQRT3 = 1.73205080757
        // (11 digits; optics_core.rs:18), so its W_1 differs from full-precision
        // evaluation at ~1e-13. Ours uses the full literal (more precise); the
        // Python-level G5 gate (<=1e-9) is unaffected — code 1 dominates its
        // 2.2e-14 worst row for exactly this reason.
        for q in qs {
            let a = w_function(q, t);
            let b = w_function_inner(q, t);
            // Truncation error propagates as |q|·|W'|·δs (δs=1.23e-12);
            // the bound below tracks it with ~5x margin over the grid.
            let tol = if t == 1 {
                2e-12 * (1.0 + q.norm() * (1.0 + b.norm()))
            } else {
                1e-15
            };
            assert!(
                close(a, b, tol),
                "W mismatch code {t} q={q:?}: ours={a:?} upstream={b:?}"
            );
        }
    }
}

/// §8.3 pin on an isotropic lossy fixture pair (kz1 = entry-side,
///
/// kz2 = film-side, k0 folded in): every reflection entry of our type-5
/// block equals upstream `f`, every transmission entry equals the theoretical
/// Gaussian transfer factor W_4 — while upstream `ga` is the sign-flipped
/// variant (asserted to differ, with the expected magnitude).
#[test]
fn type5_blocks_vs_theory_and_upstream() {
    let k0 = 2.0 * std::f64::consts::PI / 550.0;
    let k1 = c(k0 * 1.0, 0.0); // entry (air, normal incidence)
    let k2 = c(k0 * 2.0, k0 * 1.5); // lossy film n = 2+1.5j
    let sigma = 8.0;
    let q1 = [k1 / k0, k1 / k0, -k1 / k0, -k1 / k0];
    let q2 = [k2 / k0, k2 / k0, -k2 / k0, -k2 / k0];
    let w = interface_factors(&q1, &q2, k0, sigma, 5);

    let (f_up, ga_up) = nevot_croce_factors(k1, k2, sigma);
    let ga_theory = w_function((k1 - k2) * c(sigma, 0.0), 4);
    // Sharp pin of the sign flip: upstream ga = exp(+x), theory = exp(−x),
    // so ga_up * ga_theory == 1 to fp noise. If a future upstream release
    // fixes the sign, ga_up == ga_theory and this product becomes ga² ≠ 1 —
    // the test fails loudly, which is the intended re-audit tripwire.
    assert!(
        close(ga_up * ga_theory, cone(), 1e-14),
        "upstream ga relation changed; re-audit §8.6 (sign bug fixed?)"
    );

    for i in 0..4 {
        for j in 0..4 {
            match (i < 2, j < 2) {
                (true, true) | (false, false) => assert!(
                    close(w[i][j], ga_theory, 1e-15),
                    "T-block [{i}][{j}] != W_4 transfer: {:?}",
                    w[i][j]
                ),
                _ => assert!(
                    close(w[i][j], f_up, 1e-15),
                    "R-block [{i}][{j}] != upstream f: {:?}",
                    w[i][j]
                ),
            }
        }
    }
}
