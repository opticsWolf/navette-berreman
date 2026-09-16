// SPDX-License-Identifier: LGPL-3.0-or-later
//! Complex 4x4 matrix exponential for the EM transfer path.
//! Method: scale A -> A/2^s so ||.||_1 <= 0.5, direct Taylor sum to order 16,
//! then square s times. Scaling-and-squaring + Taylor (not Padé) so every
//! coefficient is trivially auditable; see §3.5 for the upgrade path.
//!
//! Remainder for the record: with ||As|| ≤ 0.5, truncation after As^16/16! is
//! bounded by ||As||^17/17! · e^||As|| ≈ 0.5^17/17!·1.65 ≈ 2e-23. Squaring
//! amplifies relative error negligibly at our sizes. G9a pins empirically.

use crate::cmatrix::{mat4_add, mat4_identity, mat4_mul, mat4_norm1, mat4_scale, Mat4, C};

/// Number of Taylor terms (T = I + As + ... + As^16/16!).
pub const TAYLOR_N: u32 = 16;
/// Target scaled 1-norm.
pub const SCALE_TARGET: f64 = 0.5;

/// exp(A). Accurate to ~1e-15 for ||A||_1 <= ~25 (G9a).
pub fn mat4_expm(a: &Mat4) -> Mat4 {
    // scale: halve until ||.||_1 <= 0.5 (cap guards denormal/overflow input)
    let mut s = 0u32;
    let mut scaled = *a;
    while mat4_norm1(&scaled) > SCALE_TARGET && s < 32 {
        scaled = mat4_scale(&scaled, C::new(0.5, 0.0));
        s += 1;
    }
    // Taylor: T = I + As + As^2/2! + ... (direct sum; factorials exact in
    // f64 through 16! — no Horner, keeps each term inspectable).
    let mut term = mat4_identity();
    let mut sum = mat4_identity();
    let mut fact = 1.0_f64;
    for k in 1..=TAYLOR_N {
        term = mat4_mul(&term, &scaled);
        fact *= k as f64;
        sum = mat4_add(&sum, &mat4_scale(&term, C::new(1.0 / fact, 0.0)));
    }
    // square back
    for _ in 0..s {
        sum = mat4_mul(&sum, &sum);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmatrix::{c, cone, czero, mat4_identity, mat4_zero};

    fn max_diff(a: &Mat4, b: &Mat4) -> f64 {
        let mut m: f64 = 0.0;
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
        n[0][1] = c(0.5, -0.2);
        n[1][2] = c(0.3, 0.4);
        n[2][3] = c(-0.1, 0.6);
        n[0][2] = c(0.2, 0.1);
        n[1][3] = c(0.4, -0.3);
        n[0][3] = c(0.1, 0.1);
        let n2 = mat4_mul(&n, &n);
        let n3 = mat4_mul(&n2, &n);
        let expect = mat4_add(
            &mat4_add(&mat4_identity(), &n),
            &mat4_add(
                &mat4_scale(&n2, c(0.5, 0.0)),
                &mat4_scale(&n3, c(1.0 / 6.0, 0.0)),
            ),
        );
        assert!(max_diff(&mat4_expm(&n), &expect) < 1e-14);
    }

    #[test]
    fn expm_scaled_berreman_sized() {
        // realistic magnitude: ||i·k0·d·Δ|| ~ 20 (1000 nm slab at 550 nm).
        // Hand-built Δ-like matrix with 1-norm ≈ 2.2, scaled by 10.
        let mut d = mat4_zero();
        d[0][1] = c(0.9, 0.0);
        d[1][0] = c(2.1, 0.1);
        d[1][2] = c(0.4, 0.0);
        d[2][3] = cone();
        d[3][1] = c(0.3, 0.0);
        d[3][2] = c(1.8, -0.2);
        let a = mat4_scale(&d, c(0.0, 10.0));
        // group property: exp(A) = exp(A/2)² to 1e-13
        let half = mat4_expm(&mat4_scale(&a, c(0.5, 0.0)));
        let sq = mat4_mul(&half, &half);
        assert!(max_diff(&mat4_expm(&a), &sq) < 1e-13);
    }
}
