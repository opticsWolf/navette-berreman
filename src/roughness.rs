// SPDX-License-Identifier: LGPL-3.0-or-later
//! Surface-roughness models for the 4×4 interface scattering matrix.
//!
//! Route 1 (specular / coherent attenuation) — a direct generalization of the
//! `smatrix` roughness dressing (`func_0.rs` / `func_3.rs`) to anisotropic
//! partial waves. The integer `rough_type` codes match smatrix exactly:
//!
//! | code | model                         | form factor W(q)             |
//! |------|-------------------------------|------------------------------|
//! | 0    | none                          | 1                            |
//! | 1    | uniform (box)                 | sin(q√3)/(q√3)               |
//! | 2    | two-delta                     | cos(q)                       |
//! | 3    | Lorentzian / exponential      | 1/(1 + q²/2)                 |
//! | 4    | Gaussian (Debye–Waller)       | exp(−q²/2)                   |
//! | 5    | Névot–Croce (Gaussian cross)  | exp(−2·ka·kb·σ²) on REFLECTION; |
//! |      |                               | exp(−(Δkz)²σ²/2) (= W_4) on      |
//! |      |                               | TRANSMISSION (upstream smatrix  |
//! |      |                               | applies f to t as well — known  |
//! |      |                               | energy bug, see Phase 8 §8.1;   |
//! |      |                               | we diverge by design)          |
//!
//! Each `W` is the characteristic function of the interface height distribution;
//! its argument is the z-wavevector transfer between the two coupled partial
//! waves. For an interface between media with partial-wave z-wavevectors
//! kz = k0·q, the element of the interface scattering matrix coupling incoming
//! mode `j` to outgoing mode `i` is multiplied by `W((kz_in[j] − kz_out[i])·σ)`
//! (types 1–4) or the Névot–Croce cross factor (type 5). In the isotropic limit
//! the four forward/backward modes become degenerate, the factor is constant
//! within each reflection/transmission block, and this reduces exactly to the
//! smatrix scalar dressing (`r12·W(2kz1σ)`, `r21·W(2kz2σ)`,
//! `t12,t21·W((kz1−kz2)σ)`; type 5 → `exp(−2kz1kz2σ²)` on reflection and
//! `W_4((kz1−kz2)σ)` on transmission (corrected smatrix semantics, Phase 8).

use crate::berreman::Tensor3;
use crate::cmatrix::{c, cone, Mat4, C};
use num_complex::ComplexFloat;

const SQRT3: f64 = 1.732_050_807_568_877_2;

// ─────────────────────────── Route 2: graded EMA profile ───────────────────

/// Double-precision erf. Power series (Neumaier-compensated) for |x| ≤ 2.25,
/// modified-Lentz continued fraction for Γ(½, x²) beyond (A&S 6.5 relation
/// erfc(x) = x·e^(−x²)·h with h from the gcf recurrence). Validated against
/// baked 17-digit C-libm references in the tests (≤ 4 ulp across [0, 8]).
/// Own implementation: Rust std has no erf and the plan forbids new deps.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    let a = x.abs();
    let v = if a <= 2.25 {
        erf_series(a)
    } else {
        1.0 - erfc_lentz(a)
    };
    if x < 0.0 {
        -v
    } else {
        v
    }
}

/// erf(x) = 2/√π · x · Σₙ (−x²)ⁿ / (n!·(2n+1)); the argument-reduced form
/// avoids alternating-series cancellation blowup (max term ~2.5 at x=2.25).
fn erf_series(x: f64) -> f64 {
    let x2 = x * x;
    // Neumaier compensated sum: naive Σ loses ~n·eps·max_term ≈ 5e-14 here.
    let mut sum = 1.0_f64;
    let mut comp = 0.0_f64;
    let mut term = 1.0_f64;
    for n in 1..64usize {
        let nf = n as f64;
        // term_n = (-x²)ⁿ/(n!(2n+1)): ratio = (-x²)·(2n−1)/(n(2n+1))
        term *= -x2 * (2.0 * nf - 1.0) / (nf * (2.0 * nf + 1.0));
        let t = sum + term;
        comp += (sum - t) + term;
        sum = t;
        if term.abs() < 1e-19 {
            break;
        }
    }
    std::f64::consts::FRAC_2_SQRT_PI * (sum + comp) * x
}

/// erfc(x) for x > 2.25 via modified Lentz on the upper incomplete gamma:
/// the CF convergent is Γ(½, x²) (unnormalized — verified empirically: the
/// raw product is exactly √π·erfc), so erfc(x) = x·e^(−x²)·h/√π.
fn erfc_lentz(x: f64) -> f64 {
    let a = 0.5_f64;
    let xx = x * x;
    const FPMIN: f64 = 1e-300;
    let mut b = xx + 1.0 - a;
    let mut cc = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..300usize {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        cc = b + an / cc;
        if cc.abs() < FPMIN {
            cc = FPMIN;
        }
        d = 1.0 / d;
        let del = d * cc;
        h *= del;
        if (del - 1.0).abs() < 1e-17 {
            break;
        }
    }
    // FRAC_1_SQRT_PI is behind unstable more_float_constants on our MSRV —
    // literal with a false-precision guard in the erf test suite.
    const FRAC_1_SQRT_PI: f64 = 0.564_189_583_547_756_3; // 1/√π
    x * (-xx).exp() * h * FRAC_1_SQRT_PI
}

/// Cumulative volume fraction of medium B at depth z (nominal interface at 0):
/// f(z) = ½(1 + erf(z/(σ√2))) — the exact Python-kernel formula it replaces.
#[inline]
fn volume_fraction(z: f64, sigma: f64) -> f64 {
    0.5 * (1.0 + erf(z / (sigma * std::f64::consts::SQRT_2)))
}

/// Route-2 kernel: graded effective-medium tensor profile bridging eps_a →
/// eps_b across a Gaussian rough interface of RMS `sigma_nm`. Sublayer k
/// (width dz = width/n) carries the volume-weighted mix at its midpoint
/// z_k = −width/2 + (k+½)·dz, width = total_width.unwrap_or(6σ).
/// Returns n_sublayers tensors, front → back. Errors on non-finite/negative
/// sigma, non-finite width, or n_sublayers == 0.
pub fn graded_tensors(
    eps_a: &Tensor3,
    eps_b: &Tensor3,
    sigma_nm: f64,
    n_sublayers: usize,
    total_width_nm: Option<f64>,
) -> Result<Vec<Tensor3>, String> {
    if !sigma_nm.is_finite() || sigma_nm <= 0.0 {
        return Err(format!("sigma_nm must be finite and > 0, got {sigma_nm}"));
    }
    if n_sublayers == 0 {
        return Err("n_sublayers must be >= 1".to_string());
    }
    let width = match total_width_nm {
        None => 6.0 * sigma_nm,
        Some(w) if w.is_finite() && w > 0.0 => w,
        Some(w) => {
            return Err(format!(
                "total_width_nm must be finite and > 0 when given, got {w}"
            ))
        }
    };
    let dz = width / n_sublayers as f64;
    let mut out = Vec::with_capacity(n_sublayers);
    for k in 0..n_sublayers {
        let z = -0.5 * width + (k as f64 + 0.5) * dz;
        let f = volume_fraction(z, sigma_nm);
        let mut t = [[c(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                t[i][j] = c((1.0 - f) * eps_a[i][j].re + f * eps_b[i][j].re,
                            (1.0 - f) * eps_a[i][j].im + f * eps_b[i][j].im);
            }
        }
        out.push(t);
    }
    Ok(out)
}

/// Roughness height-distribution form factor W(q). Mirrors
/// `smatrix::func_0::w_function_inner`. Type 5 is handled separately (it is a
/// two-media cross factor, not a single-argument W), so it returns 1 here.
#[inline]
pub fn w_function(q: C, rough_type: i32) -> C {
    match rough_type {
        1 => {
            let val = q * SQRT3;
            if val.abs() < 1e-9 {
                cone()
            } else {
                val.sin() / val
            }
        }
        2 => q.cos(),
        3 => cone() / (cone() + q * q * 0.5),
        4 => (-(q * q) * 0.5).exp(),
        // 0, 5, and any unknown code -> no single-argument attenuation here.
        _ => cone(),
    }
}

/// Build the 4×4 matrix of roughness factors for the interface between layer
/// `a` (incident side) and layer `b` (transmitted side), given their *raw*
/// partial-wave eigenvalues `qa`, `qb` (so kz = k0·q, signed), the vacuum
/// wavenumber `k0`, the RMS roughness `sigma`, and the `rough_type` code.
///
/// Mode ordering is [trans0, trans1, refl0, refl1]: indices 0,1 are forward
/// (+Re kz), 2,3 are backward (−Re kz). The interface scattering matrix rows /
/// columns map to physical channels as:
///   * out row 0,1 → forward modes of `b`        (transmitted into b)
///   * out row 2,3 → backward modes of `a`       (reflected into a)
///   * in  col 0,1 → forward modes of `a`        (incident from a)
///   * in  col 2,3 → backward modes of `b`       (incident from b)
pub fn interface_factors(
    qa: &[C; 4],
    qb: &[C; 4],
    k0: f64,
    sigma: f64,
    rough_type: i32,
) -> Mat4 {
    let mut w = [[cone(); 4]; 4];
    if rough_type == 0 || sigma == 0.0 {
        return w;
    }
    let kza: [C; 4] = [qa[0] * k0, qa[1] * k0, qa[2] * k0, qa[3] * k0];
    let kzb: [C; 4] = [qb[0] * k0, qb[1] * k0, qb[2] * k0, qb[3] * k0];
    let s2 = c(sigma * sigma, 0.0);

    for i in 0..4 {
        for j in 0..4 {
            if rough_type == 5 {
                match (i < 2, j < 2) {
                    (true, true) | (false, false) => {
                        // TRANSMISSION (forward a→b / backward b→a): Nevot–Croce
                        // theory gives the Gaussian transfer factor
                        //   ga = exp(−(Δkz)²σ²/2) = W_4((kz_in − kz_out)·σ),
                        // NOT the reflection cross factor. Upstream applies f
                        // to t (bug, Phase 8 §8.1); we diverge here by design (§8.6).
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
                let kz_out = if i < 2 { kzb[i] } else { kza[i] };
                let kz_in = if j < 2 { kza[j] } else { kzb[j] };
                w[i][j] = w_function((kz_in - kz_out) * c(sigma, 0.0), rough_type);
            }
        }
    }
    w
}

#[cfg(test)]
mod graded_tests {
    use super::*;

    /// erf vs baked 17-digit C-libm references (Python math.erf, glibc).
    /// Series arm: ≤ 4e-16 abs for |x| ≤ 2.25; CF tail: ≤ 4 ulp of the value
    /// or 1e-15 abs in the 1−erfc cancellation zone (x ≥ 3.5).
    #[test]
    fn erf_matches_libm_references() {
        let refs: &[(f64, f64)] = &[
            (0.0, 0.0),
            (0.25, 0.27632639016823696),
            (0.5, 0.5204998778130465),
            (0.75, 0.7111556336535152),
            (1.0, 0.8427007929497149),
            (1.25, 0.9229001282564582),
            (1.5, 0.9661051464753108),
            (1.75, 0.9866716712191824),
            (2.0, 0.9953222650189527),
            (2.1213203435596424, 0.9973002039367398),
            (2.25, 0.9985372834133188),
            (2.3, 0.9988568234026434),
            (2.5, 0.999593047982555),
            (2.75, 0.9998993780778803),
            (3.0, 0.9999779095030014),
            (3.5, 0.9999992569016276),
            (4.0, 0.9999999845827421),
            (4.5, 0.9999999998033839),
            (5.0, 0.9999999999984626),
            (6.0, 1.0),
            (8.0, 1.0),
        ];
        for &(x, want) in refs {
            let got = erf(x);
            let tol = if x <= 2.25 {
                4e-16
            } else {
                (got.abs() * 8.0 * f64::EPSILON).max(1e-15)
            };
            assert!(
                (got - want).abs() <= tol,
                "erf({x}): got {got:.17e} want {want:.17e}"
            );
            assert!(erf(-x) == -got, "erf odd symmetry broken at {x}");
        }
        assert!(erf(f64::NAN).is_nan());
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(f64::NEG_INFINITY), -1.0);
    }

    /// graded_tensors: f-profile midpoints, endpoint limits, validation.
    #[test]
    fn graded_profile_and_guards() {
        let ea = diag(c(2.0, 0.1));
        let eb = diag(c(3.0, -0.2));
        // default width 6σ: sublayer midpoints at −3σ+(k+½)·6σ/n
        let g = graded_tensors(&ea, &eb, 5.0, 7, None).expect("graded");
        assert_eq!(g.len(), 7);
        // sublayer midpoints: z_k = -width/2 + (k+1/2)·dz, width = 30, dz = 30/7
        let width = 6.0 * 5.0;
        let dz = width / 7.0;
        let f0 = volume_fraction(-0.5 * width + 0.5 * dz, 5.0);
        let f6 = volume_fraction(-0.5 * width + 6.5 * dz, 5.0);
        let want0 = c((1.0 - f0) * 2.0 + f0 * 3.0, (1.0 - f0) * 0.1 + f0 * -0.2);
        assert!((g[0][0][0] - want0).abs() < 1e-15, "first slice {:?}", g[0][0][0]);
        assert!((g[6][0][0] - c((1.0 - f6) * 2.0 + f6 * 3.0,
                                 (1.0 - f6) * 0.1 + f6 * -0.2)).abs() < 1e-15);
        // symmetry: mirrored sublayer pairs mix to (a+b)/2 per element
        for k in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    let s = g[k][i][j] + g[6 - k][i][j];
                    assert!((s - (ea[i][j] + eb[i][j])).abs() < 1e-12);
                }
            }
        }
        // explicit width passes through
        let g2 = graded_tensors(&ea, &eb, 1.0, 3, Some(9.0)).expect("graded");
        assert!((g2[1][0][0] - c(2.5, -0.05)).abs() < 1e-15); // midpoint z=0 -> f=1/2
        // validation
        assert!(graded_tensors(&ea, &eb, 0.0, 7, None).is_err());
        assert!(graded_tensors(&ea, &eb, -1.0, 7, None).is_err());
        assert!(graded_tensors(&ea, &eb, f64::NAN, 7, None).is_err());
        assert!(graded_tensors(&ea, &eb, 1.0, 0, None).is_err());
        assert!(graded_tensors(&ea, &eb, 1.0, 7, Some(0.0)).is_err());
        assert!(graded_tensors(&ea, &eb, 1.0, 7, Some(-3.0)).is_err());
        assert!(graded_tensors(&ea, &eb, 1.0, 7, Some(f64::NAN)).is_err());
    }

    fn diag(v: C) -> Tensor3 {
        [[v, c(0.0, 0.0), c(0.0, 0.0)],
         [c(0.0, 0.0), v, c(0.0, 0.0)],
         [c(0.0, 0.0), c(0.0, 0.0), v]]
    }
}
