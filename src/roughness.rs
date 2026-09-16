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

use crate::cmatrix::{c, cone, Mat4, C};
use num_complex::ComplexFloat;

const SQRT3: f64 = 1.732_050_807_568_877_2;

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
