// SPDX-License-Identifier: LGPL-3.0-or-later
//! Phase 11 — POLARIZANCE v2: Brown differential Mueller decomposition.
//!
//! Material-level forward model (expansion-only: the Berreman engine is
//! untouched; these kernels consume tensors, they never drive the solver).
//! Two routes produce the same differential quantities (b3, d3, absorbance),
//! per unit length; length enters only in the exponentials:
//!
//! - **Route A** (live-compatible scalar path, validation oracle): live
//!   `dielectric_tensor.py::linear_optics_from_dielectric_tensor` +
//!   `get_ldlb_params_from_n_tensor`, replicated exactly: elementwise
//!   `n = sqrt(eps)` (live `get_refractive_index_tensor` semantics — exact
//!   only for diagonal eps in the x/y measurement basis), prime tensor
//!   `eps' = R(-45°)·eps·R(+45°)` (live `get_xy_rotation_matrix(3, -pi/4)` +
//!   the einsum composition), and
//!   `ld = -(Im n_yy - Im n_xx)·omega·length_over_c`,
//!   `lb = -(Re n_yy - Re n_xx)·omega·length_over_c` (primes likewise).
//!   Live's `absorbance = max(v1, v2)` stability hack is NOT inherited: both
//!   orientation sums are returned and the caller chooses (plan §11.4).
//! - **Route B** (the deliverable, general): the q-eigenvalues/modes of the
//!   Berreman matrix at kx = 0 → bulk Jones generator
//!   `k = W·diag(i k0 q_j)·W⁻¹` (psi = [Ex, Hy, Ey, -Hx]; W = [[Ex1,Ex2],[Ey1,Ey2]]
//!   from the forward pair = LayerWaves columns 0,1) → the infinitesimal form
//!   of our G11-anchored `mueller_from_jones`: with the same A basis,
//!   `H_full = A·(kron(k, I) + kron(I, conj(k)))·A⁻¹` (the identity
//!   kron(expm kL, conj expm kL) = expm((kron k, I) + (I, kron conj k))·L
//!   makes this the differential Mueller generator: 7 real characteristics —
//!   the isotropic-phase direction k = i·phi·I maps to zero, so H_full is
//!   real and lies in the span of live's `POLARIZANCE.diff_matrix()`
//!   structure — asserted by test).
//!
//! Shared kernels: `polarizance_decompose` (p = b + i·d, p_m = sqrt(p·p) NO
//! conjugation — live `decompose_polarizance`), `brown_params` (live formulas,
//! Brown 1999 DOI 10.1117/12.366361), `diff_mueller_matrix` (live
//! `POLARIZANCE.diff_matrix()` placement), `mueller_from_diff`
//! (= exp(-absorbance·L)·expm(H·L)).
//!
//! Literature: Brown, Proc. SPIE (1999), DOI 10.1117/12.366361; Salij,
//! Goldsmith, Tempelaar, arXiv:2208.14461v2 Supporting Information §S2–S3
//! (differential formalism dM/dz = H·M, M = exp(H·l), Brown B-params S5–S9,
//! second-order consistency S10–S11); Gil & Ossikovski (coherency side,
//! already G11-anchored). Plan §11.

use crate::berreman::{layer_waves_full, layer_waves_simple, Tensor3};
use crate::cmatrix::{c, czero, cone, Mat2, Mat4};
use crate::transfer::mueller_a_basis;

/// Real 4x4 (row-major), the Mueller-matrix carrier type.
pub type Mat4R = [[f64; 4]; 4];

// ───────────────────────── Route A (live-compatible) ─────────────────────────

/// Elementwise complex sqrt of a 3x3 tensor (live `get_refractive_index_tensor`).
fn eps_sqrt(t: &Tensor3) -> Tensor3 {
    let mut n = [[czero(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            n[i][j] = t[i][j].sqrt();
        }
    }
    n
}

/// Live `get_xy_rotation_matrix(3, angle)`: rotation in the x/y plane.
fn rot_xy3(angle: f64) -> [[f64; 3]; 3] {
    let (cs, sn) = (angle.cos(), angle.sin());
    [[cs, -sn, 0.0], [sn, cs, 0.0], [0.0, 0.0, 1.0]]
}

/// Route A, one wavelength. `omega` is the caller's spectrum value in the same
/// unit system the permittivity was built for (rad/s in ours: 2·pi·c0/lambda);
/// `length_over_c` is live's unit-scaling knob (pass L/c0 for a physical
/// thickness, c0 = 299792458e9 nm/s; pass 1.0 for per-unit-length differentials).
/// Returns (ld, ldp, lb, lbp, absorbance_v1, absorbance_v2).
pub fn linear_optics_scalar(eps: &Tensor3, omega: f64, length_over_c: f64) -> [f64; 6] {
    // live: dt_prime = R(-45°)·eps·R(+45°)
    // (einsum("ij,jkl->ikl", R, einsum("ijl,jk->ikl", eps, Rᵀ)))
    let r = rot_xy3(-std::f64::consts::FRAC_PI_4);
    let rt = [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ];
    let mut e2 = [[czero(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = czero();
            for k in 0..3 {
                for l in 0..3 {
                    // (R·eps·Rᵀ)[i][j] = Σ_kl R[i][k]·eps[k][l]·Rᵀ[l][j]
                    acc = acc + c(r[i][k], 0.0) * eps[k][l] * c(rt[l][j], 0.0);
                }
            }
            e2[i][j] = acc;
        }
    }
    let n = eps_sqrt(eps);
    let n_p = eps_sqrt(&e2);
    let w = omega * length_over_c;
    [
        -(n[1][1].im - n[0][0].im) * w,
        -(n_p[1][1].im - n_p[0][0].im) * w,
        -(n[1][1].re - n[0][0].re) * w,
        -(n_p[1][1].re - n_p[0][0].re) * w,
        (n_p[1][1].im + n_p[0][0].im) * w,
        (n[1][1].im + n[0][0].im) * w,
    ]
}

// ─────────────────────── shared scalar kernels ───────────────────────────────

/// (r_p, i_p, n_p) from the (b3, d3) differential vectors: p = b + i·d,
/// p_m = sqrt(p·p) (NO conjugation — live `decompose_polarizance` einsum),
/// r_p = Re(p_m), i_p = Im(p_m), n_p = sqrt(r_p² + i_p²) = |p_m|.
pub fn polarizance_decompose(b3: &[f64; 3], d3: &[f64; 3]) -> [f64; 3] {
    let mut pp = czero();
    for j in 0..3 {
        let pj = c(b3[j], d3[j]);
        pp = pp + pj * pj;
    }
    let pm = pp.sqrt();
    let (r, i) = (pm.re, pm.im);
    [r, i, (r * r + i * i).sqrt()]
}

/// Brown 1999 a0..a3 closed forms, live `brown_params` formulas exactly
/// (cross-paired: r_p with cosh(i_p·L), i_p with cos(r_p·L)). None when
/// n_p == 0 or non-finite (live yields NaN silently; the core refuses —
/// the pure-absorption limit is documented on the Python side).
pub fn brown_params(r_p: f64, i_p: f64, n_p: f64, length: f64) -> Option<[f64; 4]> {
    if n_p == 0.0 || !n_p.is_finite() {
        return None;
    }
    let n2 = n_p * n_p;
    let cl = (i_p * length).cosh();
    let cr = (r_p * length).cos();
    let a0 = (r_p / n_p) * (r_p / n_p) * cl + (i_p / n_p) * (i_p / n_p) * cr;
    let a1 = (cl - cr) / n2;
    let a2 = (r_p * (r_p * length).sin() + i_p * (i_p * length).sinh()) / n2;
    let a3 = (i_p * (r_p * length).sin() - r_p * (i_p * length).sinh()) / n2;
    Some([a0, a1, a2, a3])
}

/// Differential Mueller generator, live `POLARIZANCE.diff_matrix()` placement
/// exactly (b_matrix/d_matrix rows = linear, linear', circular axes; the
/// circular slots (b2, d2) carry CB/CD and are zero in Route A):
/// H[0j] = H[j0] = d_{j-1}; H[1,2] = -b2, H[2,1] = b2; H[1,3] = b1,
/// H[3,1] = -b1; H[2,3] = -b0, H[3,2] = b0; diagonal zero (traceless).
pub fn diff_mueller_matrix(b3: &[f64; 3], d3: &[f64; 3]) -> Mat4R {
    let (b0, b1, b2) = (b3[0], b3[1], b3[2]);
    let (d0, d1, d2) = (d3[0], d3[1], d3[2]);
    [
        [0.0, d0, d1, d2],
        [d0, 0.0, -b2, b1],
        [d1, b2, 0.0, -b0],
        [d2, -b1, b0, 0.0],
    ]
}

/// Bulk propagation Mueller matrix: M = exp(-absorbance·L)·expm(H·L), with
/// H = diff_mueller_matrix (traceless; the isotropic loss factor carries the
/// mean absorption — matching live's absorbance convention:
/// absorbance = (n_xx.im + n_yy.im)·k0·L for the diagonal case).
pub fn mueller_from_diff(b3: &[f64; 3], d3: &[f64; 3], absorbance: f64, length: f64) -> Mat4R {
    let h = diff_mueller_matrix(b3, d3);
    let mut arg: Mat4 = [[czero(); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            arg[i][j] = c(h[i][j] * length, 0.0);
        }
    }
    let m = crate::expm::mat4_expm(&arg);
    let f = (-absorbance * length).exp();
    let mut o = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            o[i][j] = m[i][j].re * f;
        }
    }
    o
}

/// Inverse read-off of (b3, d3) from the generator (live diff_matrix inverse):
/// d_j = (H[0,j+1] + H[j+1,0])/2 (symmetric slots),
/// b0 = (H[3,2] - H[2,3])/2, b1 = (H[1,3] - H[3,1])/2, b2 = (H[2,1] - H[1,2])/2.
pub fn diff_vectors_from_matrix(h: &Mat4R) -> ([f64; 3], [f64; 3]) {
    let d = [
        (h[0][1] + h[1][0]) / 2.0,
        (h[0][2] + h[2][0]) / 2.0,
        (h[0][3] + h[3][0]) / 2.0,
    ];
    let b = [
        (h[3][2] - h[2][3]) / 2.0,
        (h[1][3] - h[3][1]) / 2.0,
        (h[2][1] - h[1][2]) / 2.0,
    ];
    (b, d)
}

// ─────────────────────────── Route B (eigenpath) ─────────────────────────────

/// Route B result: bulk differential data at kx = 0 for one wavelength.
pub struct BulkDiff {
    pub b: [f64; 3],
    pub d: [f64; 3],
    /// Isotropic loss per unit length (≥ 0 for passive media):
    /// absorbance = -Re(tr k) = k0·(Im q0 + Im q1) for the diagonal case.
    pub absorbance: f64,
    /// Bulk Jones propagator J(L) = W·diag(exp(i k0 q_j L))·W⁻¹ at kx = 0
    /// (infinitely extended homogeneous medium — NO interface factors;
    /// carries the attenuation in its complex eigenvalues).
    pub jones: Mat2,
}

fn is_trivial_mo(rho: &Tensor3, rhop: &Tensor3, mu: &Tensor3) -> bool {
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                if rho[i][j].norm() > 1e-14 || rhop[i][j].norm() > 1e-14 {
                    return false;
                }
                if (mu[i][j] - cone()).norm() > 1e-12 {
                    return false;
                }
            } else if rho[i][j].norm() > 1e-14
                || rhop[i][j].norm() > 1e-14
                || (mu[i][j] - czero()).norm() > 1e-12
            {
                return false;
            }
        }
    }
    true
}

/// Route B: bulk differential extraction from a full tensor set at normal
/// incidence. None when the eigen-sort is not clean or the forward-mode
/// E-projection W is (near-)singular (degenerate polarization eigenmodes —
/// e.g. z-uniaxial at kx = 0; documented limitation, plan §11.2 Route B).
pub fn bulk_differential(
    eps: &Tensor3,
    rho: &Tensor3,
    rhop: &Tensor3,
    mu: &Tensor3,
    k0: f64,
    length: f64,
) -> Option<BulkDiff> {
    let waves = if is_trivial_mo(rho, rhop, mu) {
        layer_waves_simple(eps, 0.0)
    } else {
        layer_waves_full(eps, rho, rhop, mu, 0.0)
    };
    if !waves.clean {
        return None;
    }
    // Forward pair = columns 0,1 (transmitted modes; LayerWaves convention).
    let phase = [c(0.0, k0) * waves.q[0], c(0.0, k0) * waves.q[1]];
    // W = [[Ex1, Ex2], [Ey1, Ey2]] (psi = [Ex, Hy, Ey, -Hx]: rows 0 and 2).
    let w: Mat2 = [
        [waves.p[0][0], waves.p[0][1]],
        [waves.p[2][0], waves.p[2][1]],
    ];
    let det = w[0][0] * w[1][1] - w[0][1] * w[1][0];
    if det.norm() < 1e-10 {
        return None;
    }
    let winv: Mat2 = [
        [w[1][1] / det, -w[0][1] / det],
        [-w[1][0] / det, w[0][0] / det],
    ];
    // k = W·diag(i k0 q_j)·W⁻¹ (per unit length)
    let mut k: Mat2 = [[czero(); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut acc = czero();
            for m in 0..2 {
                acc = acc + w[i][m] * phase[m] * winv[m][j];
            }
            k[i][j] = acc;
        }
    }
    // J(L) = W·diag(exp(i k0 q_j L))·W⁻¹
    let mut jl: Mat2 = [[czero(); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut acc = czero();
            for m in 0..2 {
                acc = acc + w[i][m] * (phase[m] * c(length, 0.0)).exp() * winv[m][j];
            }
            jl[i][j] = acc;
        }
    }
    // H_full = A·(kron(k, I) + kron(I, conj(k)))·A⁻¹ — the infinitesimal form
    // of mueller_from_jones (same A basis). Real up to noise (iso-phase kernel).
    let h_full = a_map_kron_gen(&k);
    let scale: f64 = h_full
        .iter()
        .flat_map(|row| row.iter())
        .map(|v| v.norm())
        .fold(0.0, f64::max);
    let imax: f64 = h_full
        .iter()
        .flat_map(|row| row.iter())
        .map(|v| v.im.abs())
        .fold(0.0, f64::max);
    if imax > 1e-8 * scale.max(1.0) {
        return None;
    }
    let mut h = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            h[i][j] = h_full[i][j].re;
        }
    }
    let absorbance = -h[0][0]; // == -Re(tr k) (asserted in tests)
    let (b, d) = diff_vectors_from_matrix(&h);
    Some(BulkDiff { b, d, absorbance, jones: jl })
}

/// The infinitesimal Jones→Mueller map (same A basis as `mueller_from_jones`):
/// H_full = A·(kron(k, I) + kron(I, conj(k)))·A⁻¹.
fn a_map_kron_gen(k: &Mat2) -> Mat4 {
    let mut s: Mat4 = [[czero(); 4]; 4];
    for i in 0..2 {
        for j in 0..2 {
            for p in 0..2 {
                for q in 0..2 {
                    let a = if p == q { k[i][j] } else { czero() };
                    let b = if i == j { k[p][q].conj() } else { czero() };
                    s[i * 2 + p][j * 2 + q] = a + b;
                }
            }
        }
    }
    let (a, ainv) = mueller_a_basis();
    mat4c_mul(&mat4c_mul(&a, &s), &ainv)
}

/// Complex 4x4 multiply (local helper; cmatrix stays untouched).
fn mat4c_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut o = [[czero(); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut acc = czero();
            for kk in 0..4 {
                acc = acc + a[i][kk] * b[kk][j];
            }
            o[i][j] = acc;
        }
    }
    o
}

// ═══════════════════════════════ tests ═══════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmatrix::C;
    use crate::transfer::mueller_from_jones;

    /// expm of a (length-scaled) 2x2 complex matrix, closed form
    /// e^A = e^{tr/2}·[cosh(s)·I + sinh(s)/s·(A − tr/2·I)], s² = ((a−d)/2)² + bc
    /// (standard 2x2 formula; test-utility only, well-conditioned fixtures).
    fn mat2_expm_scaled(a: &Mat2, length: f64) -> Mat2 {
        let mut aa = *a;
        for row in &mut aa {
            for v in row {
                *v = *v * c(length, 0.0);
            }
        }
        let tr2 = (aa[0][0] + aa[1][1]) / 2.0;
        let mut ac = aa;
        ac[0][0] = ac[0][0] - tr2;
        ac[1][1] = ac[1][1] - tr2;
        let s2 = ac[0][0] * ac[0][0] + ac[0][1] * ac[1][0];
        let s = s2.sqrt();
        let (ch, sh): (C, C) = if s2.norm() < 1e-30 {
            (cone(), cone())
        } else {
            (s.cosh(), s.sinh() / s)
        };
        let f = (tr2 * cone()).exp();
        let mut o = [[czero(); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let ident = if i == j { cone() } else { czero() };
                o[i][j] = f * (ident * ch + ac[i][j] * sh);
            }
        }
        o
    }

    /// Differential generator == Richardson-extrapolated finite difference of
    /// mueller_from_jones: H = d/dε mueller_from_jones(I + ε·k)|₀. Sign errors
    /// show at O(1); Richardson removes the O(ε) truncation.
    #[test]
    fn generator_matches_jones_kernel_finite_difference() {
        let ks: [Mat2; 3] = [
            [[c(0.3, -0.2), c(0.7, 0.4)], [c(-0.2, 0.1), c(0.15, 0.6)]],
            [[c(-0.5, 0.2), c(0.0, -0.3)], [c(0.4, 0.0), c(0.25, -0.15)]],
            [[c(0.1, 0.9), c(0.2, 0.0)], [c(0.0, 0.3), c(-0.4, 0.5)]],
        ];
        let eps = 1e-5_f64;
        for (t, k) in ks.iter().enumerate() {
            let h = a_map_kron_gen(k);
            let fd = |e: f64, i: usize, j: usize| -> f64 {
                let mut ji = [[czero(); 2]; 2];
                for p in 0..2 {
                    for q in 0..2 {
                        ji[p][q] = if p == q { c(1.0, 0.0) } else { czero() }
                            + c(e, 0.0) * k[p][q];
                    }
                }
                let m = mueller_from_jones(&ji);
                (m[i][j] - if i == j { 1.0 } else { 0.0 }).re / e
            };
            for i in 0..4 {
                for j in 0..4 {
                    let f1 = fd(eps, i, j);
                    let f2 = fd(eps / 2.0, i, j);
                    // forward differences: Richardson kills the O(ε) term via
                    // 2·fd(ε/2) − fd(ε) = H + O(ε²) (the (4f₂−f₁)/3 form is for
                    // central differences)
                    let rich = 2.0 * f2 - f1;
                    assert!(
                        (h[i][j].re - rich).abs() < 1e-9,
                        "case {t} ({i},{j}): h={} rich={}",
                        h[i][j].re, rich
                    );
                    assert!(h[i][j].im.abs() < 1e-12);
                }
            }
        }
    }

    /// Round trip through the Jones algebra: (b, d, alpha) → H (real) →
    /// k = A⁻¹·H·A must be a valid Jones generator whose A-map regenerates H
    /// exactly, and expm(H·L) == mueller_from_jones(expm(k·L)) to 1e-12.
    /// (Recovery note: S = A⁻¹·H·A satisfies
    ///  S[2i+p][2j+q] = k[i][j]·δ_pq + δ_ij·conj(k[p][q]); the diagonal slots
    ///  S[2i][2i] = 2·Re(k[i][i]) fix only the real diagonal — the invisible
    ///  isotropic phase is fixed to zero here; the Mueller matrix is
    ///  phase-invariant, so expm comparisons are unaffected.)
    #[test]
    fn brown_differential_matches_jones_expm() {
        let fixtures: [([f64; 3], [f64; 3], f64); 3] = [
            ([0.4, -0.15, 0.0], [0.1, 0.25, 0.0], 0.3),
            ([0.9, 0.2, 0.5], [-0.3, 0.4, 0.7], -0.2),
            ([0.05, 0.05, 0.6], [0.5, -0.2, 0.1], 0.8),
        ];
        let length = 0.7_f64;
        for (t, (b3, d3, alpha)) in fixtures.iter().enumerate() {
            let h_target: Mat4R = {
                let mut hh = diff_mueller_matrix(b3, d3);
                for i in 0..4 {
                    hh[i][i] += alpha;
                }
                hh
            };
            let hc: Mat4 = {
                let mut h = [[czero(); 4]; 4];
                for i in 0..4 {
                    for j in 0..4 {
                        h[i][j] = c(h_target[i][j], 0.0);
                    }
                }
                h
            };
            let (a, ainv) = mueller_a_basis();
            let s = mat4c_mul(&mat4c_mul(&ainv, &hc), &a);
            // kron structure: S[2i+p][2j+q] = k[i][j]·δ_pq + δ_ij·conj(k[p][q]).
            // Diagonals MIX the two Jones entries:
            //   S[0][0] = 2·Re(k00), S[3][3] = 2·Re(k11),
            //   S[1][1] = k00 + conj(k11), S[2][2] = k11 + conj(k00).
            // The invisible direction is the isotropic phase Im(k00)+Im(k11);
            // fix it to zero (Mueller is phase-invariant, expm unaffected).
            let mut k: Mat2 = [[czero(); 2]; 2];
            k[0][0] = c(s[0][0].re / 2.0, s[1][1].im / 2.0);
            k[1][1] = c(s[3][3].re / 2.0, -s[1][1].im / 2.0);
            k[0][1] = s[1][3];
            k[1][0] = s[2][0];
            // structural consistency of the remaining slots
            assert!(s[3][3].im.abs() < 1e-10);
            assert!((s[2][2].re - (s[0][0].re + s[3][3].re) / 2.0).abs() < 1e-10);
            assert!((s[2][2].im + s[1][1].im).abs() < 1e-10);
            let s_ref = a_map_kron_gen(&k);
            for i in 0..4 {
                for j in 0..4 {
                    assert!(
                        (s_ref[i][j] - hc[i][j]).norm() < 1e-10,
                        "case {t} ({i},{j}): A-map not closed"
                    );
                }
            }
            let mut arg: Mat4 = [[czero(); 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    arg[i][j] = c(h_target[i][j] * length, 0.0);
                }
            }
            let m_expm = crate::expm::mat4_expm(&arg);
            let jl = mat2_expm_scaled(&k, length);
            let m_direct = mueller_from_jones(&jl);
            for i in 0..4 {
                for j in 0..4 {
                    assert!(
                        (m_expm[i][j] - m_direct[i][j]).norm() < 1e-12,
                        "case {t} ({i},{j}): expm {} vs jones {}",
                        m_expm[i][j], m_direct[i][j]
                    );
                }
            }
        }
    }

    /// Isotropic reduction: (b, d) = 0 for lossless n ⇒ b/d vanish,
    /// absorbance = 0, J(L) = e^{i k0 n L}·I.
    #[test]
    fn isotropic_reduction() {
        let eps = crate::berreman::diag_eps(c(2.25, 0.0), c(2.25, 0.0), c(2.25, 0.0));
        let id = crate::berreman::identity_tensor();
        let zt = crate::berreman::zero_tensor();
        let k0 = std::f64::consts::TAU / 600.0;
        let bd = bulk_differential(&eps, &zt, &zt, &id, k0, 0.5).expect("isotropic must extract");
        for v in bd.b.iter().chain(bd.d.iter()) {
            assert!(v.abs() < 1e-14, "b/d must vanish, got {v}");
        }
        assert!(bd.absorbance.abs() < 1e-12);
        let n = 1.5_f64;
        let ph = c(0.0, k0 * n * 0.5).exp();
        assert!((bd.jones[0][0] - ph).norm() < 1e-12);
        assert!(bd.jones[0][1].norm() < 1e-12 && bd.jones[1][0].norm() < 1e-12);
    }

    /// Route A transcription check, hand-evaluable diagonal fixture:
    /// eps = diag(2.25, 2.3104, 2.5) → n_xx = 1.5, n_yy = 1.52 exactly.
    /// ld = ldp = abs1 = abs2 = 0 (lossless); lb = -(1.52-1.5)·omega·loc.
    /// The 45°-rotated tensor of a diagonal eps is diag((exx+eyy)/2, same, ezz)
    /// EXACTLY (the rotation isotropizes the xy plane) → ldp == 0.
    #[test]
    fn route_a_diagonal_transcription() {
        let eps = [
            [c(2.25, 0.0), czero(), czero()],
            [czero(), c(2.3104, 0.0), czero()],
            [czero(), czero(), c(2.5, 0.0)],
        ];
        let omega = std::f64::consts::TAU * 299792458e9 / 600.0; // rad/s at 600 nm
        let loc = 0.25;
        let [ld, ldp, lb, lbp, abs1, abs2] = linear_optics_scalar(&eps, omega, loc);
        assert!(ld.abs() < 1e-18);
        assert!(ldp.abs() < 1e-15, "ldp must be 0 for diagonal eps: {ldp}");
        assert!((lb - -(1.52 - 1.5) * omega * loc).abs() < 1e-18);
        assert!(lbp.abs() < 1e-15);
        assert!(abs1.abs() < 1e-15);
        assert!(abs2.abs() < 1e-15);
    }

    /// Route A lossy transcription (hand-evaluable): choose eps_yy so that
    /// n_yy = 1.5 + 0.1i exactly (eps_yy = (1.5+0.1i)²). Then
    /// ld = -(0.1)·omega·loc, abs2 = (0.2)·omega·loc, lb = 0.
    #[test]
    fn route_a_lossy_transcription() {
        let nyy = c(1.5, 0.1);
        let eps = [
            [c(2.25, 0.0), czero(), czero()],
            [czero(), nyy * nyy, czero()],
            [czero(), czero(), c(2.5, 0.0)],
        ];
        let omega = std::f64::consts::TAU * 299792458e9 / 600.0;
        let loc = 0.5;
        let [ld, ldp, lb, lbp, abs1, abs2] = linear_optics_scalar(&eps, omega, loc);
        assert!((ld - -0.1 * omega * loc).abs() < 1e-15);
        // unprimed orientation: abs2 = (n_yy.im + n_xx.im)·w = 0.1·w
        assert!((abs2 - 0.1 * omega * loc).abs() < 1e-15, "abs2={abs2}");
        // primed orientation: n'_xx = n'_yy = sqrt((eps_xx + eps_yy)/2)
        // (independent arithmetic, not the kernel; compare on the O(1) ratio —
        // the raw product is ~1.5e14, so absolute tolerances are meaningless)
        let navg = ((c(2.25, 0.0) + nyy * nyy) / c(2.0, 0.0)).sqrt();
        assert!(
            (abs1 / (omega * loc) - 2.0 * navg.im).abs() < 1e-12,
            "abs1={abs1}"
        );
        assert!(lb.abs() < 1e-15);
        assert!(ldp.abs() < 1e-15);
        assert!(lbp.abs() < 1e-15);
    }

    /// Brown pure-case checks per the LIVE formulas (cross-paired):
    /// pure retardance (i_p = 0, n_p = r_p): a0 = 1, a1 = (1−cos(RL))/R²,
    /// a2 = sin(RL)/R, a3 = 0.
    /// pure dichroism (r_p = 0, i_p = n_p): a0 = 1, a1 = (cosh(IL)−1)/I²,
    /// a2 = sinh(IL)/I, a3 = 0.
    /// Small-L Taylor structure (SI S10–S11 claims): a0 = 1 + O(L⁴)
    /// (the L² term cancels exactly in the cross-pairing), a1 = L²/2 + O(L⁴),
    /// a2 = L + O(L³), a3 = −r_p·i_p·L³/6 + O(L⁵).
    #[test]
    fn brown_params_pure_cases_and_taylor() {
        let (r, l) = (0.4_f64, 1.3_f64);
        let [a0, a1, a2, a3] = brown_params(r, 0.0, r, l).unwrap();
        assert!((a0 - 1.0).abs() < 1e-15);
        assert!((a1 - (1.0 - (r * l).cos()) / (r * r)).abs() < 1e-15);
        assert!((a2 - (r * l).sin() / r).abs() < 1e-15);
        assert!(a3.abs() < 1e-15);
        let i = 0.25_f64;
        let [a0, a1, a2, a3] = brown_params(0.0, i, i, l).unwrap();
        assert!((a0 - 1.0).abs() < 1e-15);
        assert!((a1 - ((i * l).cosh() - 1.0) / (i * i)).abs() < 1e-15);
        assert!((a2 - (i * l).sinh() / i).abs() < 1e-15);
        assert!(a3.abs() < 1e-15);
        assert!(brown_params(0.0, 0.0, 0.0, 1.0).is_none());

        // Taylor structure (SI S10–S11): tiny L, cross-paired expansion.
        // (Fresh names — the earlier bindings r/l must not leak into np.)
        let (rr, ii) = (0.3_f64, 0.2_f64);
        let np = (rr * rr + ii * ii).sqrt();
        let lt = 1e-3_f64;
        let [a0, a1, a2, a3] = brown_params(rr, ii, np, lt).unwrap();
        assert!((a0 - 1.0).abs() < 1e-12, "a0 = 1 + O(L^4): {a0}");
        assert!((a1 - lt * lt / 2.0).abs() < 1e-12, "a1 = L²/2 + O(L⁴): {a1}");
        // a2 carries a documented O(L³) term: a2 = L − L³·(R²−I²)/6 + O(L⁵)
        let a2_expect = lt - lt * lt * lt * (rr * rr - ii * ii) / 6.0;
        assert!((a2 - a2_expect).abs() < 1e-15, "a2 = L − L³(R²−I²)/6: {a2}");
        let a3_expect = -rr * ii * lt * lt * lt / 6.0;
        assert!(
            (a3 - a3_expect).abs() < 1e-12,
            "a3 = -RI·L³/6 + O(L⁵): {a3} vs {a3_expect}"
        );
    }
}
