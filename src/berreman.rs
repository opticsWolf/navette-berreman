//! Berreman 4x4 layer optics: matrix construction, partial waves, and the
//! eigenmode sort. This is a faithful Rust port of `pyllama.Layer` /
//! `HalfSpace` plus `full_berreman.calc_berreman_matrix`.
//!
//! Conventions follow pyllama exactly:
//!   * the field 4-vector is  ψ = [E_x, H_y, E_y, -H_x],
//!   * eigenvectors are the columns of P,
//!   * Q = diag(exp(i k0 q_k d)),
//!   * partial waves are split into transmitted / reflected by the sign of the
//!     (noise-rounded) eigenvalue, then ordered within each pair by the
//!     Poynting (or, if degenerate, electric-field) p/s ratio.

use num_complex::ComplexFloat;
use crate::cmatrix::{c, cone, czero, eig4, Mat4, C};

/// 3x3 complex tensor, row-major. Built from the caller's permittivity etc.
pub type Tensor3 = [[C; 3]; 3];

#[inline]
fn iden3() -> Tensor3 {
    [
        [cone(), czero(), czero()],
        [czero(), cone(), czero()],
        [czero(), czero(), cone()],
    ]
}

#[inline]
fn zero3() -> Tensor3 {
    [[czero(); 3]; 3]
}

/// Reduced Berreman matrix Δ for a non-magnetic, non-optically-active layer
/// (pyllama `_build_D`, method = "simple").
pub fn berreman_simple(eps: &Tensor3, kx: f64) -> Mat4 {
    let kx = c(kx, 0.0);
    let e = eps;
    let e22 = e[2][2];
    let inv = e22.recip();
    let mut d = [[czero(); 4]; 4];

    // row 0
    d[0][0] = -kx * e[2][0] * inv;
    d[0][1] = cone() - kx * kx * inv;
    d[0][2] = -kx * e[2][1] * inv;
    d[0][3] = czero();
    // row 1
    d[1][0] = e[0][0] - e[0][2] * e[2][0] * inv;
    d[1][1] = -kx * e[0][2] * inv;
    d[1][2] = e[0][1] - e[0][2] * e[2][1] * inv;
    d[1][3] = czero();
    // row 2
    d[2][0] = czero();
    d[2][1] = czero();
    d[2][2] = czero();
    d[2][3] = cone();
    // row 3
    d[3][0] = e[1][0] - e[1][2] * e[2][0] * inv;
    d[3][1] = -kx * e[1][2] * inv;
    d[3][2] = -kx * kx + e[1][1] - e[1][2] * e[2][1] * inv;
    d[3][3] = czero();

    d
}

/// Full Berreman matrix with magneto-electric (ρ, ρ') and magnetic (μ) terms.
/// Direct port of `full_berreman.calc_berreman_matrix`. Reduces to the simple
/// form when μ = I and ρ = ρ' = 0.
pub fn berreman_full(
    eps: &Tensor3,
    rho: &Tensor3,
    rhop: &Tensor3,
    mu: &Tensor3,
    kx: f64,
) -> Mat4 {
    let kx = c(kx, 0.0);
    let e = eps;
    let r = rho;
    let rp = rhop;
    let m = mu;

    // NOTE on convention: the reference full_berreman.py writes
    //     d = 1/(eps22·mu22 − rho22·rhop22);  a_i = numerator / d
    // which evaluates to numerator·(eps22·mu22 − rho22·rhop22). That is a latent
    // typo: with it the matrix does NOT reduce to the reduced ("simple") form
    // when mu=I, rho=rhop=0, contradicting the file's own docstring and the
    // cited Mazur / Azzam–Bashara derivation. The intended (and physically
    // correct) operation is division by the determinant. We divide by `det`
    // directly, which reproduces the simple matrix exactly in the reduction
    // limit (verified to 1e-16).
    let det = e[2][2] * m[2][2] - r[2][2] * rp[2][2];
    let dinv = det.recip();

    let a1 = (rp[2][0] * r[2][2] - e[2][0] * m[2][2]) * dinv;
    let a2 = ((rp[2][1] - kx) * r[2][2] - e[2][1] * m[2][2]) * dinv;
    let a3 = (m[2][1] * r[2][2] - r[2][0] * m[2][2]) * dinv;
    let a4 = (m[2][1] * r[2][2] - (r[2][1] + kx) * m[2][2]) * dinv;
    let a5 = (rp[2][2] * e[2][0] - e[2][2] * rp[2][0]) * dinv;
    let a6 = (rp[2][2] * e[2][1] - (rp[2][1] - kx) * e[2][2]) * dinv;
    let a7 = (rp[2][2] * r[2][1] - e[2][2] * m[2][0]) * dinv;
    let a8 = ((r[2][1] + kx) * rp[2][2] - e[2][2] * m[2][1]) * dinv;

    let mut b = [[czero(); 4]; 4];
    let rpk = rp[1][2] + kx;
    b[0][0] = rp[1][0] + rpk * a1 + m[1][2] * a5;
    b[0][1] = m[1][1] + rpk * a4 + m[1][2] * a8;
    b[0][2] = rp[1][1] + rpk * a2 + m[1][2] * a6;
    b[0][3] = -(m[1][0] + rpk * a3 + m[1][2] * a7);

    b[1][0] = e[0][0] + e[0][2] * a1 + rp[0][2] * a5;
    b[1][1] = r[0][1] + e[0][2] * a4 + rp[0][2] * a8;
    b[1][2] = e[0][1] + e[0][2] * a2 + rp[0][2] * a6;
    b[1][3] = -(r[0][0] + e[0][2] * a3 + rp[0][2] * a7);

    b[2][0] = -(rp[0][0] + rp[0][2] * a1 + m[0][2] * a5);
    b[2][1] = -(m[0][1] + rp[0][2] * a4 + m[0][2] * a8);
    b[2][2] = -(rp[0][1] + rp[0][2] * a2 + m[0][2] * a6);
    b[2][3] = m[0][0] + rp[0][2] * a3 + m[0][2] * a7;

    let rmk = r[1][2] - kx;
    b[3][0] = e[1][0] + e[1][2] * a1 + rmk * a5;
    b[3][1] = r[1][1] + e[1][2] * a4 + rmk * a8;
    b[3][2] = e[1][1] + e[1][2] * a2 + rmk * a6;
    b[3][3] = -(r[1][0] + e[1][2] * a3 + rmk * a7);

    b
}

// ───────────────────────────── Wave ────────────────────────────────────

/// A partial wave's fields and Poynting vector, derived from one ψ column.
#[derive(Clone, Copy)]
pub struct Wave {
    pub ex: C,
    pub ey: C,
    pub ez: C,
    pub hx: C,
    pub hy: C,
    pub hz: C,
    pub sx: C,
    pub sy: C,
    pub sz: C,
}

impl Wave {
    /// Build from a ψ column [E_x, H_y, E_y, -H_x] and the layer permittivity.
    pub fn from_psi(col: &[C; 4], eps: &Tensor3, kx: f64) -> Wave {
        let kx = c(kx, 0.0);
        let ex = col[0];
        let hy = col[1];
        let ey = col[2];
        let hx = -col[3];
        let inv22 = eps[2][2].recip();
        let ez = -(eps[2][0] * inv22) * ex - (eps[2][1] * inv22) * ey - (kx * inv22) * hy;
        let hz = kx * ey;
        // S = E × H (no conjugation; matches pyllama)
        let sx = ey * hz - ez * hy;
        let sy = ez * hx - ex * hz;
        let sz = ex * hy - ey * hx;
        Wave {
            ex,
            ey,
            ez,
            hx,
            hy,
            hz,
            sx,
            sy,
            sz,
        }
    }

    /// Build from a psi column for a FULL-tensor layer (Phase 10).
    /// from_psi assumes the simple constitutive (Dz/Bz rows with no
    /// magnetoelectric terms); with (rho, rhop, mu) != (0, 0, I) the
    /// longitudinal components must be reconstructed from the eliminated
    /// 2x2 system — equivalently, from the feedback weights visible in
    /// berreman_full's b-matrix rows, e.g. b[1][0] = e00 + e02*a1 + rp02*a5
    /// proves Ez carries a1*Ex and Hz carries a5*Ex. Columns of psi are
    /// (Ex, Hy, Ey, -Hx), so with Hx = -col[3]:
    ///   Ez = a1*Ex + a2*Ey + a4*Hy - a3*Hx
    ///   Hz = a5*Ex + a6*Ey + a8*Hy - a7*Hx
    /// (a1..a8 are Azzam-Bashara's p.344 weights, same expressions as in
    /// berreman_full — duplicated, not shared, so the hot matrix builder
    /// keeps its exact op order; the reduction test below pins equality.)
    /// P10 FINDING: from_psi's ez/hz are WRONG for full-tensor layers
    /// (caught by the Pasteur oblique-circularity probe: Ev/Eu off by 3e-4).
    /// Tangential (Ex, Ey, Hx, Hy) come straight from psi and were always
    /// exact — Jones, fluxes, G8b continuity unaffected; only Ez/Hz (fields
    /// longitudinal output, absorption_density's |E|^2, Wave Sz) were wrong.
    pub fn from_psi_full(
        col: &[C; 4],
        eps: &Tensor3,
        rho: &Tensor3,
        rhop: &Tensor3,
        mu: &Tensor3,
        kx: f64,
    ) -> Wave {
        let kx = c(kx, 0.0);
        let e = eps;
        let r = rho;
        let rp = rhop;
        let m = mu;
        let ex = col[0];
        let hy = col[1];
        let ey = col[2];
        let hx = -col[3];
        let det = e[2][2] * m[2][2] - r[2][2] * rp[2][2];
        let dinv = det.recip();
        let a1 = (rp[2][0] * r[2][2] - e[2][0] * m[2][2]) * dinv;
        let a2 = ((rp[2][1] - kx) * r[2][2] - e[2][1] * m[2][2]) * dinv;
        let a3 = (m[2][1] * r[2][2] - r[2][0] * m[2][2]) * dinv;
        let a4 = (m[2][1] * r[2][2] - (r[2][1] + kx) * m[2][2]) * dinv;
        let a5 = (rp[2][2] * e[2][0] - e[2][2] * rp[2][0]) * dinv;
        let a6 = (rp[2][2] * e[2][1] - (rp[2][1] - kx) * e[2][2]) * dinv;
        let a7 = (rp[2][2] * r[2][1] - e[2][2] * m[2][0]) * dinv;
        let a8 = ((r[2][1] + kx) * rp[2][2] - e[2][2] * m[2][1]) * dinv;
        let ez = a1 * ex + a2 * ey + a4 * hy - a3 * hx;
        let hz = a5 * ex + a6 * ey + a8 * hy - a7 * hx;
        let sx = ey * hz - ez * hy;
        let sy = ez * hx - ex * hz;
        let sz = ex * hy - ey * hx;
        Wave { ex, ey, ez, hx, hy, hz, sx, sy, sz }
    }

    #[inline]
    fn cp(x: C, y: C) -> f64 {
        let deno = x.norm_sqr() + y.norm_sqr();
        if deno == 0.0 {
            0.0
        } else {
            x.norm_sqr() / deno
        }
    }

    #[inline]
    pub fn cp_poynting(&self) -> f64 {
        Wave::cp(self.sx, self.sy)
    }

    #[inline]
    pub fn cp_elec(&self) -> f64 {
        Wave::cp(self.ex, self.ey)
    }
}

// ─────────────────────────── partial waves ─────────────────────────────

/// Sorted eigenvectors (columns of P) and eigenvalues for one layer.
pub struct LayerWaves {
    pub p: Mat4,
    pub q: [C; 4],
    /// True when the eigenvalues could be split cleanly 2 transmitted / 2
    /// reflected (the normal case). False signals a fallback ordering was used.
    pub clean: bool,
}

const THR: f64 = 1e-7;

#[inline]
fn round10(x: f64) -> f64 {
    // mimic numpy's .round(decimals=10): kills tiny imaginary noise
    (x * 1e10).round() / 1e10
}

/// Sort the four partial waves exactly as `pyllama._sort_p_q` (wavevector
/// style). `vals`/`vecs` come from a general eig of the Berreman matrix.
/// Simple-path sort entry (bit-identical history — layer_waves_simple).
/// Full-tensor layers use sort_partial_waves_full (Phase 10) so the
/// cp_poynting/cp_elec heuristics see the true longitudinal fields.
pub fn sort_partial_waves(vals: &[C; 4], vecs: &Mat4, eps: &Tensor3, kx: f64) -> LayerWaves {
    let mut waves = [Wave::from_psi(&[czero(); 4], eps, kx); 4];
    for k in 0..4 {
        let col = [vecs[0][k], vecs[1][k], vecs[2][k], vecs[3][k]];
        waves[k] = Wave::from_psi(&col, eps, kx);
    }
    sort_with_waves(vals, vecs, &waves)
}

/// Full-tensor sort entry (Phase 10): Wave records via from_psi_full.
#[allow(clippy::too_many_arguments)]
pub fn sort_partial_waves_full(
    vals: &[C; 4],
    vecs: &Mat4,
    eps: &Tensor3,
    rho: &Tensor3,
    rhop: &Tensor3,
    mu: &Tensor3,
    kx: f64,
) -> LayerWaves {
    let mut waves = [Wave::from_psi(&[czero(); 4], eps, kx); 4];
    for k in 0..4 {
        let col = [vecs[0][k], vecs[1][k], vecs[2][k], vecs[3][k]];
        waves[k] = Wave::from_psi_full(&col, eps, rho, rhop, mu, kx);
    }
    sort_with_waves(vals, vecs, &waves)
}

/// Shared sort core: split/sort given prebuilt Wave records.
fn sort_with_waves(vals: &[C; 4], vecs: &Mat4, waves: &[Wave; 4]) -> LayerWaves {

    let mut id_trans: Vec<usize> = Vec::with_capacity(2);
    let mut id_refl: Vec<usize> = Vec::with_capacity(2);
    for k in 0..4 {
        let test = vals[k].re + round10(vals[k].im);
        if test > 0.0 {
            id_trans.push(k);
        } else {
            id_refl.push(k);
        }
    }

    if id_trans.len() != 2 || id_refl.len() != 2 {
        return fallback_sort(vals, vecs);
    }

    // Birefringence test on the transmitted pair.
    let cp0 = waves[id_trans[0]].cp_poynting();
    let cp1 = waves[id_trans[1]].cp_poynting();
    if (cp0 - cp1).abs() > THR {
        if cp1 < cp0 {
            id_trans.swap(0, 1);
        }
        let r0 = waves[id_refl[0]].cp_poynting();
        let r1 = waves[id_refl[1]].cp_poynting();
        if r1 < r0 {
            id_refl.swap(0, 1);
        }
    } else {
        let t0 = waves[id_trans[0]].cp_elec();
        let t1 = waves[id_trans[1]].cp_elec();
        if (t1 - t0) < THR {
            id_trans.swap(0, 1);
        }
        let r0 = waves[id_refl[0]].cp_elec();
        let r1 = waves[id_refl[1]].cp_elec();
        if (r1 - r0) < THR {
            id_refl.swap(0, 1);
        }
    }

    let order = [id_trans[1], id_trans[0], id_refl[1], id_refl[0]];
    assemble(vals, vecs, &order, true)
}

fn fallback_sort(vals: &[C; 4], vecs: &Mat4) -> LayerWaves {
    // Best-effort: order by descending (re + rounded im), split 2/2. Not
    // pyllama's _correct_p, but deterministic for the rare unsortable case.
    let mut idx: Vec<usize> = (0..4).collect();
    idx.sort_by(|&a, &b| {
        let va = vals[a].re + round10(vals[a].im);
        let vb = vals[b].re + round10(vals[b].im);
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });
    let order = [idx[0], idx[1], idx[2], idx[3]];
    let mut lw = assemble(vals, vecs, &order, false);
    lw.clean = false;
    lw
}

fn assemble(vals: &[C; 4], vecs: &Mat4, order: &[usize; 4], clean: bool) -> LayerWaves {
    let mut q = [czero(); 4];
    let mut p = [[czero(); 4]; 4];
    for (m, &o) in order.iter().enumerate() {
        q[m] = vals[o];
        for i in 0..4 {
            p[i][m] = vecs[i][o];
        }
    }
    LayerWaves { p, q, clean }
}

/// True when `eps` is (numerically) a scalar multiple of the identity, i.e. an
/// optically isotropic medium. Such a layer has doubly-degenerate Berreman
/// eigenvalues where the p/s split is ill-defined; we use the analytic
/// half-space eigenbasis for it instead (physically identical, well-conditioned).
fn is_isotropic(eps: &Tensor3) -> bool {
    let tol = 1e-12;
    let e00 = eps[0][0];
    let diag_equal =
        (eps[1][1] - e00).norm() < tol * (1.0 + e00.norm()) && (eps[2][2] - e00).norm() < tol * (1.0 + e00.norm());
    let mut off_zero = true;
    for i in 0..3 {
        for j in 0..3 {
            if i != j && eps[i][j].norm() > tol * (1.0 + e00.norm()) {
                off_zero = false;
            }
        }
    }
    diag_equal && off_zero
}

/// Full solve of one anisotropic layer: build Δ, eig, sort. Isotropic layers
/// are handled analytically (same eigenbasis as a half-space).
pub fn layer_waves_simple(eps: &Tensor3, kx: f64) -> LayerWaves {
    if is_isotropic(eps) {
        return halfspace_waves(eps[0][0], kx);
    }
    let d = berreman_simple(eps, kx);
    let (vals, vecs) = eig4(&d);
    sort_partial_waves(&vals, &vecs, eps, kx)
}

pub fn layer_waves_full(
    eps: &Tensor3,
    rho: &Tensor3,
    rhop: &Tensor3,
    mu: &Tensor3,
    kx: f64,
) -> LayerWaves {
    let d = berreman_full(eps, rho, rhop, mu, kx);
    let (vals, vecs) = eig4(&d);
    sort_partial_waves_full(&vals, &vecs, eps, rho, rhop, mu, kx)
}

/// Analytic partial waves for an isotropic semi-infinite medium
/// (`pyllama.HalfSpace`). `eps_xx` is the (scalar) permittivity.
pub fn halfspace_waves(eps_xx: C, kx: f64) -> LayerWaves {
    let kx = c(kx, 0.0);
    let n = eps_xx.sqrt();
    let sin_phi = kx / n;
    let cos_phi = (cone() - sin_phi * sin_phi).sqrt();
    let q = [n * cos_phi, n * cos_phi, -n * cos_phi, -n * cos_phi];
    // columns are the ψ vectors:
    //  [cos_phi, n, 0, 0], [0, 0, 1, n cos_phi],
    //  [cos_phi, -n, 0, 0], [0, 0, 1, -n cos_phi]
    let p = [
        [cos_phi, czero(), cos_phi, czero()],
        [n, czero(), -n, czero()],
        [czero(), cone(), czero(), cone()],
        [czero(), n * cos_phi, czero(), -n * cos_phi],
    ];
    LayerWaves { p, q, clean: true }
}

/// Propagation phase diagonal Q = diag(exp(i k0 q_k d)).
pub fn propagation_diag(q: &[C; 4], k0: f64, thickness: f64) -> [C; 4] {
    let mut out = [czero(); 4];
    let f = c(0.0, k0 * thickness);
    for k in 0..4 {
        out[k] = (f * q[k]).exp();
    }
    out
}

/// Convenience: build a real (lossless) diagonal permittivity tensor.
pub fn diag_eps(exx: C, eyy: C, ezz: C) -> Tensor3 {
    let mut t = zero3();
    t[0][0] = exx;
    t[1][1] = eyy;
    t[2][2] = ezz;
    t
}

pub fn identity_tensor() -> Tensor3 {
    iden3()
}

pub fn zero_tensor() -> Tensor3 {
    zero3()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmatrix::c;

    /// Task 0 (Phase 10): Pasteur-Tellegen mapping, RECORDED assignment
    /// rho = -i k I, rhop = +i k I (reciprocal pair, rho = -rhop^T).
    /// Derivation note (TIME_CONVENTION record): the e^{-iwt} SI Pasteur form
    /// D = eE + i x H suggests rho = +i k I, but the operational probe showed
    /// +i k I pairs q = n+k with F-col1 (Ey/Ex = +i, LCP channel) — i.e. the
    /// code's (rho, rhop) slots couple with opposite sign to that derivation
    /// (as if e^{+iwt}; achiral observables can't pin this, chiral ones do).
    /// The RECORDED mapping below restores the standard n_R = n+k on the
    /// frozen channel-0 = RCP basis (pyllama fresnel_to_fresnel_circ docstring:
    /// J_circ[0,0] = RCP->RCP). Same Delta slots as the MO path validated in
    /// test_full.py against live's matrix form — reviewer sign-off, §10.2(3).
    /// At Kx = 0: spectrum {+-(n+k), +-(n-k)} real; forward q = n+k mode has
    /// Ey/Ex = -i, forward q = n-k mode has Ey/Ex = +i. Any {+-(n+-ik)} here
    /// means the rho slot normalization is structurally wrong — stop.
    #[test]
    fn pasteur_task0_normal_spectrum_and_handedness() {
        let n = 1.5_f64;
        let e = c(n * n, 0.0);
        let eps = diag_eps(e, e, e);
        let mu = identity_tensor();
        for &kappa in &[0.01_f64, 0.1_f64] {
            let ik = c(0.0, kappa);
            let rho = diag_eps(-ik, -ik, -ik);
            let rhop = diag_eps(ik, ik, ik);
            let d = berreman_full(&eps, &rho, &rhop, &mu, 0.0);
            let (vals, vecs) = eig4(&d);
            let mut got: Vec<f64> = vals.iter().map(|v| v.re).collect();
            got.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut want = vec![-(n + kappa), -(n - kappa), n - kappa, n + kappa];
            want.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut worst_q = 0.0_f64;
            for (g, w) in got.iter().zip(want.iter()) {
                worst_q = worst_q.max((g - w).abs());
            }
            let worst_im: f64 =
                vals.iter().map(|v| v.im.abs()).fold(0.0, f64::max);
            // eig4 absolute accuracy scales with ||D|| ~ n^2 (obs 5.6e-15 at
            // n = 1.5); gate 1e-14 << 2k separation. Imag parts stay ~1e-51.
            assert!(
                worst_q < 1e-14 && worst_im < 1e-15,
                "kappa={kappa}: re-err={worst_q:e} im-max={worst_im:e} got={got:?}"
            );
            let pick = |target: f64| -> usize {
                let mut k0 = 0;
                let mut best = f64::MAX;
                for (k, v) in vals.iter().enumerate() {
                    let dd = (v.re - target).abs() + v.im.abs();
                    if dd < best {
                        best = dd;
                        k0 = k;
                    }
                }
                k0
            };
            let k_hi = pick(n + kappa);
            let ratio_hi = vecs[2][k_hi] / vecs[0][k_hi];
            let err_hi = (ratio_hi + c(0.0, 1.0)).norm();
            assert!(err_hi < 1e-12, "kappa={kappa}: hi Ey/Ex={ratio_hi:?}");
            let k_lo = pick(n - kappa);
            let ratio_lo = vecs[2][k_lo] / vecs[0][k_lo];
            let err_lo = (ratio_lo - c(0.0, 1.0)).norm();
            assert!(err_lo < 1e-12, "kappa={kappa}: lo Ey/Ex={ratio_lo:?}");
        }
    }

    /// Task 0 oblique arm (Kx = 0.3): no closed-form q, so structural gates —
    /// forward/backward pairing q <-> -q (reciprocity), exact spherical
    /// dispersion Kx^2+q^2 = (n+-k)^2, and transverse circularity E_v/E_u = -+i
    /// in (u = y-hat, v = k-hat x u), branch-identified by sorted Re(q).
    #[test]
    fn pasteur_task0_oblique_structure() {
        let n = 1.5_f64;
        let kx = 0.3_f64;
        let e = c(n * n, 0.0);
        let eps = diag_eps(e, e, e);
        let mu = identity_tensor();
        for &kappa in &[0.01_f64, 0.1_f64] {
            let ik = c(0.0, kappa);
            let rho = diag_eps(-ik, -ik, -ik);
            let rhop = diag_eps(ik, ik, ik);
            let d = berreman_full(&eps, &rho, &rhop, &mu, kx);
            let (vals, vecs) = eig4(&d);
            let mut idx: Vec<usize> = (0..4).collect();
            idx.sort_by(|&a, &b| {
                vals[a].re.partial_cmp(&vals[b].re).unwrap()
            });
            // pairing: q[0]+q[3] ~ 0, q[1]+q[2] ~ 0
            let pair = ((vals[idx[0]] + vals[idx[3]]).norm())
                .max((vals[idx[1]] + vals[idx[2]]).norm());
            assert!(pair < 1e-12, "kappa={kappa}: pairing err={pair:e}");
            // forward pair = idx[2] (lo), idx[3] (hi)
            for (slot, n_pm) in [(idx[2], n - kappa), (idx[3], n + kappa)] {
                let q = vals[slot];
                let kmag = (c(kx * kx, 0.0) + q * q).sqrt();
                let derr = (kmag.re - n_pm).abs() + kmag.im.abs();
                assert!(derr < 1e-9, "kappa={kappa}: |k| err={derr:e}");
                let col = [vecs[0][slot], vecs[1][slot], vecs[2][slot], vecs[3][slot]];
                // NOTE: from_psi (simple constitutive) gives 2.7e-4 here —
                // the P10 from_psi_full finding; this line pins the fix.
                let w = Wave::from_psi_full(&col, &eps, &rho, &rhop, &mu, kx);
                let kn = c((kx * kx + q.norm_sqr()).sqrt(), 0.0);
                let khx = c(kx, 0.0) / kn;
                let khz = q / kn;
                // E_v = -khz*Ex + khx*Ez, E_u = Ey
                let ev = -khz * w.ex + khx * w.ez;
                let ratio = ev / w.ey;
                let want = if n_pm > n { c(0.0, -1.0) } else { c(0.0, 1.0) };
                let err = (ratio - want).norm();
                assert!(err < 1e-9, "kappa={kappa}: Ev/Eu={ratio:?} err={err:e}");
            }
        }
    }

    /// from_psi_full with (0,0,I) must reproduce from_psi to fp noise (NOT
    /// 0.0: a6 = kx*e22/e22 rounds by ~1ulp, so Hz differs ~1e-17 — hence the
    /// simple path keeps from_psi and the full path is opt-in).
    #[test]
    fn from_psi_full_reduces_to_simple() {
        let eps = diag_eps(c(2.25, 0.1), c(2.89, 0.0), c(2.4, -0.05));
        let psi = [c(0.7, -0.2), c(1.1, 0.4), c(-0.3, 0.9), c(0.5, 0.5)];
        let kx = 0.6;
        let a = Wave::from_psi(&psi, &eps, kx);
        let b = Wave::from_psi_full(
            &psi, &eps, &zero_tensor(), &zero_tensor(), &identity_tensor(), kx,
        );
        let worst = (a.ex - b.ex)
            .norm()
            .max((a.ey - b.ey).norm())
            .max((a.ez - b.ez).norm())
            .max((a.hx - b.hx).norm())
            .max((a.hy - b.hy).norm())
            .max((a.hz - b.hz).norm());
        assert!(worst < 1e-15, "reduction worst = {worst:e}");
    }

    /// P10 eig-hardening boundary record: residuals must now be <= 1e-9 for
    /// separations down to 1e-9 (pre-fix: 2.2 garbage for iso k=0, k=1e-9
    /// and aniso dn<=1e-6; 8.5e-7 for iso k=1e-6).
    #[test]
    fn full_degenerate_boundary_probe() {
        let cases: Vec<(&str, Tensor3, Tensor3, Tensor3, Tensor3)> = vec![
            ("iso k=0", diag_eps(c(2.25, 0.0), c(2.25, 0.0), c(2.25, 0.0)),
             zero_tensor(), zero_tensor(), identity_tensor()),
            ("iso k=1e-9", diag_eps(c(2.25, 0.0), c(2.25, 0.0), c(2.25, 0.0)),
             diag_eps(c(0.0, -1e-9), c(0.0, -1e-9), c(0.0, -1e-9)),
             diag_eps(c(0.0, 1e-9), c(0.0, 1e-9), c(0.0, 1e-9)), identity_tensor()),
            ("iso k=1e-6", diag_eps(c(2.25, 0.0), c(2.25, 0.0), c(2.25, 0.0)),
             diag_eps(c(0.0, -1e-6), c(0.0, -1e-6), c(0.0, -1e-6)),
             diag_eps(c(0.0, 1e-6), c(0.0, 1e-6), c(0.0, 1e-6)), identity_tensor()),
            ("aniso dn=1e-9", diag_eps(c(2.25, 0.0), c(2.25 + 1e-9, 0.0), c(2.25, 0.0)),
             zero_tensor(), zero_tensor(), identity_tensor()),
            ("aniso dn=1e-6", diag_eps(c(2.25, 0.0), c(2.25 + 1e-6, 0.0), c(2.25, 0.0)),
             zero_tensor(), zero_tensor(), identity_tensor()),
        ];
        for (name, eps, rho, rhop, mu) in &cases {
            let d = berreman_full(eps, rho, rhop, mu, 0.26);
            let (vals, vecs) = eig4(&d);
            let mut worst_res = 0.0_f64;
            for k in 0..4 {
                let col = [vecs[0][k], vecs[1][k], vecs[2][k], vecs[3][k]];
                for i in 0..4 {
                    let mut dv = c(0.0, 0.0);
                    for j in 0..4 {
                        dv = dv + d[i][j] * col[j];
                    }
                    worst_res = worst_res.max((dv - vals[k] * col[i]).norm());
                }
            }
            println!("probe {name}: residual {worst_res:e}");
            assert!(
                worst_res <= 1e-9,
                "boundary {name}: residual {worst_res:e}"
            );
        }
    }

    /// full(rho=rhop=0, mu=I) must equal the reduced ("simple") matrix.
    #[test]
    fn full_reduces_to_simple() {
        let eps = diag_eps(c(2.25, 0.0), c(2.89, 0.0), c(2.4, 0.1));
        let kx = 0.6;
        let simple = berreman_simple(&eps, kx);
        let full = berreman_full(&eps, &zero_tensor(), &zero_tensor(), &identity_tensor(), kx);
        let mut worst = 0.0_f64;
        for i in 0..4 {
            for j in 0..4 {
                worst = worst.max((simple[i][j] - full[i][j]).norm());
            }
        }
        assert!(worst < 1e-12, "full vs simple worst = {worst:e}");
    }
}
