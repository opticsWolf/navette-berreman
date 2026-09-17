// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stack assembly: transfer-matrix and scattering-matrix methods, Fresnel
//! coefficient extraction, reflectance/transmittance, the linear→circular
//! Jones conversion, and the Jones→Mueller map. Ports of the corresponding
//! `pyllama.Structure` methods and the Mueller helper in `berreman_mueller.py`.

use num_complex::ComplexFloat;
use crate::cmatrix::{c, cone, czero, mat2_inv, mat2_mul, mat4_inv, mat4_mul, mat4_similarity_diag,
                     Mat2, Mat4, C};
use crate::berreman::{berreman_full, berreman_simple, halfspace_waves, layer_waves_full,
                      layer_waves_simple, propagation_diag, LayerWaves, Tensor3};
use crate::expm::mat4_expm;
use crate::roughness::interface_factors;

/// One layer as seen by the stack solver: its interface matrix P (eigenvectors
/// in columns), the diagonal propagation phases Q, and the raw partial-wave
/// eigenvalues `q_raw` (kz = k0·q_raw, needed for roughness). Half-spaces carry
/// Q = [1,1,1,1] (zero thickness).
#[derive(Clone)]
pub struct SLayer {
    pub p: Mat4,
    pub q: [C; 4],
    pub q_raw: [C; 4],
}

impl SLayer {
    pub fn from_waves(w: &LayerWaves, k0: f64, thickness: f64) -> SLayer {
        SLayer {
            p: w.p,
            q: propagation_diag(&w.q, k0, thickness),
            q_raw: w.q,
        }
    }
    pub fn half_space(w: &LayerWaves) -> SLayer {
        SLayer {
            p: w.p,
            q: [cone(); 4],
            q_raw: w.q,
        }
    }
}

/// 2x2 Jones matrices: reflection and transmission, p/s basis.
#[derive(Clone, Copy)]
pub struct Jones {
    pub refl: Mat2,
    pub trans: Mat2,
}

// ───────────────────────── transfer-matrix path ─────────────────────────

/// Transfer matrix of the interior layers, product T_{N-1}…T_0 (no half-spaces).
fn transfer_partial(layers: &[SLayer]) -> Option<Mat4> {
    let mut t = crate::cmatrix::mat4_identity();
    for l in layers {
        let t_layer = mat4_similarity_diag(&l.p, &l.q)?;
        t = mat4_mul(&t_layer, &t);
    }
    Some(t)
}

/// Full transfer matrix: exit.P^{-1} · T_partial · entry.P.
pub fn transfer_matrix(entry: &SLayer, layers: &[SLayer], exit: &SLayer) -> Option<Mat4> {
    let t = transfer_partial(layers)?;
    let exit_pinv = mat4_inv(&exit.p)?;
    let tmp = mat4_mul(&t, &entry.p);
    Some(mat4_mul(&exit_pinv, &tmp))
}

// ───────────────────────── exponential-matrix path ───────────────────────

/// Exponential layer matrix E_j = exp(i·k0·d_j·Δ_j) (live
/// `_build_exponential_matrix_partial`: `expm(1j·D·thickness·k0)`).
/// Needs eps (+ MO set) and kx, so it takes &LayerSpec, not &SLayer.
fn exp_layer(ls: &LayerSpec, kx: f64, k0: f64) -> Mat4 {
    let d = match &ls.full {
        Some((rho, rhop, mu)) => berreman_full(&ls.eps, rho, rhop, mu, kx),
        None => berreman_simple(&ls.eps, kx),
    };
    let h = c(0.0, k0 * ls.thickness_nm);
    let mut a = d;
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] *= h;
        }
    }
    mat4_expm(&a)
}

/// Full exponential matrix: exit.P⁻¹ · (E_{N-1}…E_0) · entry.P.
/// Mirrors transfer_matrix() line-for-line (same half-space sandwich, same
/// pre-multiply layer order); only the interior layer matrices differ.
/// `slayers` (eigen-basis) are still built in solve_stack — EM needs the
/// entry/exit SLayer sandwich; interior eig work is duplicated for EM in v1.
pub fn exponential_matrix(
    entry: &SLayer,
    layers: &[LayerSpec],
    exit: &SLayer,
    kx: f64,
    k0: f64,
) -> Option<Mat4> {
    let mut t = crate::cmatrix::mat4_identity();
    for ls in layers {
        t = mat4_mul(&exp_layer(ls, kx, k0), &t);
    }
    let exit_pinv = mat4_inv(&exit.p)?;
    Some(mat4_mul(&mat4_mul(&exit_pinv, &t), &entry.p))
}

/// Fresnel coefficients from a transfer matrix (`pyllama._get_fresnel_TM`).
pub fn fresnel_from_transfer(tm: &Mat4) -> Jones {
    let deno = tm[2][2] * tm[3][3] - tm[3][2] * tm[2][3];
    let r_pp = (tm[3][0] * tm[2][3] - tm[2][0] * tm[3][3]) / deno;
    let r_ps = (tm[2][0] * tm[3][2] - tm[3][0] * tm[2][2]) / deno;
    let r_sp = (tm[3][1] * tm[2][3] - tm[2][1] * tm[3][3]) / deno;
    let r_ss = (tm[2][1] * tm[3][2] - tm[3][1] * tm[2][2]) / deno;
    let t_pp = tm[0][0] + tm[0][2] * r_pp + tm[0][3] * r_ps;
    let t_ps = tm[1][0] + tm[1][2] * r_pp + tm[1][3] * r_ps;
    let t_sp = tm[0][1] + tm[0][2] * r_sp + tm[0][3] * r_ss;
    let t_ss = tm[1][1] + tm[1][2] * r_sp + tm[1][3] * r_ss;
    Jones {
        refl: [[r_pp, r_sp], [r_ps, r_ss]],
        trans: [[t_pp, t_sp], [t_ps, t_ss]],
    }
}

// ───────────────────────── scattering-matrix path ───────────────────────

/// Partial scattering matrix between two successive layers
/// (`pyllama.build_scattering_matrix_to_next`). Q is diagonal so the forward /
/// backward propagation matrices are diagonal too. If `rough_type != 0` the
/// interface is dressed with the roughness form factors (Route 1).
fn s_to_next(a: &SLayer, b: &SLayer, k0: f64, rough_type: i32, sigma: f64) -> Option<Mat4> {
    // P_out columns: a0, a1, -b2, -b3 ; P_in columns: b0, b1, -a2, -a3
    let mut p_out = [[czero(); 4]; 4];
    let mut p_in = [[czero(); 4]; 4];
    for i in 0..4 {
        p_out[i][0] = a.p[i][0];
        p_out[i][1] = a.p[i][1];
        p_out[i][2] = -b.p[i][2];
        p_out[i][3] = -b.p[i][3];
        p_in[i][0] = b.p[i][0];
        p_in[i][1] = b.p[i][1];
        p_in[i][2] = -a.p[i][2];
        p_in[i][3] = -a.p[i][3];
    }
    let p_in_inv = mat4_inv(&p_in)?;
    let mut mid = mat4_mul(&p_in_inv, &p_out); // P_in^{-1} P_out

    // Roughness dressing of the bare interface coupling. Applying the factors to
    // `mid` (before the propagation phases) is equivalent to applying them to
    // the final S, since both are element-wise diagonal scalings.
    if rough_type != 0 && sigma != 0.0 {
        let wfac = interface_factors(&a.q_raw, &b.q_raw, k0, sigma, rough_type);
        for i in 0..4 {
            for j in 0..4 {
                mid[i][j] = mid[i][j] * wfac[i][j];
            }
        }
    }

    // Q_forward = diag(q0, q1, 1, 1); Q_backward = diag(1, 1, q2, q3).
    // S = Q_backward^{-1} · mid · Q_forward.
    let qf = [a.q[0], a.q[1], cone(), cone()];
    let qb_inv = [cone(), cone(), a.q[2].recip(), a.q[3].recip()];
    let mut s = [[czero(); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            s[i][j] = qb_inv[i] * mid[i][j] * qf[j];
        }
    }
    Some(s)
}

#[inline]
fn block(s: &Mat4, br: usize, bc: usize) -> Mat2 {
    let r = br * 2;
    let cc = bc * 2;
    [
        [s[r][cc], s[r][cc + 1]],
        [s[r + 1][cc], s[r + 1][cc + 1]],
    ]
}

#[inline]
fn mat2_add(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [a[0][0] + b[0][0], a[0][1] + b[0][1]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1]],
    ]
}

#[inline]
fn mat2_iden() -> Mat2 {
    [[cone(), czero()], [czero(), cone()]]
}

/// Raise a transfer/exponential unit-cell matrix to the p-th power
/// (binary exponentiation, O(log p) matmuls). Left-multiplication preserves
/// our transfer layer order (t = T_layer · t). p = 0 gives identity.
pub fn mat4_pow(m: &Mat4, mut p: u32) -> Mat4 {
    let mut base = *m;
    let mut acc = crate::cmatrix::mat4_identity();
    while p > 0 {
        if p & 1 == 1 {
            acc = mat4_mul(&base, &acc);
        }
        base = mat4_mul(&base, &base);
        p >>= 1;
    }
    acc
}

/// Redheffer power S^⊗p for a unit-cell scattering matrix.
/// S^⊗2 = combine(S, S); binary exponentiation identical to mat4_pow
/// but with s_combine as the product. Valid because every cell is identical
/// (and s_combine is associative — pinned by G10a, load-bearing).
fn s_pow(s: &Mat4, mut p: u32) -> Mat4 {
    let mut base = *s;
    let mut acc = crate::cmatrix::mat4_identity();
    while p > 0 {
        if p & 1 == 1 {
            acc = s_combine(&base, &acc);
        }
        base = s_combine(&base, &base);
        p >>= 1;
    }
    acc
}

/// Redheffer combination of two 4x4 scattering matrices
/// (`pyllama.combine_scattering_matrices`).
fn s_combine(ab: &Mat4, bc: &Mat4) -> Mat4 {
    let ab00 = block(ab, 0, 0);
    let ab01 = block(ab, 0, 1);
    let ab10 = block(ab, 1, 0);
    let ab11 = block(ab, 1, 1);
    let bc00 = block(bc, 0, 0);
    let bc01 = block(bc, 0, 1);
    let bc10 = block(bc, 1, 0);
    let bc11 = block(bc, 1, 1);

    // C = (I - ab01·bc10)^{-1}
    let prod = mat2_mul(&ab01, &bc10);
    let mut imp = mat2_iden();
    for i in 0..2 {
        for j in 0..2 {
            imp[i][j] = imp[i][j] - prod[i][j];
        }
    }
    let cmat = mat2_inv(&imp).unwrap_or_else(mat2_iden);

    let ac00 = mat2_mul(&mat2_mul(&bc00, &cmat), &ab00);
    let ac01 = mat2_add(&bc01, &mat2_mul(&mat2_mul(&mat2_mul(&bc00, &cmat), &ab01), &bc11));
    let ac10 = mat2_add(&ab10, &mat2_mul(&mat2_mul(&mat2_mul(&ab11, &bc10), &cmat), &ab00));
    let inner = mat2_add(&mat2_iden(), &mat2_mul(&mat2_mul(&bc10, &cmat), &ab01));
    let ac11 = mat2_mul(&mat2_mul(&ab11, &inner), &bc11);

    [
        [ac00[0][0], ac00[0][1], ac01[0][0], ac01[0][1]],
        [ac00[1][0], ac00[1][1], ac01[1][0], ac01[1][1]],
        [ac10[0][0], ac10[0][1], ac11[0][0], ac11[0][1]],
        [ac10[1][0], ac10[1][1], ac11[1][0], ac11[1][1]],
    ]
}

/// Full scattering matrix of a single-period stack (`pyllama.build_scattering_matrix`
/// with N_periods = 1). `rough` holds one (rough_type, sigma) per interface, in
/// order: [entry|layer0, layer0|layer1, …, layer_{n-1}|exit] (length n+1).
pub fn scattering_matrix(
    entry: &SLayer,
    layers: &[SLayer],
    exit: &SLayer,
    k0: f64,
    rough: &[(i32, f64)],
) -> Option<Mat4> {
    let n = layers.len();
    // interior interfaces l[k]|l[k+1] carry rough[k+1] (interface index kl+1)
    let s_in = scattering_interior(layers, k0, &rough[1..n])?;
    // attach entry and exit half-spaces (interfaces 0 and n)
    let (rt0, sg0) = rough[0];
    let (rtn, sgn) = rough[n];
    let s_entry = s_to_next(entry, &layers[0], k0, rt0, sg0)?;
    let s_exit = s_to_next(&layers[n - 1], exit, k0, rtn, sgn)?;
    let s = s_combine(&s_in, &s_exit);
    let s = s_combine(&s_entry, &s);
    Some(s)
}

/// Interior scattering matrix of a bare layer sequence (no half-spaces):
/// combines s_to_next(l[k], l[k+1]) for k = n-2..0. Extracted verbatim from
/// scattering_matrix(); `rough_interior` has length n-1, element k dressing
/// l[k]|l[k+1].
fn scattering_interior(
    slayers: &[SLayer],
    k0: f64,
    rough_interior: &[(i32, f64)],
) -> Option<Mat4> {
    let n = slayers.len();
    let mut s = crate::cmatrix::mat4_identity();
    if n >= 2 {
        for kl in (0..(n - 1)).rev() {
            let (rt, sg) = rough_interior[kl];
            let s_layer = s_to_next(&slayers[kl], &slayers[kl + 1], k0, rt, sg)?;
            s = s_combine(&s_layer, &s);
        }
    }
    Some(s)
}

/// Unit-cell transfer product T_cell = T_{L-1}·…·T_0 (no half-spaces).
fn transfer_cell(cell_sl: &[SLayer]) -> Option<Mat4> {
    let mut t = crate::cmatrix::mat4_identity();
    for sl in cell_sl {
        t = mat4_mul(&mat4_similarity_diag(&sl.p, &sl.q)?, &t);
    }
    Some(t)
}

/// Fresnel coefficients from a scattering matrix (`pyllama._get_fresnel_SM`).
pub fn fresnel_from_scattering(sm: &Mat4) -> Jones {
    Jones {
        refl: [[sm[2][0], sm[2][1]], [sm[3][0], sm[3][1]]],
        trans: [[sm[0][0], sm[0][1]], [sm[1][0], sm[1][1]]],
    }
}

// ───────────────────────── observables ──────────────────────────────────

/// Reflectance / transmittance 2x2 power matrices from a Jones pair.
/// `factor = Kz_exit / Kz_entry`.
pub fn refl_trans_power(j: &Jones, factor: f64) -> (Mat2, Mat2) {
    let mut r = [[czero(); 2]; 2];
    let mut t = [[czero(); 2]; 2];
    for i in 0..2 {
        for k in 0..2 {
            r[i][k] = c(j.refl[i][k].norm_sqr(), 0.0);
            t[i][k] = c(factor * j.trans[i][k].norm_sqr(), 0.0);
        }
    }
    (r, t)
}

/// Convert a linear-basis Jones pair to the circular basis
/// (`pyllama.fresnel_to_fresnel_circ`).
pub fn jones_to_circular(j: &Jones) -> Jones {
    let f: Mat2 = [[cone(), cone()], [c(0.0, -1.0), c(0.0, 1.0)]];
    let b: Mat2 = [[cone(), cone()], [c(0.0, 1.0), c(0.0, -1.0)]];
    let binv = mat2_inv(&b).unwrap();
    let finv = mat2_inv(&f).unwrap();
    let refl_c = mat2_mul(&mat2_mul(&binv, &j.refl), &f);
    let trans_c = mat2_mul(&mat2_mul(&finv, &j.trans), &f);
    Jones {
        refl: refl_c,
        trans: trans_c,
    }
}

/// Mueller matrix (4x4, returned complex; imaginary parts are ~0) from a 2x2
/// Jones matrix. `berreman_mueller.mueller_from_jones_matrix`.
pub fn mueller_from_jones(j: &Mat2) -> Mat4 {
    let (a, ainv) = mueller_a_basis();
    // kron(J, conj(J)): 4x4
    let mut k = [[czero(); 4]; 4];
    for i in 0..2 {
        for j2 in 0..2 {
            for p in 0..2 {
                for q in 0..2 {
                    k[i * 2 + p][j2 * 2 + q] = j[i][j2] * j[p][q].conj();
                }
            }
        }
    }
    let ak = mat4_mul(&a, &k);
    mat4_mul(&ak, &ainv)
}

/// The Stokes-basis A matrix of `mueller_from_jones` (1/√2 factors,
/// `berreman_mueller.py:156-160`) and its inverse — shared with the Phase-11
/// differential generator (`polarizance.rs`), which is its infinitesimal form
/// (same A ⇒ the two formalisms are anchored to one basis).
pub(crate) fn mueller_a_basis() -> (Mat4, Mat4) {
    let s = 1.0 / 2.0_f64.sqrt();
    let a: Mat4 = [
        [c(s, 0.0), czero(), czero(), c(s, 0.0)],
        [c(s, 0.0), czero(), czero(), c(-s, 0.0)],
        [czero(), c(s, 0.0), c(s, 0.0), czero()],
        [czero(), c(0.0, s), c(0.0, -s), czero()],
    ];
    let ainv = mat4_inv(&a).unwrap();
    (a, ainv)
}

// ───────────────────────── full stack solve ─────────────────────────────

/// One anisotropic interior layer's optical inputs.
pub struct LayerSpec {
    pub eps: Tensor3,
    pub thickness_nm: f64,
    /// Optional full magneto-optic description (ρ, ρ', μ). When `None`, the
    /// reduced Berreman matrix is used.
    pub full: Option<(Tensor3, Tensor3, Tensor3)>,
    /// Roughness (rough_type, sigma_nm) of the interface at the *front* of this
    /// layer (between the previous medium and this layer). `(0, _)` = smooth.
    /// Route-1 specular attenuation; only applied in the scattering method.
    pub front_roughness: (i32, f64),
}

/// Method selector for the stack solve.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Method {
    Scattering,
    Transfer,
    Exponential, // transfer via direct matrix exponential (== live "EM")
}

/// How the exit half-space is described.
#[derive(Clone)]
pub enum ExitSpec {
    /// Scalar index -> analytic isotropic basis (existing path, bit-identical).
    Isotropic(C),
    /// Full permittivity tensor (+ optional MO set) -> numeric eigenbasis.
    Anisotropic {
        eps: Tensor3,
        full: Option<(Tensor3, Tensor3, Tensor3)>, // (rho, rhop, mu)
    },
}

/// Which power formula produced SolveResult.t_power.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PowerMethod {
    /// T = Re(kz_exit/kz_entry)·|J|² (isotropic exits; bit-identical legacy).
    JonesFactor,
    /// Flux ratio over exit eigenmodes (anisotropic exits).
    Flux,
}

/// Geometry / boundary conditions for a single (wavelength, angle) point.
pub struct Geometry {
    pub wl_nm: f64,
    pub theta_in_rad: f64,
    pub n_entry: C,
    /// Scalar exit index: used iff exit == Isotropic (Snell + JonesFactor).
    pub n_exit: C,
    /// Exit-half-space description (default: Isotropic(n_exit)).
    pub exit: ExitSpec,
    /// Roughness (rough_type, sigma_nm) of the final interface (last layer |
    /// exit half-space). `(0, _)` = smooth.
    pub exit_roughness: (i32, f64),
}

impl Geometry {
    /// Convenience: isotropic exit (today's behavior).
    pub fn isotropic(
        wl_nm: f64,
        theta_in_rad: f64,
        n_entry: C,
        n_exit: C,
        exit_roughness: (i32, f64),
    ) -> Geometry {
        Geometry {
            wl_nm,
            theta_in_rad,
            n_entry,
            n_exit,
            exit: ExitSpec::Isotropic(n_exit),
            exit_roughness,
        }
    }
}

/// Build the exit SLayer from an ExitSpec. Returns None when the exit
/// eigenbasis cannot be sorted cleanly (grazing/evanescent exit modes).
pub(crate) fn build_exit(exit: &ExitSpec, kx: f64) -> Option<SLayer> {
    let w: LayerWaves = match exit {
        ExitSpec::Isotropic(n) => halfspace_waves(*n * *n, kx),
        ExitSpec::Anisotropic { eps, full } => {
            // A half-space needs sorted partial waves but NO propagation
            // (Q = 1 via SLayer::half_space). layer_waves_simple special-cases
            // isotropic tensors back to the analytic basis, so this is a
            // strict generalization of the scalar path.
            match full {
                Some((rho, rhop, mu)) => layer_waves_full(eps, rho, rhop, mu, kx),
                None => layer_waves_simple(eps, kx),
            }
        }
    };
    if !w.clean {
        return None; // unsortable exit basis fails the point (NaN row)
    }
    Some(SLayer::half_space(&w))
}

/// (Ex, Ey, Hx, Hy) of eigenmode column m of an SLayer P matrix.
/// ψ-column layout is [Ex, Hy, Ey, −Hx] — see berreman.rs::Wave::from_psi.
#[inline]
pub fn mode_eh4(p: &Mat4, m: usize) -> (C, C, C, C) {
    (p[0][m], p[2][m], -p[3][m], p[1][m])
}

/// Time-averaged z-flux of a superposition b0·mode0 + b1·mode1, WITH the
/// cross term (exit modes are generally NOT power-orthogonal):
///   F = ½·Re(Ex·Hy* − Ey·Hx*),  E = Σ bm·Em, H = Σ bm·Hm.
/// Building totals first (instead of F0/F1/Fx separately) is exact and short.
/// Fields are in Berreman units (H impedance-scaled); the common scale
/// cancels in ratios, so only Re(E × H*)_z matters.
fn superposition_flux(p: &Mat4, b: &[C; 2]) -> f64 {
    let (ex0, ey0, hx0, hy0) = mode_eh4(p, 0);
    let (ex1, ey1, hx1, hy1) = mode_eh4(p, 1);
    let ex = b[0] * ex0 + b[1] * ex1;
    let ey = b[0] * ey0 + b[1] * ey1;
    let hx = b[0] * hx0 + b[1] * hx1;
    let hy = b[0] * hy0 + b[1] * hy1;
    0.5 * (ex * hy.conj() - ey * hx.conj()).re
}

/// Incident flux per canonical input: unit forward mode of the entry basis.
/// Columns 0 (p) and 1 (s) of entry.p at unit amplitude.
fn entry_flux(entry: &SLayer) -> Option<[f64; 2]> {
    let fp = superposition_flux(&entry.p, &[cone(), czero()]);
    let fs = superposition_flux(&entry.p, &[czero(), cone()]);
    if fp <= 1e-300 || fs <= 1e-300 {
        return None; // grazing/complex incidence: no well-defined incidence
    }
    Some([fp, fs])
}

/// Reflection power only (|J|²) — the T half of refl_trans_power is skipped
/// for anisotropic exits.
fn refl_power_only(j: &Jones) -> Mat2 {
    let mut r = [[czero(); 2]; 2];
    for i in 0..2 {
        for k in 0..2 {
            r[i][k] = c(j.refl[i][k].norm_sqr(), 0.0);
        }
    }
    r
}

/// Transmitted power per input polarization from exit-mode amplitudes.
/// `b_cols[j]` = [b0, b1] exit forward amplitudes for input pol j
/// (from fields::exit_amplitudes on the transfer matrix).
/// Diagonal-only: with a mode-resolved exit basis there is no (p,s)-projected
/// cross power (documented shape choice).
fn transmitted_power_flux(
    exit: &SLayer,
    b_cols: &[[C; 2]; 2],
    inc_flux: &[f64; 2],
) -> Option<Mat2> {
    let mut t = [[czero(); 2]; 2];
    for j in 0..2 {
        let f = superposition_flux(&exit.p, &b_cols[j]);
        if !(f.is_finite()) {
            return None;
        }
        t[j][j] = c(f / inc_flux[j], 0.0);
    }
    Some(t)
}

/// Result of one solve.
pub struct SolveResult {
    pub jones: Jones,
    pub jones_circ: Jones,
    pub r_power: Mat2,
    pub t_power: Mat2,
    pub factor: f64,
    pub power_method: PowerMethod,
}

/// Per-point head data shared by solve_stack and solve_stack_periodic:
/// wavevectors, scalar power factor, and the entry/exit half-space bases.
/// (Phase 2 reuses this for the fields engine — extracted here in Phase 4.)
pub(crate) struct PointHead {
    pub(crate) k0: f64,
    pub(crate) kx: f64,
    pub(crate) factor: f64,
    pub(crate) entry: SLayer,
    pub(crate) exit_sl: SLayer,
}

pub(crate) fn build_head(geom: &Geometry) -> Option<PointHead> {
    let k0 = 2.0 * std::f64::consts::PI / geom.wl_nm;
    let sin_in = geom.theta_in_rad.sin();
    let cos_in = geom.theta_in_rad.cos();
    // Kx = n_entry · sinθ (pyllama uses a real entry index; take the real part).
    let kx = geom.n_entry.re * sin_in;

    let kz_entry = geom.n_entry * c(cos_in, 0.0);
    // Scalar factor is only meaningful for isotropic exits; for anisotropic
    // exits it is still computed (harmless) but NOT used — see finish().
    // theta_out from Snell, allow complex
    let sin_out = (geom.n_entry / geom.n_exit) * c(sin_in, 0.0);
    let cos_out = (cone() - sin_out * sin_out).sqrt();
    let kz_exit = geom.n_exit * cos_out;
    let factor = (kz_exit / kz_entry).re;

    // entry (always isotropic) / exit (isotropic or tensor) half-spaces
    let entry_w = halfspace_waves(geom.n_entry * geom.n_entry, kx);
    if !entry_w.clean {
        return None;
    }
    let entry = SLayer::half_space(&entry_w);
    let exit_sl = build_exit(&geom.exit, kx)?; // anisotropic-capable
    Some(PointHead { k0, kx, factor, entry, exit_sl })
}

/// Eigen-basis for a layer sequence (L eigs, no expansion).
pub(crate) fn build_slayers(layers: &[LayerSpec], kx: f64, k0: f64) -> Vec<SLayer> {
    let mut slayers: Vec<SLayer> = Vec::with_capacity(layers.len());
    for ls in layers {
        let w = match &ls.full {
            Some((rho, rhop, mu)) => layer_waves_full(&ls.eps, rho, rhop, mu, kx),
            None => layer_waves_simple(&ls.eps, kx),
        };
        slayers.push(SLayer::from_waves(&w, k0, ls.thickness_nm));
    }
    slayers
}

/// Shared tail: Jones → circular → power → SolveResult (both power arms,
/// incl. the Phase-1 Flux arm). `tm_full` is the entry→exit transfer matrix,
/// needed only by the Flux arm; callers pass None for isotropic exits
/// (zero extra cost on the legacy path).
fn finish(
    jones: Jones,
    factor: f64,
    exit_spec: &ExitSpec,
    entry: &SLayer,
    exit_sl: &SLayer,
    tm_full: Option<Mat4>,
) -> Option<SolveResult> {
    let jones_circ = jones_to_circular(&jones);
    let (r_power, t_power, power_method) = match exit_spec {
        ExitSpec::Isotropic(_) => {
            // Legacy path, untouched (G6 bit-identity).
            let (r, t) = refl_trans_power(&jones, factor);
            (r, t, PowerMethod::JonesFactor)
        }
        ExitSpec::Anisotropic { .. } => {
            // Amplitude path needs the transfer matrix even for SM solves
            // (one extra sandwich per point; documented cost).
            let tm = tm_full?;
            let a_p = crate::fields::exit_amplitudes(&tm, [cone(), czero()])?;
            let a_s = crate::fields::exit_amplitudes(&tm, [czero(), cone()])?;
            let inc = entry_flux(entry)?;
            let b = [[a_p[0], a_p[1]], [a_s[0], a_s[1]]];
            let t = transmitted_power_flux(exit_sl, &b, &inc)?;
            (refl_power_only(&jones), t, PowerMethod::Flux)
        }
    };

    Some(SolveResult {
        jones,
        jones_circ,
        r_power,
        t_power,
        factor,
        power_method,
    })
}

/// Solve a full stack at one geometry point.
pub fn solve_stack(geom: &Geometry, layers: &[LayerSpec], method: Method) -> Option<SolveResult> {
    let head = build_head(geom)?;
    let (k0, kx, factor) = (head.k0, head.kx, head.factor);
    let (entry, exit) = (head.entry, head.exit_sl);
    // interior layers
    let slayers = build_slayers(layers, kx, k0);

    let (jones, tm_hint) = match method {
        Method::Scattering => {
            // one (rough_type, sigma) per interface: [entry|l0, l0|l1, …, l_{n-1}|exit]
            let mut rough: Vec<(i32, f64)> = Vec::with_capacity(layers.len() + 1);
            for ls in layers {
                rough.push(ls.front_roughness);
            }
            rough.push(geom.exit_roughness);
            let sm = scattering_matrix(&entry, &slayers, &exit, k0, &rough)?;
            (fresnel_from_scattering(&sm), None)
        }
        Method::Transfer => {
            let tm = transfer_matrix(&entry, &slayers, &exit)?;
            let j = fresnel_from_transfer(&tm);
            (j, Some(tm))
        }
        Method::Exponential => {
            // Same half-space sandwich as TM; layer matrices are
            // exp(i·k0·d·Δ) instead of P·Q·P⁻¹ (== live _get_fresnel_EM
            // sharing _get_fresnel_TM's extraction).
            let em = exponential_matrix(&entry, layers, &exit, kx, k0)?;
            (fresnel_from_transfer(&em), None)
        }
    };

    // Transfer matrix for the Flux arm: reuse the TM arm's, else build once.
    let tm_full = match (&geom.exit, tm_hint) {
        (ExitSpec::Isotropic(_), h) => h, // ignored by finish (zero cost)
        (ExitSpec::Anisotropic { .. }, Some(tm)) => Some(tm),
        (ExitSpec::Anisotropic { .. }, None) => {
            Some(transfer_matrix(&entry, &slayers, &exit)?)
        }
    };
    finish(jones, factor, &geom.exit, &entry, &exit, tm_full)
}

/// Solve a unit cell repeated `periods` times (fast path: L eigs, O(log N)
/// combines — no layer expansion). `periods == 1` delegates to solve_stack
/// (zero delta, G6). `periods == 0` or empty cell returns None.
///
/// SM construction (matches live `_build_scattering_matrix_partial` and the
/// expanded stack exactly): total interior = I ⊗ (W⊗I)^{N-1}, where I is the
/// bare cell interior and W⊗I the wrap-first cell (wrap = dressed last→first
/// interface, folded LEFT like live's S_period). Powering an interior-first
/// block N times would append a spurious Nth wrap — see §4.4.
pub fn solve_stack_periodic(
    geom: &Geometry,
    cell: &[LayerSpec],
    periods: u32,
    method: Method,
) -> Option<SolveResult> {
    if periods == 0 || cell.is_empty() {
        return None;
    }
    if periods == 1 {
        return solve_stack(geom, cell, method);
    }
    let head = build_head(geom)?;
    let (k0, kx, factor) = (head.k0, head.kx, head.factor);
    let (entry, exit) = (head.entry, head.exit_sl);
    // cell eigen-basis (L eigs, not N·L)
    let cell_sl = build_slayers(cell, kx, head.k0);
    let l = cell_sl.len();

    // Powered transfer sandwich (TM answer directly; Flux arm for SM/EM).
    // O(log N) matmuls, no expansion.
    let tcell = transfer_cell(&cell_sl)?;
    let tpow = mat4_pow(&tcell, periods);
    let exit_pinv = mat4_inv(&exit.p)?;
    let tm_full = mat4_mul(&mat4_mul(&exit_pinv, &tpow), &entry.p);

    let jones = match method {
        Method::Scattering => {
            // interior dressings: interface l[k]|l[k+1] carries
            // cell[k+1].front_roughness (== expanded stacking).
            let interior: Vec<(i32, f64)> =
                (1..l).map(|k| cell[k].front_roughness).collect();
            let bare = scattering_interior(&cell_sl, k0, &interior)?;
            // wrap interface (last|first of next copy) dressed with
            // cell[0].front — also used once at entry|cell below, exactly
            // the expanded count (N-1 wraps + 1 entry = N uses).
            let (rt0, sg0) = cell[0].front_roughness;
            let wrap = s_to_next(&cell_sl[l - 1], &cell_sl[0], k0, rt0, sg0)?;
            let wcell = s_combine(&wrap, &bare); // wrap-first cell [W, I]
            let s = s_combine(&bare, &s_pow(&wcell, periods - 1));
            let s_entry = s_to_next(&entry, &cell_sl[0], k0, rt0, sg0)?;
            let (rtn, sgn) = geom.exit_roughness;
            let s_exit = s_to_next(&cell_sl[l - 1], &exit, k0, rtn, sgn)?;
            let s = s_combine(&s_combine(&s_entry, &s), &s_exit);
            fresnel_from_scattering(&s)
        }
        Method::Transfer => fresnel_from_transfer(&tm_full),
        Method::Exponential => {
            let mut ecell = crate::cmatrix::mat4_identity();
            for ls in cell {
                ecell = mat4_mul(&exp_layer(ls, kx, k0), &ecell);
            }
            let epow = mat4_pow(&ecell, periods);
            let em = mat4_mul(&mat4_mul(&exit_pinv, &epow), &entry.p);
            fresnel_from_transfer(&em)
        }
    };
    finish(jones, factor, &geom.exit, &entry, &exit, Some(tm_full))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::berreman::diag_eps;
    use crate::cmatrix::{c, mat4_scale, mat4_zero};

    fn xorshift(state: &mut u64) -> f64 {
        // deterministic [-1, 1) doubles; no new deps (MSRV-safe, audit-trivial)
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn rand_mat4(state: &mut u64) -> Mat4 {
        let mut m = mat4_zero();
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = c(xorshift(state), xorshift(state));
            }
        }
        m
    }

    fn max_diff(a: &Mat4, b: &Mat4) -> f64 {
        let mut m: f64 = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                m = m.max((a[i][j] - b[i][j]).norm());
            }
        }
        m
    }

    /// G10a: mat4_pow vs naive repeated product (p = 0 must be identity).
    #[test]
    fn pow_algebra() {
        let mut st = 0x9E3779B97F4A7C15u64;
        for _ in 0..20 {
            let a = rand_mat4(&mut st);
            let mut naive = crate::cmatrix::mat4_identity();
            assert_eq!(mat4_pow(&a, 0), naive); // pins the p=0 edge
            for p in 1..8u32 {
                naive = mat4_mul(&a, &naive);
                let d = max_diff(&mat4_pow(&a, p), &naive);
                assert!(d < 1e-12, "pow {p} diff {d:e}");
            }
        }
    }

    /// G10a: Redheffer identity (both sides), s_pow vs naive, associativity.
    /// Associativity is LOAD-BEARING: binary exponentiation is invalid without
    /// it (Redheffer cascading of two-ports is associative — verified here).
    #[test]
    fn redheffer_identity_and_pow() {
        let mut st = 0x243F6A8885A308D3u64;
        let i = crate::cmatrix::mat4_identity();
        for _ in 0..20 {
            // physically-shaped S (blocks < 1 in norm, like real scatterers)
            let s = mat4_scale(&rand_mat4(&mut st), c(0.1, 0.0));
            assert!(max_diff(&s_combine(&i, &s), &s) < 1e-14);
            assert!(max_diff(&s_combine(&s, &i), &s) < 1e-14);
            // s_pow vs naive repeated LEFT-combine, p = 1..6
            let mut naive = i;
            for p in 1..7u32 {
                naive = s_combine(&s, &naive);
                let d = max_diff(&s_pow(&s, p), &naive);
                assert!(d < 1e-12, "spow {p} diff {d:e}");
            }
        }
        // associativity on random triples (scaled 0.1 like real scatterers so
        // the internal (I − ab01·bc10) inverse stays well-conditioned and the
        // unwrap_or_else-identity fallback never fires — the fallback is not
        // associative, and testing THROUGH it would be a false negative).
        for _ in 0..10 {
            let sc = |m: Mat4| mat4_scale(&m, c(0.1, 0.0));
            let (a, b, cc) = (sc(rand_mat4(&mut st)), sc(rand_mat4(&mut st)), sc(rand_mat4(&mut st)));
            let ab_c = s_combine(&s_combine(&a, &b), &cc);
            let a_bc = s_combine(&a, &s_combine(&b, &cc));
            let d = max_diff(&ab_c, &a_bc);
            assert!(d < 1e-12, "associativity diff {d:e}");
        }
    }

    /// Single dielectric slab, normal incidence: both methods agree and the
    /// front-interface reflectance matches the analytic Fresnel value for a
    /// thick/again-isotropic exit. Here we check a bare interface (zero-thickness
    /// limit is awkward, so use a half-wave-ish slab and just assert SM==TM and
    /// energy conservation for a lossless isotropic slab).
    #[test]
    fn isotropic_slab_methods_agree_and_conserve_energy() {
        let eps = diag_eps(c(2.25, 0.0), c(2.25, 0.0), c(2.25, 0.0));
        let layers = [LayerSpec { eps, thickness_nm: 137.0, full: None, front_roughness: (0, 0.0) }];
        let geom = Geometry::isotropic(
            550.0,
            20.0_f64.to_radians(),
            c(1.0, 0.0),
            c(1.0, 0.0),
            (0, 0.0),
        );
        let sm = solve_stack(&geom, &layers, Method::Scattering).unwrap();
        let tm = solve_stack(&geom, &layers, Method::Transfer).unwrap();
        let mut worst = 0.0_f64;
        for i in 0..2 {
            for j in 0..2 {
                worst = worst.max((sm.jones.refl[i][j] - tm.jones.refl[i][j]).norm());
                worst = worst.max((sm.jones.trans[i][j] - tm.jones.trans[i][j]).norm());
            }
        }
        assert!(worst < 1e-9, "SM vs TM worst = {worst:e}");
        // lossless, symmetric ambient: R+T = 1 for each input polarization.
        for col in 0..2 {
            let tot = sm.r_power[0][col].re + sm.r_power[1][col].re
                + sm.t_power[0][col].re + sm.t_power[1][col].re;
            assert!((tot - 1.0).abs() < 1e-9, "R+T col {col} = {tot}");
        }
    }

    /// G7a: anisotropic exit carrying an ISOTROPIC tensor must reproduce the
    /// scalar path bit-identically (via the is_isotropic shortcut), while
    /// reporting PowerMethod::Flux (the aniso *route* was taken).
    #[test]
    fn aniso_isotropic_exit_is_bit_identical() {
        // (scalar exit index, matching tensor): each pair is the SAME medium —
        // the draft compared a vacuum tensor against a 1.5 scalar (different
        // physics, not a code bug). 1.5² = 2.25 is exact in f64, so the
        // shortcut input is bit-identical to the scalar path.
        let exits = [
            (c(1.0, 0.0), diag_eps(c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0))),
            (c(1.5, 0.0), diag_eps(c(2.25, 0.0), c(2.25, 0.0), c(2.25, 0.0))),
        ];
        for (n_exit, eps) in exits {
            for method in [Method::Scattering, Method::Transfer] {
                let g_iso =
                    Geometry::isotropic(550.0, 0.35, c(1.0, 0.0), n_exit, (0, 0.0));
                let mut g_an =
                    Geometry::isotropic(550.0, 0.35, c(1.0, 0.0), n_exit, (0, 0.0));
                g_an.exit = ExitSpec::Anisotropic { eps, full: None };
                let layers = [LayerSpec {
                    eps: diag_eps(c(2.25, 0.0), c(2.89, 0.0), c(2.25, 0.0)),
                    thickness_nm: 250.0,
                    full: None,
                    front_roughness: (0, 0.0),
                }];
                let a = solve_stack(&g_iso, &layers, method).unwrap();
                let b = solve_stack(&g_an, &layers, method).unwrap();
                assert_eq!(b.power_method, PowerMethod::Flux);
                assert_eq!(a.power_method, PowerMethod::JonesFactor);
                for i in 0..2 {
                    for j in 0..2 {
                        assert_eq!(a.jones.refl[i][j], b.jones.refl[i][j]);
                        assert_eq!(a.jones.trans[i][j], b.jones.trans[i][j]);
                        assert_eq!(a.r_power[i][j], b.r_power[i][j]);
                    }
                }
            }
        }
    }

    /// G7c flux bridge: on an ISOTROPIC exit, transmitted_power_flux (via
    /// fields::exit_amplitudes on the TM matrix) must reproduce the scalar
    /// JonesFactor T — proves the flux machinery before trusting it on
    /// anisotropic exits. Canonical-ish epsilons, both angles.
    #[test]
    fn flux_power_matches_scalar_on_isotropic_exit() {
        let cases = [
            diag_eps(c(2.25, 0.0), c(2.25, 0.0), c(2.25, 0.0)),
            diag_eps(c(2.25, 0.0), c(2.89, 0.0), c(2.25, 0.0)),
            diag_eps(c(2.25, 0.05), c(3.0, 0.2), c(2.25, 0.05)),
        ];
        for eps in cases {
            for theta in [0.0, 0.35, 0.7] {
                let g =
                    Geometry::isotropic(550.0, theta, c(1.0, 0.0), c(1.5, 0.0), (0, 0.0));
                let layers = [LayerSpec {
                    eps,
                    thickness_nm: 250.0,
                    full: None,
                    front_roughness: (0, 0.0),
                }];
                let kx = 1.0 * theta.sin();
                let entry =
                    SLayer::half_space(&halfspace_waves(c(1.0, 0.0), kx));
                let exit =
                    SLayer::half_space(&halfspace_waves(c(2.25, 0.0), kx));
                let mut slayers = Vec::new();
                let k0 = 2.0 * std::f64::consts::PI / 550.0;
                for ls in &layers {
                    let w = layer_waves_simple(&ls.eps, kx);
                    slayers.push(SLayer::from_waves(&w, k0, ls.thickness_nm));
                }
                let tm = transfer_matrix(&entry, &slayers, &exit).unwrap();
                let a_p =
                    crate::fields::exit_amplitudes(&tm, [cone(), czero()]).unwrap();
                let a_s =
                    crate::fields::exit_amplitudes(&tm, [czero(), cone()]).unwrap();
                let inc = entry_flux(&entry).unwrap();
                let t = transmitted_power_flux(
                    &exit,
                    &[[a_p[0], a_p[1]], [a_s[0], a_s[1]]],
                    &inc,
                )
                .unwrap();
                let s = solve_stack(&g, &layers, Method::Transfer).unwrap();
                let mut worst: f64 = 0.0;
                for j in 0..2 {
                    worst = worst.max((t[j][j] - s.t_power[j][j]).norm());
                }
                assert!(worst < 1e-12, "flux bridge worst = {worst:e}");
            }
        }
    }
}
