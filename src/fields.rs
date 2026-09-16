// SPDX-License-Identifier: LGPL-3.0-or-later
//! Eigenmode amplitude helpers shared by the anisotropic-exit power path
//! (Phase 1, `transfer.rs`) and the internal-fields engine (Phase 2).
//!
//! v1 scope: entry/exit amplitude solves only. Depth propagation (`StackFrames`,
//! `fields()`) arrives in Phase 2; this module exists now so Phase 1's flux
//! power imports from the same place Phase 2 will extend.

use crate::berreman::{diag_eps, Tensor3, Wave};
use crate::cmatrix::{
    c, cone, czero, mat4_identity, mat4_inv, mat4_mul, mat4_similarity_diag, Mat4, C,
};
use crate::transfer::{
    build_head, build_slayers, transfer_matrix, ExitSpec, Geometry, LayerSpec, Method, SLayer,
};
use num_complex::ComplexFloat;

/// Solve entry-plane amplitude vector a = [i0, i1, a2, a3] from the full
/// transfer matrix M (entry-plane → exit-plane amplitudes) and the incident
/// pair. Outgoing condition b2 = b3 = 0 gives the reflected pair:
///   tm[r][0]*i0 + tm[r][1]*i1 + tm[r][2]*a2 + tm[r][3]*a3 = 0  (r = 2, 3).
/// Returns None if the 2x2 system is singular.
pub fn entry_amplitudes(tm: &Mat4, incident: [C; 2]) -> Option<[C; 4]> {
    let d = tm[2][2] * tm[3][3] - tm[2][3] * tm[3][2];
    if d.norm() < 1e-300 {
        return None;
    }
    let di = d.recip();
    let rhs2 = -(tm[2][0] * incident[0] + tm[2][1] * incident[1]);
    let rhs3 = -(tm[3][0] * incident[0] + tm[3][1] * incident[1]);
    let a2 = (rhs2 * tm[3][3] - rhs3 * tm[2][3]) * di;
    let a3 = (tm[2][2] * rhs3 - tm[3][2] * rhs2) * di;
    Some([incident[0], incident[1], a2, a3])
}

/// Exit-plane forward amplitudes b = [b0, b1] for one incident pair:
/// b = M·a with a from `entry_amplitudes` (b2 = b3 = 0 by construction —
/// asserted in the unit test). Components are in the exit eigenbasis, i.e.
/// per unit incident mode amplitude.
pub fn exit_amplitudes(tm: &Mat4, incident: [C; 2]) -> Option<[C; 2]> {
    let a = entry_amplitudes(tm, incident)?;
    let mut b = [C::new(0.0, 0.0); 2];
    for r in 0..2 {
        b[r] = tm[r][0] * a[0] + tm[r][1] * a[1] + tm[r][2] * a[2] + tm[r][3] * a[3];
    }
    Some(b)
}

// ───────────────────────── depth propagation (Phase 2) ───────────────────

/// Full 6-vector field at one depth for one input polarization
/// (Berreman units — H impedance-scaled; ratios/flux only).
#[derive(Clone, Copy)]
pub struct FieldVec {
    pub e: [C; 3],
    pub h: [C; 3],
}

/// One depth sample: global z (nm, 0 = entry plane) + fields per input pol.
/// `per_input[0]` = response to p-incidence `[1,0]`, `per_input[1]` to `[0,1]`.
pub struct FieldPoint {
    pub z_nm: f64,
    pub per_input: [FieldVec; 2],
}

/// ψ = [Ex, Hy, Ey, −Hx] (+ layer eps, kx) -> full 6-vector.
/// Implemented VIA Wave::from_psi (sync by construction, not by test).
/// Ez from ∇·D = 0 (no free charges); Hz = kx·Ey (ky = 0 plane of incidence).
/// SIMPLE-constitutive layers only: with (rho, rhop, mu) != (0, 0, I) the
/// ez/hz reconstruction needs the full 2x2 elimination — use psi_to_eh_full
/// (P10 finding; the old "valid for MO too" claim was wrong for Ez/Hz).
/// Tangential components come straight from ψ and are exact either way.
pub fn psi_to_eh(psi: &[C; 4], eps: &Tensor3, kx: f64) -> FieldVec {
    let w = Wave::from_psi(psi, eps, kx);
    FieldVec {
        e: [w.ex, w.ey, w.ez],
        h: [w.hx, w.hy, w.hz],
    }
}

/// Full-tensor unpack (Phase 10): longitudinal components via
/// Wave::from_psi_full. Chosen per-sample by fields_at_depths from
/// layer.full / exit-spec full (None -> psi_to_eh, bit-identical history).
#[allow(clippy::too_many_arguments)]
pub fn psi_to_eh_full(
    psi: &[C; 4],
    eps: &Tensor3,
    rho: &Tensor3,
    rhop: &Tensor3,
    mu: &Tensor3,
    kx: f64,
) -> FieldVec {
    let w = Wave::from_psi_full(psi, eps, rho, rhop, mu, kx);
    FieldVec {
        e: [w.ex, w.ey, w.ez],
        h: [w.hx, w.hy, w.hz],
    }
}

/// Per-layer transfer matrices, inverses, prefix products, depth bookkeeping.
/// Built once per (λ,θ), reused for all z + both pols.
struct StackFrames {
    /// P_j⁻¹ for each interior layer (local propagation without re-inversion).
    layer_pinv: Vec<Mat4>,
    /// pref[j] = T_{j-1}·…·T_0 (pref[0] = I): ψ at front of layer j =
    /// pref[j]·ψ0. Length n_layers + 1; pref[n] pushes to the exit plane.
    pref: Vec<Mat4>,
    /// Front depth of layer j in nm (layer 0 starts at 0). Length n_layers.
    fronts: Vec<f64>,
    total: f64,
}

fn build_frames(slayers: &[SLayer], thick: &[f64]) -> Option<StackFrames> {
    let n = slayers.len();
    let mut layer_pinv = Vec::with_capacity(n);
    let mut pref = Vec::with_capacity(n + 1);
    pref.push(mat4_identity());
    let mut fronts = Vec::with_capacity(n);
    let mut z = 0.0;
    for (sl, &d) in slayers.iter().zip(thick.iter()) {
        fronts.push(z);
        z += d;
        layer_pinv.push(mat4_inv(&sl.p)?);
        // T_j = P·diag(Q)·P⁻¹ == mat4_similarity_diag (same helper as TM path)
        let tj = mat4_similarity_diag(&sl.p, &sl.q)?;
        let acc = mat4_mul(&tj, &pref[pref.len() - 1]);
        pref.push(acc);
    }
    Some(StackFrames { layer_pinv, pref, fronts, total: z })
}

/// ψ-column times eigen-coefficients: ψ = P · c.
fn apply_p(p: &Mat4, coeff: &[C; 4]) -> [C; 4] {
    let mut out = [czero(); 4];
    for i in 0..4 {
        let mut s = czero();
        for k in 0..4 {
            s += p[i][k] * coeff[k];
        }
        out[i] = s;
    }
    out
}

/// Local propagation inside one layer/half-space over distance d (nm):
/// c_k(d) = c_k(0)·exp(i·k0·q_raw[k]·d), then ψ = P·c.
/// Negative d back-propagates (entry side); diagonal phases handle forward
/// and backward modes symmetrically into the correct standing wave.
fn propagate_local(sl: &SLayer, coeff_front: &[C; 4], k0: f64, d: f64) -> [C; 4] {
    let f = c(0.0, k0 * d);
    let mut cd = [czero(); 4];
    for k in 0..4 {
        cd[k] = coeff_front[k] * (f * sl.q_raw[k]).exp();
    }
    apply_p(&sl.p, &cd)
}

/// Which region `z` falls in. Half-spaces reuse the same propagate_local
/// machinery (no special-case plane-wave code).
enum Region {
    /// d = −z > 0 distance back from the entry plane.
    Entry(f64),
    /// (layer index, d = z − fronts[j]).
    Layer(usize, f64),
    /// d = z − total ≥ 0 distance past the last interface.
    Exit(f64),
}

fn locate(frames: &StackFrames, z: f64) -> Region {
    if z < 0.0 {
        return Region::Entry(-z);
    }
    if z >= frames.total {
        return Region::Exit(z - frames.total);
    }
    // linear scan is fine (n_layers small); binary search if profiling says so
    let mut j = 0;
    while j + 1 < frames.fronts.len() && frames.fronts[j + 1] <= z {
        j += 1;
    }
    Region::Layer(j, z - frames.fronts[j])
}

/// Time-averaged absorbed power density at a field point, in the SAME
/// (Berreman-unit) scale as the flux ½·Re(Ex·Hy* − Ey·Hx*) — so that
/// ∫A·dz + F_out == F_in exactly (G8e), with no fitted prefactor:
///   A = ½·(ω/c0)·Σ_ij Im(ε_ij)·Re(E_i·conj(E_j)),  ω = 2πc0/λ.
/// The 1/c0 is ε0·Z0: physical A_SI = ½·ω·ε0·ΣIm(ε)|E|², and Berreman fields
/// carry H impedance-scaled (H_berr = Z0·H_SI), so fluxes and densities in
/// code units are Z0× their SI values: A_code = Z0·A_SI = ½·ω·(ε0·Z0)·… =
/// ½·(ω/c0)·…. (DRAFT CORRECTION at implementation: the plan sketch had plain
/// ½·ω·s, off by c0 — G8e's energy balance caught it: 2.86e8 vs 1.0. The
/// corrected prefactor balances to 1 ± 1e-9, which PINS it — a wrong scale
/// cannot accidentally satisfy an independent R/T conservation identity.)
/// For diagonal lossless eps this is exactly 0. Off-diagonal Im parts are
/// included via the full double sum (correct for gyrotropic media).
pub fn absorption_density(f: &FieldVec, eps: &Tensor3, wl_nm: f64) -> f64 {
    const C0: f64 = 2.998e8;
    let omega = 2.0 * std::f64::consts::PI * C0 / (wl_nm * 1e-9);
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += eps[i][j].im * (f.e[i] * f.e[j].conj()).re;
        }
    }
    0.5 * (omega / C0) * s
}

/// Flat-buffer index for one (pol, E/H, depth, component) complex value,
/// pointing at its REAL part (imaginary follows at +1):
///   idx = (((pol*2 + eh) * n_z + iz) * 3 + comp) * 2,
/// pol: p = 0, s = 1; eh: E = 0, H = 1. Shared by fields.rs (Rust side
/// layout test below) and pybind.rs fields_sweep scatter — the rule lives
/// HERE so the two sides cannot drift.
#[inline]
pub fn field_buf_index(pol: usize, eh: usize, iz: usize, comp: usize, n_z: usize) -> usize {
    (((pol * 2 + eh) * n_z + iz) * 3 + comp) * 2
}

/// Compute fields at `z_nm` (any order; negatives = entry side) for both
/// canonical input polarizations. Returns None if any inversion fails.
///
/// Propagation is ALWAYS transfer-based (same P/Q as the TM path), even when
/// the far field was solved by SM — matches pyllama, whose
/// `get_in_plane_fields` likewise propagates eigenmodes directly. `method` is
/// accepted for API symmetry only (it selects the far-field engine whose
/// Jones the field asymptotes must match — asserted in G8c).
///
/// Periodic stacks: the caller expands the cell (layers × periods) and calls
/// this as a plain stack — z spans 0..periods·cell thickness. The Python
/// `fields()` does the expansion; no powering here (prefix products need the
/// full unfolded sequence anyway).
pub fn fields_at_depths(
    geom: &Geometry,
    layers: &[LayerSpec],
    z_nm: &[f64],
    _method: Method,
) -> Option<Vec<FieldPoint>> {
    let head = build_head(geom)?;
    let (k0, kx) = (head.k0, head.kx);
    let (entry, exit) = (head.entry, head.exit_sl);
    let slayers = build_slayers(layers, kx, k0);
    let thick: Vec<f64> = layers.iter().map(|ls| ls.thickness_nm).collect();
    let frames = build_frames(&slayers, &thick)?;
    let tm = transfer_matrix(&entry, &slayers, &exit)?;

    // unpacking tensors: entry n²·I; exit per ExitSpec; interior per LayerSpec
    // (dispersive layers automatically correct — same &[LayerSpec] as solve).
    let n2 = geom.n_entry * geom.n_entry;
    let entry_eps = diag_eps(n2, n2, n2);
    let exit_eps: Tensor3 = match &geom.exit {
        ExitSpec::Isotropic(n) => {
            let m2 = *n * *n;
            diag_eps(m2, m2, m2)
        }
        ExitSpec::Anisotropic { eps, .. } => *eps,
    };
    let mut out = Vec::with_capacity(z_nm.len());
    for &z in z_nm {
        // hoisted: one locate per z (not per pol).
        let region = locate(&frames, z);
        let eps_here: &Tensor3 = match &region {
            Region::Entry(_) => &entry_eps,
            Region::Layer(j, _) => &layers[*j].eps,
            Region::Exit(_) => &exit_eps,
        };
        // Full-tensor unpack weights where the sample point carries them
        // (None -> simple unpack, bit-identical history). Entry is always
        // isotropic; exit consults the spec (MO exit allowed since P1).
        let full_here: Option<(&Tensor3, &Tensor3, &Tensor3)> = match &region {
            Region::Entry(_) => None,
            Region::Layer(j, _) => layers[*j]
                .full
                .as_ref()
                .map(|f| (&f.0, &f.1, &f.2)),
            Region::Exit(_) => match &geom.exit {
                ExitSpec::Anisotropic { full: Some(f), .. } => Some((&f.0, &f.1, &f.2)),
                _ => None,
            },
        };
        let mut per_input =
            [FieldVec { e: [czero(); 3], h: [czero(); 3] }; 2];
        for (ji, incident) in [[cone(), czero()], [czero(), cone()]]
            .iter()
            .enumerate()
        {
            let a = entry_amplitudes(&tm, *incident)?;
            let psi0v = apply_p(&entry.p, &a); // entry-plane state
            let psi = match &region {
                Region::Entry(d) => propagate_local(&entry, &a, k0, -d),
                Region::Layer(j, d) => {
                    // ψ_front = pref[j]·ψ0, then local coeffs via cached P⁻¹
                    let front = apply_p(&frames.pref[*j], &psi0v);
                    let cf = apply_p(&frames.layer_pinv[*j], &front);
                    propagate_local(&slayers[*j], &cf, k0, *d)
                }
                Region::Exit(d) => {
                    // b = tm·a forward pair; b[2..] = 0 IS the outgoing
                    // condition (asserted ~1e-16 in tests) — zero, don't
                    // propagate fp residuals (a 1e-16 backward mode could be
                    // evanescent-growing in +z).
                    let b2 = exit_amplitudes(&tm, *incident)?;
                    let b = [b2[0], b2[1], czero(), czero()];
                    propagate_local(&exit, &b, k0, *d)
                }
            };
            per_input[ji] = match full_here {
                Some((r, rp, m)) => psi_to_eh_full(&psi, eps_here, r, rp, m, kx),
                None => psi_to_eh(&psi, eps_here, kx),
            };
        }
        out.push(FieldPoint { z_nm: z, per_input });
    }
    Some(out)
}

#[cfg(test)]
mod depth_tests {
    use super::*;
    use crate::transfer::{solve_stack, Geometry};

    /// G8a (explicit, on a real stack): entry_amplitudes back pair equals the
    /// solved Jones reflection column. (P1's test showed the same identity on
    /// a synthetic matrix vs fresnel_from_transfer; this closes it to solve.)
    #[test]
    fn entry_amplitudes_match_solved_jones() {
        let eps = diag_eps(c(2.25, 0.0), c(2.89, 0.0), c(2.25, 0.0));
        let layers = [LayerSpec {
            eps,
            thickness_nm: 240.0,
            full: None,
            front_roughness: (0, 0.0),
        }];
        let geom = Geometry::isotropic(
            550.0,
            25.0_f64.to_radians(),
            c(1.0, 0.0),
            c(1.5, 0.0),
            (0, 0.0),
        );
        let s = solve_stack(&geom, &layers, Method::Scattering).unwrap();
        // rebuild the transfer matrix the Flux arm uses
        let head = build_head(&geom).unwrap();
        let sl = build_slayers(&layers, head.kx, head.k0);
        let tm = transfer_matrix(&head.entry, &sl, &head.exit_sl).unwrap();
        let ap = entry_amplitudes(&tm, [cone(), czero()]).unwrap();
        let as_ = entry_amplitudes(&tm, [czero(), cone()]).unwrap();
        assert!((ap[2] - s.jones.refl[0][0]).norm() < 1e-15);
        assert!((ap[3] - s.jones.refl[1][0]).norm() < 1e-15);
        assert!((as_[2] - s.jones.refl[0][1]).norm() < 1e-15);
        assert!((as_[3] - s.jones.refl[1][1]).norm() < 1e-15);
    }

    /// psi_to_eh is Wave::from_psi by construction — pin the mapping once
    /// (field order E=[Ex,Ey,Ez], H=[Hx,Hy,Hz]) on an anisotropic tensor.
    #[test]
    fn psi_unpack_order() {
        let eps = diag_eps(c(2.25, 0.1), c(2.89, 0.0), c(1.0, 0.5));
        let psi = [c(1.0, 0.2), c(0.3, -0.4), c(-0.5, 0.6), c(0.7, 0.8)];
        let f = psi_to_eh(&psi, &eps, 0.35);
        let w = Wave::from_psi(&psi, &eps, 0.35);
        assert!((f.e[0] - w.ex).norm() == 0.0);
        assert!((f.e[1] - w.ey).norm() == 0.0);
        assert!((f.e[2] - w.ez).norm() == 0.0);
        assert!((f.h[0] - w.hx).norm() == 0.0);
        assert!((f.h[1] - w.hy).norm() == 0.0);
        assert!((f.h[2] - w.hz).norm() == 0.0);
    }

    /// Buffer layout: two writes never collide, re/im adjacent, count exact.
    #[test]
    fn buf_layout_bijective() {
        let n_z = 7;
        let mut seen = std::collections::HashSet::new();
        for pol in 0..2 {
            for eh in 0..2 {
                for iz in 0..n_z {
                    for comp in 0..3 {
                        let idx = field_buf_index(pol, eh, iz, comp, n_z);
                        assert!(idx + 1 < 2 * 2 * n_z * 3 * 2);
                        assert!(seen.insert(idx), "collision at {idx}");
                    }
                }
            }
        }
        assert_eq!(seen.len(), 2 * 2 * n_z * 3);
    }

    /// absorption_density on analytic values: diagonal eps, E along x.
    #[test]
    fn absorption_analytic() {
        const C0: f64 = 2.998e8;
        let wl = 600.0;
        let omega = 2.0 * std::f64::consts::PI * C0 / (wl * 1e-9);
        let eps = diag_eps(c(2.25, 0.5), c(2.25, 0.0), c(2.25, 0.0));
        let f = FieldVec {
            e: [c(1.0, 0.0), c(0.0, 1.0), czero()],
            h: [czero(); 3],
        };
        // only εxx carries Im (0.5); Ey drops out: A = ½·(ω/c0)·0.5·|Ex|²
        let expect = 0.5 * (omega / C0) * 0.5 * 1.0;
        assert!((absorption_density(&f, &eps, wl) - expect).abs() < 1e-9);
        // lossless diagonal → exactly 0
        let eps0 = diag_eps(c(2.25, 0.0), c(2.25, 0.0), c(2.25, 0.0));
        assert!(absorption_density(&f, &eps0, wl) == 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::berreman::{diag_eps, halfspace_waves, layer_waves_simple};
    use crate::cmatrix::{c, cone, czero, mat4_identity, mat4_inv, mat4_mul};
    use crate::transfer::{fresnel_from_transfer, SLayer};

    /// Bridge to the validated path: with incident [1,0], a2/a3 must equal
    /// (r_pp, r_ps) from `fresnel_from_transfer` on the same matrix.
    #[test]
    fn entry_amplitudes_match_fresnel_extraction() {
        let entry = SLayer::half_space(&halfspace_waves(c(1.0, 0.0), 0.35));
        let exit = SLayer::half_space(&halfspace_waves(c(2.25, 0.0), 0.35));
        let lw = layer_waves_simple(
            &diag_eps(c(2.25, 0.0), c(2.89, 0.0), c(2.25, 0.0)),
            0.35,
        );
        // single-layer transfer matrix: exit.P⁻¹ · P·Q·P⁻¹ · entry.P
        let pinv = mat4_inv(&lw.p).unwrap();
        let mut pd = lw.p;
        // scale columns by exp(i·k0·q·d)-like phases (any nonsingular Q works
        // for this algebraic identity check)
        let phases = [c(0.3, 0.9), c(-0.2, 0.4), c(0.1, -0.7), c(0.0, 0.2)];
        for j in 0..4 {
            for i in 0..4 {
                pd[i][j] *= phases[j];
            }
        }
        let tl = mat4_mul(&pd, &pinv);
        let e_pinv = mat4_inv(&exit.p).unwrap();
        let tm = mat4_mul(&e_pinv, &mat4_mul(&tl, &entry.p));
        let j = fresnel_from_transfer(&tm);
        let a = entry_amplitudes(&tm, [cone(), czero()]).unwrap();
        assert!((a[2] - j.refl[0][0]).norm() < 1e-12); // r_pp
        assert!((a[3] - j.refl[1][0]).norm() < 1e-12); // r_ps
        // exit back pair vanishes (outgoing condition)
        let b_full = {
            let mut b = [czero(); 4];
            for r in 0..4 {
                b[r] = tm[r][0] * a[0] + tm[r][1] * a[1] + tm[r][2] * a[2] + tm[r][3] * a[3];
            }
            b
        };
        assert!(b_full[2].norm() < 1e-12);
        assert!(b_full[3].norm() < 1e-12);
        // exit_amplitudes agrees with the manual product
        let e = exit_amplitudes(&tm, [cone(), czero()]).unwrap();
        assert!((e[0] - b_full[0]).norm() == 0.0);
        assert!((e[1] - b_full[1]).norm() == 0.0);
        let _ = mat4_identity();
    }


    #[test]
    fn singular_transfer_matrix_returns_none() {
        let mut tm = mat4_identity();
        // zero the back block -> d = 0
        tm[2][2] = czero();
        tm[2][3] = czero();
        tm[3][2] = czero();
        tm[3][3] = czero();
        assert!(entry_amplitudes(&tm, [cone(), czero()]).is_none());
        assert!(exit_amplitudes(&tm, [cone(), czero()]).is_none());
    }
}
