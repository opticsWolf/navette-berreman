// SPDX-License-Identifier: LGPL-3.0-or-later
//! Real 3x3 rotations applied to complex 3x3 tensors: R·ε·Rᵀ.
//! Port of pyllama `rot_mat` / `dielectric_tensor` euler+quaternion helpers.
//!
//! Convention audit (live `BerreMueller`, Phase 6 — read before touching):
//!  * application is `R·ε·Rᵀ` (`multi_dot((R, tensor, R.transpose()))`,
//!    `pyllama.py:rotate_tensor`; same in `rotate_2D_tensor` via einsum).
//!  * euler is `R = Rx·Ry·Rz` (`np.dot(r_matrix_x, np.dot(r_matrix_y,
//!    r_matrix_z))`, `dielectric_tensor.py:323`; docstring "z, then y, then
//!    x" = application order to vectors). NOT `Rz·Ry·Rx` — the plan's first
//!    draft expanded the wrong product; G12a caught it (see §6.2 note).
//!    Implemented as an explicit `mat3_mul` chain so the order is visible.
//!  * quaternion is scalar-LAST `[x, y, z, w]` at the live call site
//!    (`Rotation.from_quat`, "note that scalar is last",
//!    `pyllama.py:rotate_tensor`). Our Rust fn takes `(w, x, y, z)`
//!    scalar-first with the identical matrix formula; the Python wrapper
//!    `rot_quat(w, x, y, z, eps)` keeps scalar-first with named args, and
//!    the G12a test maps the live call to `[x, y, z, w]`.
//!  * zero axis: live `rot_mat` RAISES; our Python wrapper raises
//!    `ValueError` to match (Rust keeps a defensive identity fallback for
//!    the unreachable path — pinned in the unit test, not assumed).

use crate::berreman::Tensor3;
use crate::cmatrix::c;

pub type Mat3R = [[f64; 3]; 3];

fn ident3() -> Mat3R {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn mat3_mul(a: &Mat3R, b: &Mat3R) -> Mat3R {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}

/// Rodrigues rotation about `axis` by `theta` (== pyllama `rot_mat`).
/// R = I + sinθ·K + (1−cosθ)·K², K = skew(unit axis). Zero axis => identity
/// (defensive; the Python wrapper rejects it with ValueError like live).
pub fn axis_angle(axis: [f64; 3], theta: f64) -> Mat3R {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if n < 1e-300 {
        return ident3();
    }
    let (x, y, z) = (axis[0] / n, axis[1] / n, axis[2] / n);
    let (s, cc) = (theta.sin(), theta.cos());
    let k: Mat3R = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]];
    let k2 = mat3_mul(&k, &k);
    let mut r = ident3();
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] += s * k[i][j] + (1.0 - cc) * k2[i][j];
        }
    }
    r
}

/// Euler rotation == `dielectric_tensor.euler_rotation_matrix(rx, ry, rz)`:
/// R = Rx·Ry·Rz (live line 323; "z, then y, then x" application order).
/// Built as a visible mul chain — do NOT "simplify" to a hand expansion
/// without re-running G12a (a Rz·Ry·Rx expansion stood here in the draft).
pub fn euler(rx: f64, ry: f64, rz: f64) -> Mat3R {
    let (sx, cx) = (rx.sin(), rx.cos());
    let (sy, cy) = (ry.sin(), ry.cos());
    let (sz, cz) = (rz.sin(), rz.cos());
    let rxm: Mat3R = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    let rym: Mat3R = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rzm: Mat3R = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    mat3_mul(&rxm, &mat3_mul(&rym, &rzm))
}

/// Unit quaternion (w, x, y, z) scalar-first -> rotation matrix.
/// Formula identical to scipy `Rotation.as_matrix` for `[x, y, z, w]`.
/// Non-unit input normalized; zero norm => identity (defensive; the Python
/// wrapper rejects it like the axis case).
pub fn quaternion(w: f64, x: f64, y: f64, z: f64) -> Mat3R {
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if n < 1e-300 {
        return ident3();
    }
    let (w, x, y, z) = (w / n, x / n, y / n, z / n);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Transpose (= inverse for rotations).
pub fn transpose3(r: &Mat3R) -> Mat3R {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = r[j][i];
        }
    }
    o
}

/// Apply: out[i][j] = Σ_kl R[i][k] eps[k][l] R[j][l] (== R·ε·Rᵀ, live).
pub fn apply_rot(r: &Mat3R, eps: &Tensor3) -> Tensor3 {
    let mut out = [[c(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = c(0.0, 0.0);
            for k in 0..3 {
                for l in 0..3 {
                    s += c(r[i][k], 0.0) * eps[k][l] * c(r[j][l], 0.0);
                }
            }
            out[i][j] = s;
        }
    }
    out
}

/// Batched apply (one R over n tensors) — used by evaluate_tensor's
/// per-wavelength orientation and by the twist kernel below.
pub fn apply_rot_batch(r: &Mat3R, eps: &[Tensor3]) -> Vec<Tensor3> {
    eps.iter().map(|t| apply_rot(r, t)).collect()
}

/// Twist-grid sampling schedule (mirrors B44 InhomogeneousLayer).
/// Midpoint = the B44 default sampling (tensor evaluated at slice midpoints
/// `(k+½)·d/div` even though getSlices returns endpoint edges); endpoint =
/// explicit edge-matched grid `frac = k/(n−1)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TwistGrid {
    Midpoint,
    Endpoint,
}

/// Twist-schedule kernel (B44 `TwistedMaterial` / pyllama cholesteric):
/// slice k of `n_slices` carries `axis_angle(z, twist_rad · frac) · eps · (…)
/// ᵀ` with `frac = (k+½)/n` (Midpoint) or `frac = k/max(n−1,1)` (Endpoint).
/// Identical math to the former Python loop (same `rot_z` code path), now
/// shipped from Rust; the Python helper marshals the output into Layers.
pub fn twisted_tensors(
    eps: &Tensor3,
    twist_rad: f64,
    n_slices: usize,
    grid: TwistGrid,
) -> Result<Vec<Tensor3>, String> {
    if n_slices == 0 {
        return Err("n_slices must be >= 1".to_string());
    }
    if !twist_rad.is_finite() {
        return Err(format!("twist_rad must be finite, got {twist_rad}"));
    }
    let out: Vec<Tensor3> = (0..n_slices)
        .map(|k| {
            let frac = match grid {
                TwistGrid::Midpoint => (k as f64 + 0.5) / n_slices as f64,
                TwistGrid::Endpoint => k as f64 / (n_slices.max(2) - 1) as f64,
            };
            let r = axis_angle([0.0, 0.0, 1.0], twist_rad * frac);
            apply_rot(&r, eps)
        })
        .collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det3(r: &Mat3R) -> f64 {
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    }

    #[test]
    fn rotation_group_properties() {
        // R·Rᵀ = I and det = +1 for all three constructors (proper rotations)
        let cases: Vec<Mat3R> = vec![
            axis_angle([1.0, 2.0, 3.0], 0.7),
            euler(0.3, -0.5, 1.1),
            quaternion(0.5, 0.5, 0.5, 0.5),
        ];
        for r in &cases {
            let rt = transpose3(r);
            let should_be_i = mat3_mul(r, &rt);
            for i in 0..3 {
                for j in 0..3 {
                    let e = if i == j { 1.0 } else { 0.0 };
                    assert!((should_be_i[i][j] - e).abs() < 1e-14);
                }
            }
            assert!((det3(r) - 1.0).abs() < 1e-14);
        }
        // composition: z-rot then x-rot == euler(rx, 0, rz) since R = Rx·Ry·Rz
        let a = axis_angle([0.0, 0.0, 1.0], 0.4);
        let b = axis_angle([1.0, 0.0, 0.0], 0.9);
        let comp = mat3_mul(&b, &a); // apply a first, then b
        let eu = euler(0.9, 0.0, 0.4);
        for i in 0..3 {
            for j in 0..3 {
                assert!((comp[i][j] - eu[i][j]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn degenerate_inputs_fall_back_to_identity() {
        // Unreachable via the Python wrapper (raises like live); pinned here
        // so the fallback never silently changes.
        let id = ident3();
        for r in [axis_angle([0.0, 0.0, 0.0], 0.7), quaternion(0.0, 0.0, 0.0, 0.0)] {
            for i in 0..3 {
                for j in 0..3 {
                    assert!((r[i][j] - id[i][j]).abs() == 0.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod twist_tests {
    use super::*;

    fn diag(v: f64) -> Tensor3 {
        [[c(v, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
         [c(0.0, 0.0), c(v, 0.0), c(0.0, 0.0)],
         [c(0.0, 0.0), c(0.0, 0.0), c(v, 0.0)]]
    }

    /// Schedule fractions + equivalence with the direct per-slice rot path.
    #[test]
    fn twisted_schedule_matches_rot_path() {
        let e0 = [[c(2.89, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
                  [c(0.0, 0.0), c(2.25, 0.0), c(0.0, 0.0)],
                  [c(0.0, 0.0), c(0.0, 0.0), c(2.25, 0.0)]];
        let twist = 2.0 * std::f64::consts::PI * 5.0;
        let g = twisted_tensors(&e0, twist, 240, TwistGrid::Midpoint)
            .expect("twisted");
        assert_eq!(g.len(), 240);
        // slice k must equal apply_rot(axis_z, twist*(k+1/2)/n) directly
        for k in [0usize, 7, 119, 239] {
            let frac = (k as f64 + 0.5) / 240.0;
            let want = apply_rot(&axis_angle([0.0, 0.0, 1.0], twist * frac), &e0);
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(g[k][i][j], want[i][j], "slice {k} mismatch");
                }
            }
        }
        // z-rot keeps εzz and the in-plane block structure
        assert_eq!(g[0][2][2], e0[2][2]);
        assert_eq!(g[0][0][2], c(0.0, 0.0));
        // endpoint grid: frac = k/(n-1), first slice unrotated, last == full twist
        let ge = twisted_tensors(&e0, 0.7, 5, TwistGrid::Endpoint).expect("twisted");
        assert_eq!(ge[0][0][0], e0[0][0]);
        let last = apply_rot(&axis_angle([0.0, 0.0, 1.0], 0.7), &e0);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(ge[4][i][j], last[i][j]);
            }
        }
        // n=1: single unrotated slice (0/max(1,1) guard)
        let g1 = twisted_tensors(&e0, 0.7, 1, TwistGrid::Endpoint).expect("twisted");
        assert_eq!(g1[0][0][0], e0[0][0]);
        // midpoint n=1: frac = 1/2 (half the twist)
        let g1m = twisted_tensors(&e0, 0.7, 1, TwistGrid::Midpoint).expect("twisted");
        let half = apply_rot(&axis_angle([0.0, 0.0, 1.0], 0.35), &e0);
        assert_eq!(g1m[0][1][0], half[1][0]);
        // validation
        assert!(twisted_tensors(&e0, 0.7, 0, TwistGrid::Midpoint).is_err());
        assert!(twisted_tensors(&e0, f64::NAN, 5, TwistGrid::Midpoint).is_err());
        // batch helper agrees slice-by-slice
        let b = apply_rot_batch(&axis_angle([0.0, 0.0, 1.0], 0.3), &g[..3]);
        for k in 0..3 {
            let want = apply_rot(&axis_angle([0.0, 0.0, 1.0], 0.3), &g[k]);
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(b[k][i][j], want[i][j]);
                }
            }
        }
    }
}
