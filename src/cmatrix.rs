//! Fixed-size complex linear algebra for the 4x4 Berreman problem.
//!
//! Everything here is pure Rust over `num_complex::Complex64` so it can be
//! exercised from a plain Rust test binary with no Python in the loop. The
//! 4x4 eigensolver is the only non-trivial piece: it reproduces the result of
//! `numpy.linalg.eig` for a general (non-Hermitian, complex) 4x4 matrix
//! closely enough that the downstream partial-wave sort is bit-stable.
//!
//! The eigensolver is self-contained (no LAPACK / external eigen crate):
//!   1. characteristic polynomial via Faddeev–LeVerrier (exact for a 4x4),
//!   2. its four roots via Durand–Kerner (Weierstrass) iteration,
//!   3. each root polished with a couple of Newton steps,
//!   4. eigenvectors as the null vector of (A - lambda*I), normalized to unit
//!      Euclidean norm to match numpy's convention.
//!
//! Only the eigenvalue *set* and the per-vector Poynting ratio (a magnitude,
//! hence scale/phase invariant) feed the sort, so an arbitrary eigenvector
//! phase is harmless — verified numerically against pyllama.

use num_complex::Complex64;
use num_complex::ComplexFloat;

pub type C = Complex64;

#[inline(always)]
pub fn c(re: f64, im: f64) -> C {
    Complex64::new(re, im)
}

#[inline(always)]
pub fn czero() -> C {
    Complex64::new(0.0, 0.0)
}

#[inline(always)]
pub fn cone() -> C {
    Complex64::new(1.0, 0.0)
}

/// 4x4 complex matrix, row-major.
pub type Mat4 = [[C; 4]; 4];

#[inline]
pub fn mat4_zero() -> Mat4 {
    [[czero(); 4]; 4]
}

#[inline]
pub fn mat4_identity() -> Mat4 {
    let mut m = mat4_zero();
    for i in 0..4 {
        m[i][i] = cone();
    }
    m
}

#[inline]
pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = mat4_zero();
    for i in 0..4 {
        for k in 0..4 {
            let aik = a[i][k];
            if aik == czero() {
                continue;
            }
            for j in 0..4 {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

/// Gauss–Jordan inverse of a 4x4 complex matrix with partial pivoting.
/// Returns None on (numerical) singularity.
pub fn mat4_inv(a: &Mat4) -> Option<Mat4> {
    let mut m = *a;
    let mut inv = mat4_identity();
    for col in 0..4 {
        // partial pivot on largest magnitude
        let mut piv = col;
        let mut best = m[col][col].norm();
        for r in (col + 1)..4 {
            let v = m[r][col].norm();
            if v > best {
                best = v;
                piv = r;
            }
        }
        // P10: written as !(best >= 1e-300) so a NaN pivot (degenerate /
        // det-singular eig paths can emit NaN or zero columns) also returns
        // None (NaN >= x is false); subnormal/zero behavior unchanged.
        // Honest-NaN backstop for the eig4 zero-fill column.
        if !(best >= 1e-300) {
            return None;
        }
        if piv != col {
            m.swap(col, piv);
            inv.swap(col, piv);
        }
        let d = m[col][col];
        let dinv = d.recip();
        for j in 0..4 {
            m[col][j] *= dinv;
            inv[col][j] *= dinv;
        }
        for r in 0..4 {
            if r == col {
                continue;
            }
            let f = m[r][col];
            if f == czero() {
                continue;
            }
            for j in 0..4 {
                m[r][j] -= f * m[col][j];
                inv[r][j] -= f * inv[col][j];
            }
        }
    }
    Some(inv)
}

/// Product P * diag(q) * P^{-1}. Used to assemble a layer's transfer matrix
/// from eigenvectors (columns of P) and the propagation phases on the diagonal.
pub fn mat4_similarity_diag(p: &Mat4, diag: &[C; 4]) -> Option<Mat4> {
    let pinv = mat4_inv(p)?;
    // PD: scale column j of P by diag[j]
    let mut pd = mat4_zero();
    for i in 0..4 {
        for j in 0..4 {
            pd[i][j] = p[i][j] * diag[j];
        }
    }
    Some(mat4_mul(&pd, &pinv))
}

/// Element-wise add.
#[inline]
pub fn mat4_add(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = mat4_zero();
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

/// Element-wise scale by a complex scalar.
#[inline]
pub fn mat4_scale(a: &Mat4, s: C) -> Mat4 {
    let mut out = mat4_zero();
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][j] * s;
        }
    }
    out
}

/// 1-norm (max column absolute sum); subordinate, so ||AB|| ≤ ||A||·||B||.
#[inline]
pub fn mat4_norm1(a: &Mat4) -> f64 {
    let mut m = 0.0;
    for j in 0..4 {
        let mut s = 0.0;
        for i in 0..4 {
            s += a[i][j].norm();
        }
        if s > m {
            m = s;
        }
    }
    m
}

// ───────────────────────── 2x2 complex helpers ─────────────────────────

pub type Mat2 = [[C; 2]; 2];

#[inline]
pub fn mat2_mul(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

#[inline]
pub fn mat2_inv(a: &Mat2) -> Option<Mat2> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.norm() < 1e-300 {
        return None;
    }
    let di = det.recip();
    Some([[a[1][1] * di, -a[0][1] * di], [-a[1][0] * di, a[0][0] * di]])
}

// ─────────────────────────── eigensolver ───────────────────────────────

/// Characteristic-polynomial coefficients of a 4x4 via Faddeev–LeVerrier.
///
/// Returns `[c0, c1, c2, c3]` for
///   p(λ) = λ^4 + c3 λ^3 + c2 λ^2 + c1 λ + c0   (monic).
fn char_poly4(a: &Mat4) -> [C; 4] {
    // M_1 = A,            c_1 = -tr(M_1)
    // M_2 = A(M_1 + c_1 I), c_2 = -tr(M_2)/2
    // M_3 = A(M_2 + c_2 I), c_3 = -tr(M_3)/3
    //                       c_4 = -tr(A(M_3 + c_3 I))/4
    let tr = |m: &Mat4| m[0][0] + m[1][1] + m[2][2] + m[3][3];
    let add_scaled_iden = |m: &Mat4, s: C| {
        let mut o = *m;
        for i in 0..4 {
            o[i][i] += s;
        }
        o
    };

    let m1 = *a;
    let c1 = -tr(&m1);
    let m2 = mat4_mul(a, &add_scaled_iden(&m1, c1));
    let c2 = -tr(&m2) / c(2.0, 0.0);
    let m3 = mat4_mul(a, &add_scaled_iden(&m2, c2));
    let c3 = -tr(&m3) / c(3.0, 0.0);
    let m4 = mat4_mul(a, &add_scaled_iden(&m3, c3));
    let c4 = -tr(&m4) / c(4.0, 0.0);

    // p(λ) = λ^4 + c1 λ^3 + c2 λ^2 + c3 λ + c4
    [c4, c3, c2, c1]
}

#[inline]
fn poly_eval(coeffs: &[C; 4], x: C) -> C {
    // λ^4 + c3 λ^3 + c2 λ^2 + c1 λ + c0
    let [c0, c1, c2, c3] = *coeffs;
    (((x + c3) * x + c2) * x + c1) * x + c0
}

#[inline]
fn poly_deriv_eval(coeffs: &[C; 4], x: C) -> C {
    let [_c0, c1, c2, c3] = *coeffs;
    // 4λ^3 + 3c3 λ^2 + 2c2 λ + c1
    ((c(4.0, 0.0) * x + c(3.0, 0.0) * c3) * x + c(2.0, 0.0) * c2) * x + c1
}

/// All four roots of the monic quartic via Durand–Kerner, Newton-polished.
fn quartic_roots(coeffs: &[C; 4]) -> [C; 4] {
    // Spread initial guesses around a circle to avoid clustering.
    let seed = c(0.4, 0.9);
    let mut r = [cone(); 4];
    let mut p = cone();
    for i in 0..4 {
        p *= seed;
        r[i] = p;
    }

    for _ in 0..200 {
        let mut max_step = 0.0_f64;
        let snapshot = r;
        for i in 0..4 {
            let num = poly_eval(coeffs, snapshot[i]);
            let mut den = cone();
            for j in 0..4 {
                if j != i {
                    den *= snapshot[i] - snapshot[j];
                }
            }
            if den.norm() < 1e-300 {
                continue;
            }
            let step = num / den;
            r[i] = snapshot[i] - step;
            let s = step.norm();
            if s > max_step {
                max_step = s;
            }
        }
        if max_step < 1e-14 {
            break;
        }
    }

    // Newton polish each root against the original polynomial.
    for i in 0..4 {
        for _ in 0..8 {
            let f = poly_eval(coeffs, r[i]);
            let d = poly_deriv_eval(coeffs, r[i]);
            if d.norm() < 1e-300 {
                break;
            }
            let step = f / d;
            r[i] -= step;
            if step.norm() < 1e-15 {
                break;
            }
        }
    }
    r
}

/// Build up to `want` independent null-space vectors of `m` (already a copy of
/// A - λI), via reduced row echelon. Returns orthonormalized vectors. Handles
/// the diagonalizable degenerate case (geometric multiplicity = `want`): an
/// isotropic layer at any incidence has 2-D eigenspaces, and numpy returns a
/// full basis there, so we must too or P becomes singular.
fn null_space_basis(a: &Mat4, lambda: C, want: usize) -> Vec<[C; 4]> {
    let mut m = *a;
    for i in 0..4 {
        m[i][i] -= lambda;
    }
    // Gauss–Jordan to reduced row echelon, tracking pivot columns.
    let mut pivot_of_row = [usize::MAX; 4];
    let mut col_is_pivot = [false; 4];
    let mut row = 0usize;
    for col in 0..4 {
        if row >= 4 {
            break;
        }
        let mut piv = row;
        let mut best = m[row][col].norm();
        for r in (row + 1)..4 {
            let v = m[r][col].norm();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-9 {
            continue;
        }
        m.swap(row, piv);
        let d = m[row][col].recip();
        for j in 0..4 {
            m[row][j] *= d;
        }
        for r in 0..4 {
            if r != row {
                let f = m[r][col];
                if f != czero() {
                    for j in 0..4 {
                        m[r][j] -= f * m[row][j];
                    }
                }
            }
        }
        pivot_of_row[row] = col;
        col_is_pivot[col] = true;
        row += 1;
    }

    // Each free column yields one null vector.
    let mut basis: Vec<[C; 4]> = Vec::new();
    for free in 0..4 {
        if col_is_pivot[free] {
            continue;
        }
        let mut v = [czero(); 4];
        v[free] = cone();
        for r in 0..4 {
            let pc = pivot_of_row[r];
            if pc == usize::MAX {
                continue;
            }
            v[pc] = -m[r][free];
        }
        basis.push(v);
        if basis.len() >= want {
            break;
        }
    }

    // If we still don't have enough (defective / numerically full rank), fall
    // back to inverse iteration with deflation against what we have.
    while basis.len() < want {
        let mut v = inverse_iteration(a, lambda);
        // deflate
        for b in &basis {
            let mut dot = czero();
            for i in 0..4 {
                dot += b[i].conj() * v[i];
            }
            for i in 0..4 {
                v[i] -= dot * b[i];
            }
        }
        normalize(&mut v);
        basis.push(v);
    }

    // Modified Gram–Schmidt orthonormalization (matches numpy's well-separated
    // unit-norm columns closely enough for the phase-invariant sort).
    let mut ortho: Vec<[C; 4]> = Vec::new();
    for mut v in basis.into_iter() {
        for b in &ortho {
            let mut dot = czero();
            for i in 0..4 {
                dot += b[i].conj() * v[i];
            }
            for i in 0..4 {
                v[i] -= dot * b[i];
            }
        }
        normalize(&mut v);
        ortho.push(v);
    }
    ortho
}

/// Eigenvalues of a 4x4 Hermitian matrix via cyclic Jacobi rotations
/// (values only). Degenerate-safe — unlike polynomial methods, where multiple
/// roots are ill-conditioned (a quadruple root's coefficient fuzz gives complex
/// pairs with imag ~1e-4; caught by G11b's ideal depolarizer during P5).
/// Input is symmetrized first ((H+H†)/2) so exact-Hermitian structure holds to
/// the last ulp; diagonal imaginary dust (~1e-17) is discarded via `.re`.
/// Returns UNSORTED eigenvalues; callers sort.
pub fn eig_hermitian4(h: &Mat4) -> [f64; 4] {
    // symmetrize: exact Hermitian to fp precision
    let mut a = mat4_zero();
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = (h[i][j] + h[j][i].conj()) * 0.5;
        }
    }
    // cyclic sweeps over the 6 off-diagonal pairs; quadratic convergence
    for _ in 0..50 {
        let mut off = 0.0;
        for p in 0..4 {
            for q in (p + 1)..4 {
                off += a[p][q].norm_sqr();
            }
        }
        if off < 1e-28 {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                let apq = a[p][q];
                let nrm = apq.norm();
                if nrm < 1e-300 {
                    continue;
                }
                // Stable Jacobi angle (Golub & Van Loan §8.5):
                // τ = (aqq − app)/(2|apq|), t = sign(τ)/(|τ| + √(τ²+1)),
                // J = [[c, s],[−s̄, c]] with c = 1/√(1+t²), s = c·t·apq/|apq|
                // annihilates the pq element under H' = J†·H·J.
                let app = a[p][p].re;
                let aqq = a[q][q].re;
                let tau = (aqq - app) / (2.0 * nrm);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cc = 1.0 / (1.0 + t * t).sqrt();
                let ss = apq * (cc * t / nrm);
                // 2x2 block H' = J†·(H·J), direct complex multiply (no
                // closed-form sign traps): J = [[cc, ss],[−s̄, cc]].
                let b00 = a[p][p];
                let b01 = a[p][q];
                let b10 = a[q][p];
                let b11 = a[q][q];
                let j00 = c(cc, 0.0);
                let j01 = ss;
                let j10 = -ss.conj();
                let j11 = c(cc, 0.0);
                let h00 = b00 * j00 + b01 * j10;
                let h01 = b00 * j01 + b01 * j11;
                let h10 = b10 * j00 + b11 * j10;
                let h11 = b10 * j01 + b11 * j11;
                a[p][p] = j00.conj() * h00 + j10.conj() * h10;
                a[p][q] = j00.conj() * h01 + j10.conj() * h11;
                a[q][p] = j01.conj() * h00 + j11.conj() * h10;
                a[q][q] = j01.conj() * h01 + j11.conj() * h11;
                // off-plane rows/cols: right-multiply row k by J,
                // then restore Hermiticity by explicit conjugation
                for k in 0..4 {
                    if k == p || k == q {
                        continue;
                    }
                    let xkp = a[k][p];
                    let xkq = a[k][q];
                    a[k][p] = xkp * j00 + xkq * j10;
                    a[k][q] = xkp * j01 + xkq * j11;
                }
                for k in 0..4 {
                    if k == p || k == q {
                        continue;
                    }
                    a[p][k] = a[k][p].conj();
                    a[q][k] = a[k][q].conj();
                }
            }
        }
    }
    [a[0][0].re, a[1][1].re, a[2][2].re, a[3][3].re]
}

/// General complex 4x4 eigendecomposition.
///
/// Returns `(eigvals, eigvecs)` where `eigvecs[:, k]` (column k) is the
/// unit-norm eigenvector for `eigvals[k]`. Repeated eigenvalues are clustered
/// and given an orthonormal eigenspace basis so that P stays invertible.
pub fn eig4(a: &Mat4) -> ([C; 4], Mat4) {
    let coeffs = char_poly4(a);
    let mut vals = quartic_roots(&coeffs);

    // Cluster eigenvalues by proximity.
    let mut assigned = [false; 4];
    let mut vecs = mat4_zero();
    for i in 0..4 {
        if assigned[i] {
            continue;
        }
        // gather this cluster
        let mut members = vec![i];
        for j in (i + 1)..4 {
            if !assigned[j] {
                let scale = 1.0 + vals[i].norm();
                if (vals[i] - vals[j]).norm() <= 1e-6 * scale {
                    members.push(j);
                }
            }
        }
        let mult = members.len();
        // representative eigenvalue = cluster mean (more accurate kernel)
        let mut mean = czero();
        for &k in &members {
            mean += vals[k];
        }
        mean /= c(mult as f64, 0.0);
        // SINGLETONS (mult == 1): original path verbatim — quartic roots of
        // isolated eigenvalues are accurate to ~1e-15; substituting the
        // quotient would only add noise. Bit-identity of all non-degenerate
        // solves is preserved BY CONSTRUCTION (this branch is untouched).
        if mult == 1 {
            let k = members[0];
            let basis = null_space_basis(a, mean, 1);
            let v = basis.into_iter().next().unwrap_or_else(|| {
                let mut vv = inverse_iteration(a, vals[k]);
                normalize(&mut vv);
                vv
            });
            for r in 0..4 {
                vecs[r][k] = v[r];
            }
            assigned[k] = true;
            continue;
        }
        let basis = null_space_basis(a, mean, mult);
        // P10 HARDENING (degenerate-cluster repair, mult >= 2 only):
        // null_space_basis can return garbage when the cluster width sits
        // below quartic-root noise (~1e-8·scale): the mean is then off the
        // true spectrum and a noise pivot slips past the 1e-9 RREF threshold
        // (observed residual 2.2 on isotropic-full kappa = 0 → silent O(1)
        // Jones errors downstream; same landmine on the simple path for weak
        // birefringence 1e-12 < dn < 1e-6). Verify candidates by Rayleigh
        // residual (root-noise-immune); repair failures with shift-invert
        // from member-orthogonalized starts; SUBSTITUTE the Rayleigh
        // quotient for the quartic root (root noise would otherwise cap
        // degenerate-case phases at ~1e-7 in Jones; the quotient recovers
        // ~1e-10 — residual floor of shift-invert near a double eigenvalue,
        // linear in k0·d; Newton polish was tried and REVERTED: p ≈ p' ≈ 0
        // near multiple roots makes the step 0/0-dominated (observed 5e-9
        // wander). A Vieta-exact q for the double-double case is a
        // documented future optimization, not implemented.) TRUE clusters
        // (width <= 1e-9·scale: one invariant subspace) are
        // MGS-orthonormalized (same space ⇒ residuals kept); LOOSE clusters
        // (distinct-but-close) keep per-member vectors unmixed. Persistent
        // failures zero-fill the column: P singular → mat4_inv → None.
        let width = members
            .iter()
            .map(|&k| (vals[k] - mean).norm())
            .fold(0.0_f64, f64::max);
        let tight = width <= 1e-9 * (1.0 + mean.norm());
        let mut got: Vec<[C; 4]> = Vec::with_capacity(mult);
        for (bi, &k) in members.iter().enumerate() {
            let target = if tight { mean } else { vals[k] };
            let mut v = basis.get(bi).copied().unwrap_or([czero(); 4]);
            if bad_rayleigh_residual(a, &v) {
                // seeded start, orthogonalized against accepted members:
                // shift-invert preserves direction inside a true multiple,
                // so this yields an orthonormal set by construction (no
                // mixing cost); for distinct members the v1-component
                // survives generically and iteration still converges to it.
                let s = k * 2 + bi;
                let mut start = [czero(); 4];
                start[s % 4] = cone();
                start[(s + 1) % 4] = c(0.37, -0.11);
                for g in &got {
                    let mut dot = czero();
                    for r in 0..4 {
                        dot += g[r].conj() * start[r];
                    }
                    for r in 0..4 {
                        start[r] -= dot * g[r];
                    }
                }
                normalize(&mut start);
                v = inverse_iteration_from(a, target, start);
            }
            got.push(v);
        }
        if tight {
            for i in 0..got.len() {
                for j in 0..i {
                    let gj = got[j]; // [C; 4] is Copy: no aliasing
                    let mut dot = czero();
                    for r in 0..4 {
                        dot += gj[r].conj() * got[i][r];
                    }
                    for r in 0..4 {
                        got[i][r] -= dot * gj[r];
                    }
                }
                normalize(&mut got[i]);
            }
        }
        for (bi, &k) in members.iter().enumerate() {
            let v = got[bi];
            if bad_rayleigh_residual(a, &v) {
                for r in 0..4 {
                    vecs[r][k] = czero();
                }
                // vals[k] keeps the quartic root; singular P → None anyway.
            } else {
                // Rayleigh-quotient-substituted eigenvalue (degenerate floor
                // ~1e-10 in Jones, linear in k0·d — see newton_polish note).
                let qk = rayleigh_quotient(a, &v);
                for r in 0..4 {
                    vecs[r][k] = v[r];
                }
                vals[k] = qk;
            }
            assigned[k] = true;
        }
    }
    (vals, vecs)
}

/// Newton polish (P10): TRIED AND REVERTED — kept as documentation.
/// Near a multiple root p ≈ p' ≈ 0 makes the Newton step 0/0-dominated by
/// rounding: observed 5e-9 wander from a 1e-11 start (worse than the
/// Rayleigh quotient it was meant to improve). Do NOT re-enable without a
/// defective/derivative-guarded variant. The principled future fix for the
/// ~1e-10 degenerate floor is a Vieta-exact q for the double-double case
/// (s = tr/2, p = (c2−s²)/2 ⇒ λ± = (s±√(s²−4p))/2, well-conditioned
/// sqrt) — not implemented (risk/benefit poor at current gate margins).
#[allow(dead_code)]
fn newton_polish(coeffs: &[C; 4], mut z: C) -> C {
    for _ in 0..100 {
        // Horner for p and p' together.
        // p(λ) = ((((λ + c3))λ + c2)λ + c1)λ + c0
        let c3 = coeffs[3];
        let c2 = coeffs[2];
        let c1 = coeffs[1];
        let c0 = coeffs[0];
        let p = ((((z + c3) * z) + c2) * z + c1) * z + c0;
        // p'(λ) = 4λ^3 + 3c3 λ^2 + 2c2 λ + c1
        let dp = (((c(4.0, 0.0) * z) + c(3.0, 0.0) * c3) * z
            + c(2.0, 0.0) * c2)
            * z
            + c1;
        if dp.norm() < 1e-300 {
            break;
        }
        let step = p / dp;
        z -= step;
        if step.norm() <= 1e-15 * (1.0 + z.norm()) {
            break;
        }
    }
    z
}

/// Rayleigh quotient (v*·A·v)/(v*·v): eigenvalue estimate whose error is
/// ~residual (not ~root-noise) — the cluster arm substitutes it for the
/// quartic root (P10: root noise ~1e-8 on double roots would otherwise cap
/// degenerate-case phases at ~1e-7; the quotient recovers ~1e-15).
fn rayleigh_quotient(a: &Mat4, v: &[C; 4]) -> C {
    let mut num = czero();
    let mut den = 0.0_f64;
    for i in 0..4 {
        let mut av = czero();
        for j in 0..4 {
            av += a[i][j] * v[j];
        }
        num += v[i].conj() * av;
        den += v[i].norm_sqr();
    }
    if den > 1e-300 {
        num / c(den, 0.0)
    } else {
        c(f64::NAN, f64::NAN)
    }
}

/// Rayleigh-residual ||A·v − ρ(v)·v||: root-noise-immune eigenvector check.
fn eig_residual(a: &Mat4, lambda: C, v: &[C; 4]) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..4 {
        let mut av = czero();
        for j in 0..4 {
            av += a[i][j] * v[j];
        }
        worst = worst.max((av - lambda * v[i]).norm());
    }
    worst
}

/// Rayleigh residual of the quotient-substituted pair (used by the cluster
/// arm's verify step so quartic root noise ~1e-8 can't fail good vectors).
fn rayleigh_residual(a: &Mat4, v: &[C; 4]) -> f64 {
    let rho = rayleigh_quotient(a, v);
    eig_residual(a, rho, v)
}

/// True when the Rayleigh residual is unacceptable. Written as !(r <= tol)
/// so NaN (non-finite candidate) counts as bad — NaN comparisons are false.
#[inline]
fn bad_rayleigh_residual(a: &Mat4, v: &[C; 4]) -> bool {
    !(rayleigh_residual(a, v) <= 1e-9)
}

/// Shift-invert iteration from an explicit start (60 iterations).
fn inverse_iteration_from(a: &Mat4, lambda: C, start: [C; 4]) -> [C; 4] {
    let mut shifted = *a;
    let eps = c(1e-10, 1e-10);
    for i in 0..4 {
        shifted[i][i] -= lambda + eps;
    }
    let inv = match mat4_inv(&shifted) {
        Some(m) => m,
        None => return [c(f64::NAN, f64::NAN); 4],
    };
    let mut v = start;
    for _ in 0..60 {
        let mut w = [czero(); 4];
        for i in 0..4 {
            for j in 0..4 {
                w[i] += inv[i][j] * v[j];
            }
        }
        normalize(&mut w);
        v = w;
    }
    v
}

fn inverse_iteration(a: &Mat4, lambda: C) -> [C; 4] {
    // (A - (lambda+eps) I)^{-1} applied repeatedly; eps keeps it nonsingular.
    let mut shifted = *a;
    let eps = c(1e-10, 1e-10);
    for i in 0..4 {
        shifted[i][i] -= lambda + eps;
    }
    let inv = match mat4_inv(&shifted) {
        Some(m) => m,
        None => return [cone(), czero(), czero(), czero()],
    };
    let mut v = [c(1.0, 0.0), c(0.3, 0.0), c(0.7, 0.0), c(0.2, 0.0)];
    for _ in 0..6 {
        let mut w = [czero(); 4];
        for i in 0..4 {
            for j in 0..4 {
                w[i] += inv[i][j] * v[j];
            }
        }
        normalize(&mut w);
        v = w;
    }
    v
}

#[inline]
fn normalize(v: &mut [C; 4]) {
    let mut n2 = 0.0;
    for x in v.iter() {
        n2 += x.norm_sqr();
    }
    let n = n2.sqrt();
    if n > 1e-300 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

// Modified Gram–Schmidt orthonormalization happens inside `null_space_basis`.
