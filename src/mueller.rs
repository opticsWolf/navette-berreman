//! Mueller-matrix post-processing (depolarization / polarimetry metrics).
//!
//! Scalar 4x4 real algebra over f64. Conventions transcribed from BerreMueller
//! (see §5.2 spike record below) — matched to *live*, not memory:
//!
//! - Pauli basis = `mueller.py::create_pauli_stack()` default `"optics"`:
//!   [I, σz, σx, σy] = [σ0, σ1, σ2, σ3], UNNORMALIZED
//!   (mueller.py:159-178; σ3 = [[0,−i],[i,0]]).
//! - Coherency H = Σ_ij M_ij · (σ_i ⊗ σ_j*)/4, exactly live's
//!   `cloude_decompose_matrix_stack` covariance accumulation (mueller.py:545-549,
//!   with `np.conjugate(pauli_stack)` on the second factor = our kron_conj).
//!   Live then forms C = L·H·L† (mueller.py:550-555, L = Λ/√2). L is unitary
//!   (Λ rows pairwise orthogonal, norm √2 — verified in the spike), so
//!   eig(H) == eig(C): we eig H directly and skip the L round-trip.
//!   Numerically pinned: |λ(H) − λ(C_live-pipeline)| ≤ 1.4e-15 (spike §5.2).
//! - DRAFT CORRECTION (spike §5.2 finding 7): the plan draft used a 1/√2-normalized
//!   basis with divisor /4, which gives Σλ = M00/2 (trace: Tr(σ̃0)²/4 = 2/4).
//!   Correct is /2 with the normalized basis ⟺ /4 unnormalized = H exactly
//!   (Tr(H) = M00·Tr(σ0)²/16·... = M00 — and G11b asserts Σλ == M00, which would
//!   have failed loudly on the draft constant).
//! - A-matrix 1/√2 factors (`berreman_mueller.py:156-160`) already mirrored in
//!   `transfer.rs::mueller_from_jones` (validated 8.9e-16); not re-derived here.
//! - CD sign: `berreman_mueller.py::get_m03_raw` (ll.776-782) satisfies
//!   get_m03_raw(t) ≡ 2·M03 EXACTLY (spike analytic check: with the A-matrix,
//!   M03 = Im(txx·conj(txy)) + Im(tyx·conj(tyy)) ∈ ℝ, and live's raw formula is
//!   twice that sum). Our `circular_dichroism` returns M03/M00 (dimensionless,
//!   matching live's `norm_mueller_matrix_stack` convention, mueller.py:1154);
//!   G11a pins ours == get_m03_raw/(2·M00) — the /2 is a live quirk, kept on
//!   the test side, never silently absorbed.
//! - Depolarization index is NOT in live (no depolarization_index_stack exists;
//!   live has only `get_purity_components_mueller_matrix_stack`, mueller.py:1197,
//!   whose D/P arms pin our |D|,|P|). DI follows Gil–Bernabeu (textbook), pinned
//!   by G11b identities + the eigenvalue identity
//!   DI² = (4·Σλ²/(Σλ)² − 1)/3 (asserted in Rust tests, not just Python).
//! - `parallel_decompose_matrix_stack` (mueller.py:507-518, sym/antisym split) is
//!   unused by all v1 metrics — noted, not ported.
//! - Brown/POLARIZANCE-class machinery (mueller.py:658-746) is differential
//!   calculus over z-resolved (b_vec, d_vec) — DEFERRED to v2 (needs Phase 2
//!   fields). v1 exposes only the plain normalized polarizance vector.
//!
//! Crate placement: `pub mod mueller;` in lib.rs (core, not feature-gated).
//! All metrics normalize by M00; M00 <= 0 => None (unphysical / dark matrix —
//! caller maps to NaN, same as n_failed rows).

use crate::cmatrix::{c, czero, C};

/// M[0][0] guard: every metric below normalizes by M00.
fn m00(m: &[[f64; 4]; 4]) -> Option<f64> {
    let v = m[0][0];
    if v > 1e-300 { Some(v) } else { None }
}

/// Depolarization index (Gil–Bernabeu):
///   DI = sqrt( Σ_ij M_ij² − M00² ) / (√3 · M00).
/// DI == 1 ⟺ non-depolarizing (Mueller-Jones); DI == 0 for diag(1,0,0,0).
pub fn depolarization_index(m: &[[f64; 4]; 4]) -> Option<f64> {
    let m00 = m00(m)?;
    let mut s = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            s += m[i][j] * m[i][j];
        }
    }
    let v = (s - m00 * m00).max(0.0); // max() guards −1e-18 rounding on Jones inputs
    Some(v.sqrt() / (3.0_f64.sqrt() * m00))
}

/// Diattenuation vector D (first ROW, normalized): D_k = M[0][k+1]/M00.
/// |D| pins against live `get_purity_components...` diattenuation arm (G11a).
pub fn diattenuation(m: &[[f64; 4]; 4]) -> Option<[f64; 3]> {
    let m00 = m00(m)?;
    Some([m[0][1] / m00, m[0][2] / m00, m[0][3] / m00])
}

/// Polarizance vector P (first COLUMN, normalized): P_k = M[k+1][0]/M00.
/// |P| pins against live `get_purity_components...` polarizance arm (G11a).
/// (NOT live's POLARIZANCE class — that is differential machinery, v2.)
pub fn polarizance(m: &[[f64; 4]; 4]) -> Option<[f64; 3]> {
    let m00 = m00(m)?;
    Some([m[1][0] / m00, m[2][0] / m00, m[3][0] / m00])
}

/// Circular dichroism from a transmission Mueller matrix: CD = M03/M00.
/// Sign pinned by the spike: live get_m03_raw ≡ 2·M03, same sign (G11a
/// compares against get_m03_raw/(2·M00)).
pub fn circular_dichroism(m: &[[f64; 4]; 4]) -> Option<f64> {
    Some(m[0][3] / m00(m)?)
}

/// Cloude coherency eigenvalues λ0 ≥ λ1 ≥ λ2 ≥ λ3 (Σλ = M00) + entropy.
#[derive(Clone, Copy, Debug)]
pub struct Cloude {
    pub lambda: [f64; 4], // descending, Σ == M00 (normalization self-check)
    pub entropy: f64,     // S = −Σ p_k log4 p_k, p = λ/Σλ, 0·log0 := 0
}

type Pauli = [[C; 2]; 2];

/// Pauli basis, written OUT explicitly (zero ambiguity for review against
/// `mueller.py::create_pauli_stack` default "optics": [0, z, x, y]).
/// UNNORMALIZED (no 1/√2 — live's stack is raw σ; the /4 in `cloude` carries
/// the normalization, giving H exactly and Σλ == M00).
fn pauli_basis() -> [Pauli; 4] {
    [
        [[c(1.0, 0.0), czero()], [czero(), c(1.0, 0.0)]], // σ0 = I
        [[c(1.0, 0.0), czero()], [czero(), c(-1.0, 0.0)]], // σ1 = σz
        [[czero(), c(1.0, 0.0)], [c(1.0, 0.0), czero()]], // σ2 = σx
        [[czero(), c(0.0, -1.0)], [c(0.0, 1.0), czero()]], // σ3 = σy [[0,−i],[i,0]]
    ]
}

/// Kronecker product with CONJUGATION on the second factor: K = A ⊗ conj(B),
/// 4x4, row-major block layout K[i*2+p][j*2+q] = A[i][j]·conj(B[p][q])
/// (matches the kron(J, conj(J)) layout in transfer.rs::mueller_from_jones
/// and live's np.kron(pauli_i, conjugate(pauli_j))).
fn kron_conj(a: &Pauli, b: &Pauli) -> [[C; 4]; 4] {
    let mut k = [[czero(); 4]; 4];
    for i in 0..2 {
        for j in 0..2 {
            for p in 0..2 {
                for q in 0..2 {
                    k[i * 2 + p][j * 2 + q] = a[i][j] * b[p][q].conj();
                }
            }
        }
    }
    k
}

pub fn cloude(m: &[[f64; 4]; 4]) -> Option<Cloude> {
    m00(m)?;
    let sig = pauli_basis();
    // H = Σ_ij M_ij · (σ_i ⊗ σ_j*)/4 — live's covariance accumulation verbatim.
    // eig(H) == eig(live C) by L-unitary similarity (module docs).
    let mut hm = [[czero(); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let k = kron_conj(&sig[i], &sig[j]);
            let w = c(m[i][j] / 4.0, 0.0);
            for r in 0..4 {
                for cc in 0..4 {
                    hm[r][cc] += w * k[r][cc];
                }
            }
        }
    }
    // Hermitian eigenproblem → Hermitian solver (NOT the general quartic eig4:
    // multiple roots are ill-conditioned for polynomial methods — (λ−1/4)⁴
    // coefficient fuzz yields complex pairs with imag ~1e-4, which is exactly
    // how G11b's ideal depolarizer caught this during implementation).
    let mut lam = crate::cmatrix::eig_hermitian4(&hm);
    // sort descending (insertion sort, n=4).
    // (Pauli ORDER is irrelevant here: simultaneous basis permutation leaves
    // eigenvalues invariant, and we return sorted values — the "classic silent
    // bug" can only bite an implementation returning per-component vectors.)
    for n in 1..4 {
        let x = lam[n];
        let mut k = n;
        while k > 0 && lam[k - 1] < x {
            lam[k] = lam[k - 1];
            k -= 1;
        }
        lam[k] = x;
    }
    // entropy, base-4 log; clamp tiny negatives from rounding to 0
    let sum: f64 = lam.iter().sum();
    if sum <= 1e-300 {
        return None;
    }
    let mut s = 0.0;
    for v in lam {
        let p = (v / sum).max(0.0);
        if p > 0.0 {
            s -= p * p.ln() / 4.0_f64.ln();
        }
    }
    Some(Cloude { lambda: lam, entropy: s })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye() -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// G11b identities + the DI↔eigenvalue cross-identity.
    #[test]
    fn identities() {
        let e = eye();
        assert!((depolarization_index(&e).unwrap() - 1.0).abs() < 1e-15);
        let cc = cloude(&e).unwrap();
        assert!((cc.lambda[0] - 1.0).abs() < 1e-12);
        assert!(cc.lambda[1].abs() < 1e-12 && cc.lambda[2].abs() < 1e-12 && cc.lambda[3].abs() < 1e-12);
        assert!(cc.entropy.abs() < 1e-12);
        // Σλ == M00 normalization self-check (catches Pauli-scale errors —
        // this is the assert that kills the /4-with-normalized-basis draft):
        assert!((cc.lambda.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // ideal depolarizer diag(1,0,0,0): DI = 0, maximal entropy 1
        let mut dep = [[0.0; 4]; 4];
        dep[0][0] = 1.0;
        assert!(depolarization_index(&dep).unwrap().abs() < 1e-15);
        let cd = cloude(&dep).unwrap();
        assert!((cd.entropy - 1.0).abs() < 1e-12);
        assert!((cd.lambda.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // guards: zero matrix -> None everywhere
        let z = [[0.0; 4]; 4];
        assert!(depolarization_index(&z).is_none() && cloude(&z).is_none());
        assert!(diattenuation(&z).is_none() && polarizance(&z).is_none());
        assert!(circular_dichroism(&z).is_none());
        // D/P/CD of identity vanish
        assert!(diattenuation(&e).unwrap().iter().all(|v| v.abs() < 1e-300));
        assert!(polarizance(&e).unwrap().iter().all(|v| v.abs() < 1e-300));
        assert!(circular_dichroism(&e).unwrap().abs() < 1e-300);
    }

    /// eig_hermitian4 on fixed fixtures with closed-form spectra (real +
    /// complex off-diagonals — the latter exercises the phase factor
    /// s = c·t·apq/|apq|), plus spectral invariants Σλ = Tr, Σλ² = Σ|h|².
    #[test]
    fn jacobi_known_spectra() {
        use crate::cmatrix::eig_hermitian4;
        let mut h = [[c(0.0, 0.0); 4]; 4];
        // (0,1) block [[2,1+i],[1−i,2]] → {2+√2, 2−√2}
        h[0][0] = c(2.0, 0.0);
        h[1][1] = c(2.0, 0.0);
        h[0][1] = c(1.0, 1.0);
        h[1][0] = c(1.0, -1.0);
        // (2,3) block [[1,0.5],[0.5,1]] → {1.5, 0.5}
        h[2][2] = c(1.0, 0.0);
        h[3][3] = c(1.0, 0.0);
        h[2][3] = c(0.5, 0.0);
        h[3][2] = c(0.5, 0.0);
        let mut lam = eig_hermitian4(&h);
        lam.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let s2 = 2.0_f64.sqrt();
        let exp = [2.0 + s2, 1.5, 2.0 - s2, 0.5]; // descending: 3.41, 1.5, 0.59, 0.5
        for (l, e) in lam.iter().zip(exp.iter()) {
            assert!((l - e).abs() < 1e-12, "{l} vs {e}");
        }
        // invariants: Σλ = Tr(H), Σλ² = Σ|h_ij|² (Frobenius, unitary-invariant)
        let tr: f64 = (0..4).map(|i| h[i][i].re).sum();
        let frob: f64 = h.iter().flat_map(|r| r.iter()).map(|v| v.norm_sqr()).sum();
        assert!((lam.iter().sum::<f64>() - tr).abs() < 1e-12);
        assert!((lam.iter().map(|v| v * v).sum::<f64>() - frob).abs() < 1e-12);
    }

    /// DI² = (4·Σλ²/(Σλ)² − 1)/3 on a depolarizing fixture (convex mixture of
    /// two projectors): ties the textbook formula to the eigendecomposition
    /// through an independent algebraic identity.
    #[test]
    fn di_eigenvalue_identity() {
        // M = 0.7·I + 0.3·diag(1,1,-1,-1): M00 = 1, non-depolarizing parts mix
        let m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.4],
        ];
        let di = depolarization_index(&m).unwrap();
        let cc = cloude(&m).unwrap();
        let sum: f64 = cc.lambda.iter().sum();
        let sq: f64 = cc.lambda.iter().map(|v| v * v).sum();
        let di2 = (4.0 * sq / (sum * sum) - 1.0) / 3.0;
        assert!((di * di - di2).abs() < 1e-12, "di={di} di2={di2}");
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(di > 0.0 && di < 1.0);
    }
}
