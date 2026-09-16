// SPDX-License-Identifier: LGPL-3.0-or-later
//! Dispersion kernels for `navette-berreman`, via the upstream `navette`
//! crate (crates.io, 0.7.0) — depend, don't port (Phase 9 decision, §9.0).
//!
//! Every scalar kernel here is a thin slice-friendly adapter over the
//! identically-named upstream free function (`navette::materials::*`): same
//! op order, same guards, same quirks — so live parity vs
//! `navette.materials.evaluate` holds by construction (target 0.0, gate
//! G14a). The quirks are upstream's to keep; see §9.2 for the catalog and
//! `tests/navette_parity.rs` for the executable pins.
//!
//! Two deliberate deltas from upstream, both validated here not assumed:
//!  * `table_nk` validates instead of panicking (upstream `assert!`s on
//!    grid length/mismatch) — core never panics; errors become `Err`.
//!  * Oscillator matrices ride as flat `&[f64]` + width (row-major
//!    `(N, width)`) instead of `ArrayView2`, so callers never need ndarray.
//!
//! Tensor assembly (birefringent `diag(n̂²)`) lives Python-side
//! (`navette/materials.py::evaluate_tensor`) over these kernels — exact
//! arithmetic, no upstream counterpart (upstream is scalar-only). The Rust
//! side provides only the `eps_of_n` squaring helper to keep the squaring
//! op identical on both sides of the binding.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;

use crate::cmatrix::C;

// ── view helpers ─────────────────────────────────────────────────────────

#[inline]
fn v1(xs: &[f64]) -> ArrayView1<'_, f64> {
    ArrayView1::from(xs)
}

#[inline]
fn vc(xs: &[C]) -> ArrayView1<'_, Complex64> {
    // C is Complex64 (cmatrix.rs); layout-identical, safe to reinterpret.
    // num-complex Complex<f64> is #[repr(C)] { re, im } — same as ndarray's.
    let p = xs.as_ptr() as *const Complex64;
    let s = xs.len();
    // SAFETY: Complex64 and C share layout; the view borrows `xs`.
    unsafe { ArrayView1::from_shape_ptr(s, p) }
}

fn mat_of(flat: &[f64], width: usize, what: &str) -> Result<Array2<f64>, String> {
    if width == 0 || flat.len() % width != 0 {
        return Err(format!(
            "{what} needs a flat (N, {width}) row-major matrix, got len {}",
            flat.len()
        ));
    }
    let n = flat.len() / width;
    if n == 0 {
        return Err(format!("{what} needs at least one oscillator"));
    }
    Array2::from_shape_vec((n, width), flat.to_vec())
        .map_err(|e| format!("{what} reshape failed: {e}"))
}

fn view2(m: &Array2<f64>) -> ArrayView2<'_, f64> {
    m.view()
}

// ── scalar kernels (upstream delegates) ──────────────────────────────────

pub fn konstant_nk(wl: &[f64], n: f64, k: f64) -> Vec<C> {
    navette::materials::table::konstant_nk(v1(wl), n, k).to_vec()
}

pub fn table_nk(
    wl: &[f64],
    grid: &[f64],
    n_vals: &[f64],
    k_vals: Option<&[f64]>,
    n_factor: f64,
    k_factor: f64,
) -> Result<Vec<C>, String> {
    if grid.len() < 2 {
        return Err(format!(
            "table_nk needs at least 2 grid points, got {}",
            grid.len()
        ));
    }
    if grid.len() != n_vals.len() {
        return Err(format!(
            "table_nk grid/n mismatch: {} vs {}",
            grid.len(),
            n_vals.len()
        ));
    }
    if let Some(kv) = k_vals {
        if kv.len() != grid.len() {
            return Err(format!(
                "table_nk grid/k mismatch: {} vs {}",
                grid.len(),
                kv.len()
            ));
        }
    }
    let k_view: Option<ArrayView1<f64>> = k_vals.map(v1);
    Ok(navette::materials::table::table_nk(
        v1(wl),
        v1(grid),
        v1(n_vals),
        k_view,
        n_factor,
        k_factor,
    )
    .to_vec())
}

pub fn cauchy_nk(wl: &[f64], a: f64, b: f64, c: f64) -> Vec<C> {
    navette::materials::cauchy::cauchy_nk(v1(wl), a, b, c).to_vec()
}

pub fn cauchy_urbach_nk(
    wl: &[f64],
    a: f64,
    b: f64,
    c: f64,
    alpha0: f64,
    eu: f64,
    lambda_g: f64,
) -> Vec<C> {
    navette::materials::cauchy::cauchy_urbach_nk(v1(wl), a, b, c, alpha0, eu, lambda_g)
        .to_vec()
}

pub fn sellmeier_nk(wl: &[f64], b: [f64; 3], c: [f64; 3]) -> Vec<C> {
    navette::materials::sellmeier::sellmeier_nk(v1(wl), b[0], c[0], b[1], c[1], b[2], c[2])
        .to_vec()
}

pub fn sellmeier_urbach_nk(
    wl: &[f64],
    b: [f64; 3],
    c: [f64; 3],
    alpha0: f64,
    eu: f64,
    lambda_g: f64,
) -> Vec<C> {
    navette::materials::sellmeier::sellmeier_urbach_nk(
        v1(wl), b[0], c[0], b[1], c[1], b[2], c[2], alpha0, eu, lambda_g,
    )
    .to_vec()
}

pub fn lorentz_nk(wl: &[f64], osc_flat: &[f64], eps_inf: f64) -> Result<Vec<C>, String> {
    let m = mat_of(osc_flat, 3, "Lorentz osc")?;
    Ok(navette::materials::lorentz::lorentz_nk(v1(wl), view2(&m), eps_inf).to_vec())
}

pub fn drude_nk(wl: &[f64], omega_p: f64, gamma: f64, eps_inf: f64) -> Vec<C> {
    navette::materials::drude::drude_nk(v1(wl), omega_p, gamma, eps_inf).to_vec()
}

pub fn drude_lorentz_nk(
    wl: &[f64],
    omega_p: f64,
    gamma_d: f64,
    eps_inf: f64,
    osc_flat: &[f64],
) -> Result<Vec<C>, String> {
    let m = mat_of(osc_flat, 3, "DrudeLorentz osc")?;
    Ok(
        navette::materials::drude::drude_lorentz_nk(v1(wl), omega_p, gamma_d, eps_inf, view2(&m))
            .to_vec(),
    )
}

pub fn cody_lorentz_nk(
    wl: &[f64],
    eg: f64,
    et: f64,
    eu: f64,
    osc_flat: &[f64],
    eps_inf: f64,
) -> Result<Vec<C>, String> {
    let m = mat_of(osc_flat, 4, "CodyLorentz osc")?;
    navette::materials::cody_lorentz::cody_lorentz_nk(v1(wl), eg, et, eu, view2(&m), eps_inf)
        .map(|a| a.to_vec())
}

pub fn fb_interband_nk(wl: &[f64], n_inf: f64, ib_flat: &[f64]) -> Result<Vec<C>, String> {
    let m = mat_of(ib_flat, 4, "ForouhiBloomer ib")?;
    Ok(navette::materials::forouhi_bloomer::fb_interband_nk(v1(wl), n_inf, view2(&m)).to_vec())
}

pub fn fb_metal_nk(
    wl: &[f64],
    n_inf: f64,
    fe: [f64; 3],
    ib_flat: &[f64],
) -> Result<Vec<C>, String> {
    let m = mat_of(ib_flat, 4, "ForouhiBloomer ib")?;
    let fe_arr = Array1::from_vec(fe.to_vec());
    Ok(
        navette::materials::forouhi_bloomer::fb_metal_nk(v1(wl), n_inf, fe_arr.view(), view2(&m))
            .to_vec(),
    )
}

pub fn tauc_lorentz_nk(
    wl: &[f64],
    eg: f64,
    osc_flat: &[f64],
    eps_inf: f64,
) -> Result<Vec<C>, String> {
    let m = mat_of(osc_flat, 3, "TaucLorentz osc")?;
    navette::materials::tauc_lorentz::tauc_lorentz_nk(v1(wl), eg, view2(&m), eps_inf)
        .map(|a| a.to_vec())
}

pub fn ubf_nk(wl: &[f64], osc_flat: &[f64], eps_inf: f64) -> Result<Vec<C>, String> {
    let m = mat_of(osc_flat, 6, "UBF osc")?;
    navette::materials::ubf::ubf_nk(v1(wl), view2(&m), eps_inf).map(|a| a.to_vec())
}

// ── EMA (take n̂, return ε — caller applies eps_to_nk, upstream order) ────

pub fn ema_lichtenecker(n_i: &[C], n_h: &[C], f: f64) -> Vec<C> {
    navette::materials::ema::lichtenecker(vc(n_i), vc(n_h), f).to_vec()
}

pub fn ema_looyenga(n_i: &[C], n_h: &[C], f: f64) -> Vec<C> {
    navette::materials::ema::looyenga(vc(n_i), vc(n_h), f).to_vec()
}

pub fn ema_power_law(n_i: &[C], n_h: &[C], f: f64, alpha: f64) -> Vec<C> {
    navette::materials::ema::general_power_law(vc(n_i), vc(n_h), f, alpha).to_vec()
}

pub fn ema_maxwell_garnett(n_i: &[C], n_h: &[C], f: f64) -> Vec<C> {
    navette::materials::ema::maxwell_garnett(vc(n_i), vc(n_h), f).to_vec()
}

pub fn ema_mori_tanaka(n_i: &[C], n_h: &[C], f: f64, l: f64) -> Vec<C> {
    navette::materials::ema::mori_tanaka(vc(n_i), vc(n_h), f, l).to_vec()
}

pub fn ema_bruggeman(n_i: &[C], n_h: &[C], f: f64, max_iter: usize, tol: f64) -> Vec<C> {
    navette::materials::ema::bruggeman(vc(n_i), vc(n_h), f, max_iter, tol).to_vec()
}

pub fn ema_roughness(n_bottom: &[C], n_top: &[C]) -> Vec<C> {
    navette::materials::ema::roughness_interface(vc(n_bottom), vc(n_top)).to_vec()
}

pub fn eps_to_nk(eps: &[C]) -> Vec<C> {
    navette::materials::ema::eps_to_nk(vc(eps)).to_vec()
}

/// Permittivity of a complex index (element-wise n̂²) — the tensor-diag op.
pub fn eps_of_n(nk: &[C]) -> Vec<C> {
    nk.iter().map(|z| z * z).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wl_grid() -> Vec<f64> {
        (0..200).map(|i| 300.0 + i as f64 * 4.5).collect()
    }

    #[test]
    fn table_validates_instead_of_panicking() {
        assert!(table_nk(&[500.0], &[500.0], &[1.5], None, 1.0, 1.0).is_err());
        assert!(table_nk(&[500.0], &[400.0, 600.0], &[1.5], None, 1.0, 1.0).is_err());
        assert!(table_nk(
            &[500.0],
            &[400.0, 600.0],
            &[1.5, 1.6],
            Some(&[0.1]),
            1.0,
            1.0
        )
        .is_err());
    }

    #[test]
    fn osc_shapes_rejected() {
        let wl = wl_grid();
        assert!(lorentz_nk(&wl, &[], 1.0).is_err());
        assert!(lorentz_nk(&wl, &[1.0, 2.0], 1.0).is_err());
        assert!(lorentz_nk(&wl, &[1.0, 2.0, 3.0, 4.0], 1.0).is_err()); // len 4 % 3
        assert!(ubf_nk(&wl, &[1.0; 5], 1.0).is_err()); // len 5 % 6
    }

    #[test]
    fn ema_endpoints_exact() {
        let a = konstant_nk(&wl_grid(), 1.5, 0.0);
        let b = konstant_nk(&wl_grid(), 2.0, 0.1);
        // f=0 -> host eps, f=1 -> inclusion eps (upstream F1.5 short-circuit).
        for (got, want) in ema_bruggeman(&b, &a, 0.0, 100, 1e-9).iter().zip(eps_of_n(&a)) {
            assert!((got - want).norm() == 0.0);
        }
        for (got, want) in ema_bruggeman(&b, &a, 1.0, 100, 1e-9).iter().zip(eps_of_n(&b)) {
            assert!((got - want).norm() == 0.0);
        }
    }
}
