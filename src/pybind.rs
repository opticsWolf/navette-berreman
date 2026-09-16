// SPDX-License-Identifier: LGPL-3.0-or-later
//! PyO3 bindings for the Berreman/Mueller solver, styled after `_smatrix`:
//! pure-Rust `*_inner` physics in `transfer`/`berreman`/`cmatrix`, thin
//! `#[pyfunction]` wrappers here, and a Rayon-parallel sweep over the
//! (wavelength, angle) grid behind `py.detach`.
//!
//! Array marshalling (all flat `f64` ndarrays, row-major):
//!   * `wls`            : [n_wl]                      wavelengths (nm)
//!   * `thetas`         : [n_theta]                   incidence angles (rad)
//!   * `n_entry_re/im`  : [n_wl]                      entry index (dispersive)
//!   * `n_exit_re/im`   : [n_wl]                      exit index (dispersive)
//!   * `eps`            : [n_wl * n_layers * 18]      per (wl, layer) 3x3 tensor
//!                        as 9 (re, im) pairs, row-major eps[i][j] at i*3+j
//!   * `thicknesses`    : [n_layers]                  layer thicknesses (nm)
//! The `*_full` variant additionally takes `rho`, `rhop`, `mu` in the same
//! [n_wl * n_layers * 18] layout for the magneto-optic Berreman matrix.
//!
//! Output is a dict of ndarrays shaped `[n_wl, n_theta, 2, 2]` (Jones / power)
//! and `[n_wl, n_theta, 4, 4]` (Mueller).

use numpy::{PyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::materials;

use crate::berreman::Tensor3;
use crate::cmatrix::{c, C};
use crate::transfer::{mueller_from_jones, solve_stack_periodic, ExitSpec, Geometry, LayerSpec,
                     Method, PowerMethod, SolveResult};
use crate::rotations;
use crate::roughness;

/// Read a 3x3 complex tensor from a flat (re,im)-pair slice at `off`.
#[inline]
fn read_tensor(data: &[f64], off: usize) -> Tensor3 {
    let mut t = [[c(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let p = off + (i * 3 + j) * 2;
            t[i][j] = c(data[p], data[p + 1]);
        }
    }
    t
}

/// Flattened per-point observables. Fixed-size, cheap to move out of Rayon.
struct PointOut {
    j_refl: [[C; 2]; 2],
    j_trans: [[C; 2]; 2],
    j_refl_c: [[C; 2]; 2],
    j_trans_c: [[C; 2]; 2],
    r: [[f64; 2]; 2],
    t: [[f64; 2]; 2],
    m_refl: [[f64; 4]; 4],
    m_trans: [[f64; 4]; 4],
    flux: bool,
    ok: bool,
}

impl PointOut {
    fn from_solve(s: &SolveResult) -> PointOut {
        // P10: ok:false on ANY non-finite Jones/power output (defense in
        // depth — catches NaN-through-Some paths that never touch an
        // Option, e.g. det-singular fresnel divisions; previously counted
        // ok:true with NaN values, hiding n_failed).
        let mut finite = true;
        for i in 0..2 {
            for j in 0..2 {
                finite = finite
                    && s.jones.refl[i][j].is_finite()
                    && s.jones.trans[i][j].is_finite()
                    && s.r_power[i][j].is_finite()
                    && s.t_power[i][j].is_finite();
            }
        }
        let mut r = [[0.0; 2]; 2];
        let mut t = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                r[i][j] = s.r_power[i][j].re;
                t[i][j] = s.t_power[i][j].re;
            }
        }
        let mr = mueller_from_jones(&s.jones.refl);
        let mt = mueller_from_jones(&s.jones.trans);
        let mut m_refl = [[0.0; 4]; 4];
        let mut m_trans = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m_refl[i][j] = mr[i][j].re;
                m_trans[i][j] = mt[i][j].re;
            }
        }
        PointOut {
            j_refl: s.jones.refl,
            j_trans: s.jones.trans,
            j_refl_c: s.jones_circ.refl,
            j_trans_c: s.jones_circ.trans,
            r,
            t,
            m_refl,
            m_trans,
            flux: s.power_method == PowerMethod::Flux,
            ok: finite,
        }
    }
    fn nan() -> PointOut {
        let n = f64::NAN;
        let cn = c(n, n);
        PointOut {
            j_refl: [[cn; 2]; 2],
            j_trans: [[cn; 2]; 2],
            j_refl_c: [[cn; 2]; 2],
            j_trans_c: [[cn; 2]; 2],
            r: [[n; 2]; 2],
            t: [[n; 2]; 2],
            m_refl: [[n; 4]; 4],
            m_trans: [[n; 4]; 4],
            flux: false,
            ok: false,
        }
    }
}

/// Core sweep shared by the simple and full entry points.
#[allow(clippy::too_many_arguments)]
fn sweep(
    py: Python<'_>,
    wls: &[f64],
    thetas: &[f64],
    ne_re: &[f64],
    ne_im: &[f64],
    nx_re: &[f64],
    nx_im: &[f64],
    eps: &[f64],
    rho: Option<&[f64]>,
    rhop: Option<&[f64]>,
    mu: Option<&[f64]>,
    thick: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    method_code: i32,
    exit_eps: &[f64],
    exit_rho: &[f64],
    exit_rhop: &[f64],
    exit_mu: &[f64],
    periods: u32,
) -> PyResult<Py<PyDict>> {
    let n_wl = wls.len();
    let n_th = thetas.len();
    let n_layers = thick.len();
    if n_wl == 0 || n_th == 0 || n_layers == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "wls, thetas and thicknesses must all be non-empty",
        ));
    }
    if periods == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "periods must be >= 1 (periods=1 is a plain stack)",
        ));
    }
    let want = n_wl * n_layers * 18;
    if eps.len() != want {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "eps length {} != n_wl*n_layers*18 = {}",
            eps.len(),
            want
        )));
    }
    // One roughness spec per interface: n_layers + 1
    // (entry|layer0, layer0|layer1, …, layer_{n-1}|exit).
    if rough_types.len() != n_layers + 1 || rough_vals.len() != n_layers + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "rough_types/rough_vals must have length n_layers+1 = {}",
            n_layers + 1
        )));
    }
    let has_rough = rough_types.iter().any(|&t| t != 0);
    let method = match method_code {
        1 => {
            if has_rough {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "roughness (Route 1) is only supported with the scattering method; \
                     use method='scattering' or expand the interface into graded sublayers",
                ));
            }
            Method::Transfer
        }
        2 => {
            if has_rough {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "roughness (Route 1) is only supported with the scattering method; \
                     use method='scattering' or expand the interface into graded sublayers",
                ));
            }
            Method::Exponential
        }
        _ => Method::Scattering, // 0 and any unknown code stay SM (legacy)
    };

    // Pre-build the layer stack for each wavelength (eps is dispersive).
    let stacks = decode_stacks(
        n_wl, n_layers, eps, rho, rhop, mu, thick,
        Some((rough_types, rough_vals)),
    );
    let exit_roughness = (rough_types[n_layers], rough_vals[n_layers]);

    // Exit-spec decode (shared helper; same errors as before the P2 refactor).
    let exits = decode_exits(
        n_wl, nx_re, nx_im, exit_eps, exit_rho, exit_rhop, exit_mu,
        rho.is_some(),
    )?;

    let total = n_wl * n_th;
    let outs: Vec<PointOut> = py.detach(|| {
        (0..total)
            .into_par_iter()
            .map(|k| {
                let w = k / n_th;
                let th = k % n_th;
                let geom = Geometry {
                    wl_nm: wls[w],
                    theta_in_rad: thetas[th],
                    n_entry: c(ne_re[w], ne_im[w]),
                    n_exit: c(nx_re[w], nx_im[w]),
                    exit: exits[w].clone(),
                    exit_roughness,
                };
                // layers here are the UNIT CELL; Rust repeats them `periods`
                // times (periods == 1 delegates to solve_stack internally).
                match solve_stack_periodic(&geom, &stacks[w], periods, method) {
                    Some(s) => PointOut::from_solve(&s),
                    None => PointOut::nan(),
                }
            })
            .collect()
    });

    // Scatter the fixed-size per-point arrays into flat output buffers.
    let n22 = total * 4;
    let n44 = total * 16;
    let mut j_refl_re = vec![0.0; n22];
    let mut j_refl_im = vec![0.0; n22];
    let mut j_trans_re = vec![0.0; n22];
    let mut j_trans_im = vec![0.0; n22];
    let mut j_refl_c_re = vec![0.0; n22];
    let mut j_refl_c_im = vec![0.0; n22];
    let mut j_trans_c_re = vec![0.0; n22];
    let mut j_trans_c_im = vec![0.0; n22];
    let mut r_buf = vec![0.0; n22];
    let mut t_buf = vec![0.0; n22];
    let mut m_refl = vec![0.0; n44];
    let mut m_trans = vec![0.0; n44];
    let mut pm_buf = vec![0.0; total]; // 0 = JonesFactor, 1 = Flux

    for (k, o) in outs.iter().enumerate() {
        let b2 = k * 4;
        for i in 0..2 {
            for j in 0..2 {
                let idx = b2 + i * 2 + j;
                j_refl_re[idx] = o.j_refl[i][j].re;
                j_refl_im[idx] = o.j_refl[i][j].im;
                j_trans_re[idx] = o.j_trans[i][j].re;
                j_trans_im[idx] = o.j_trans[i][j].im;
                j_refl_c_re[idx] = o.j_refl_c[i][j].re;
                j_refl_c_im[idx] = o.j_refl_c[i][j].im;
                j_trans_c_re[idx] = o.j_trans_c[i][j].re;
                j_trans_c_im[idx] = o.j_trans_c[i][j].im;
                r_buf[idx] = o.r[i][j];
                t_buf[idx] = o.t[i][j];
            }
        }
        let b4 = k * 16;
        for i in 0..4 {
            for j in 0..4 {
                let idx = b4 + i * 4 + j;
                m_refl[idx] = o.m_refl[i][j];
                m_trans[idx] = o.m_trans[i][j];
            }
        }
        pm_buf[k] = if o.flux { 1.0 } else { 0.0 };
    }

    let n_fail = outs.iter().filter(|o| !o.ok).count();

    let out = PyDict::new(py);
    let s22 = [n_wl, n_th, 2, 2];
    let s44 = [n_wl, n_th, 4, 4];
    macro_rules! put22 {
        ($name:expr, $buf:expr) => {
            out.set_item($name, PyArray::from_vec(py, $buf).reshape(s22)?)?;
        };
    }
    macro_rules! put44 {
        ($name:expr, $buf:expr) => {
            out.set_item($name, PyArray::from_vec(py, $buf).reshape(s44)?)?;
        };
    }
    put22!("J_refl_re", j_refl_re);
    put22!("J_refl_im", j_refl_im);
    put22!("J_trans_re", j_trans_re);
    put22!("J_trans_im", j_trans_im);
    put22!("J_refl_c_re", j_refl_c_re);
    put22!("J_refl_c_im", j_refl_c_im);
    put22!("J_trans_c_re", j_trans_c_re);
    put22!("J_trans_c_im", j_trans_c_im);
    put22!("R", r_buf);
    put22!("T", t_buf);
    put44!("M_refl", m_refl);
    put44!("M_trans", m_trans);
    out.set_item(
        "power_method",
        PyArray::from_vec(py, pm_buf).reshape([n_wl, n_th])?,
    )?;
    out.set_item("n_failed", n_fail)?;
    Ok(out.into())
}

/// Pre-build the layer stack for each wavelength (eps is dispersive).
/// Shared by sweep() and fields_sweep(). `rough` = per-interface
/// (types, vals); None gives clean (0, 0.0) fronts (fields path — Route 1
/// dressing is a far-field power redistribution with no local-field meaning;
/// the fields() wrapper rejects roughness and points at graded_stack).
fn decode_stacks(
    n_wl: usize,
    n_layers: usize,
    eps: &[f64],
    rho: Option<&[f64]>,
    rhop: Option<&[f64]>,
    mu: Option<&[f64]>,
    thick: &[f64],
    rough: Option<(&[i32], &[f64])>,
) -> Vec<Vec<LayerSpec>> {
    let mut stacks: Vec<Vec<LayerSpec>> = Vec::with_capacity(n_wl);
    for w in 0..n_wl {
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let off = (w * n_layers + l) * 18;
            let eps_t = read_tensor(eps, off);
            let full = match (rho, rhop, mu) {
                (Some(r), Some(rp), Some(m)) => {
                    Some((read_tensor(r, off), read_tensor(rp, off), read_tensor(m, off)))
                }
                _ => None,
            };
            let front_roughness = match rough {
                Some((rt, rv)) => (rt[l], rv[l]),
                None => (0, 0.0),
            };
            layers.push(LayerSpec {
                eps: eps_t,
                thickness_nm: thick[l],
                full,
                front_roughness,
            });
        }
        stacks.push(layers);
    }
    stacks
}

/// Exit-spec decode. Length protocol (validated up front, before any
/// allocation, so Python gets ValueError not a panic):
///   exit_eps len 0       -> isotropic for all wl
///   exit_eps len n_wl*18 -> anisotropic (simple unless MO set given)
///   exit_rho/rhop/mu: each len 0 or n_wl*18; all-or-none with exit_eps.
/// Shared by sweep() and fields_sweep(). `layer_full` = whether the layer
/// tensors came from the full (MO) path — exit MO tensors require it.
#[allow(clippy::too_many_arguments)]
fn decode_exits(
    n_wl: usize,
    nx_re: &[f64],
    nx_im: &[f64],
    exit_eps: &[f64],
    exit_rho: &[f64],
    exit_rhop: &[f64],
    exit_mu: &[f64],
    layer_full: bool,
) -> PyResult<Vec<ExitSpec>> {
    let n_exit_want = n_wl * 18;
    for (name, s) in [
        ("exit_eps", exit_eps.len()),
        ("exit_rho", exit_rho.len()),
        ("exit_rhop", exit_rhop.len()),
        ("exit_mu", exit_mu.len()),
    ] {
        if s != 0 && s != n_exit_want {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} length {} != 0 or n_wl*18 = {}",
                s, n_exit_want
            )));
        }
    }
    let mo_given =
        !exit_rho.is_empty() || !exit_rhop.is_empty() || !exit_mu.is_empty();
    let mo_full =
        !exit_rho.is_empty() && !exit_rhop.is_empty() && !exit_mu.is_empty();
    if mo_given && !mo_full {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "exit_rho/rhop/mu must be given together (all len n_wl*18 or all empty)",
        ));
    }
    if mo_full && exit_eps.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "exit MO tensors require exit_eps",
        ));
    }
    if !layer_full && mo_full {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "exit MO tensors require the full entry point (solve_grid_full / fields_grid_full)",
        ));
    }
    // per-wavelength exit specs
    let mut exits: Vec<ExitSpec> = Vec::with_capacity(n_wl);
    for w in 0..n_wl {
        if exit_eps.is_empty() {
            exits.push(ExitSpec::Isotropic(c(nx_re[w], nx_im[w])));
        } else {
            let eps_t = read_tensor(exit_eps, w * 18);
            let full = if mo_full {
                Some((
                    read_tensor(exit_rho, w * 18),
                    read_tensor(exit_rhop, w * 18),
                    read_tensor(exit_mu, w * 18),
                ))
            } else {
                None
            };
            exits.push(ExitSpec::Anisotropic { eps: eps_t, full });
        }
    }
    Ok(exits)
}

/// Reduced Berreman (eps only) spectrum sweep.
#[pyfunction]
#[pyo3(name = "solve_grid_simple")]
#[pyo3(signature = (wls, thetas, n_entry_re, n_entry_im, n_exit_re, n_exit_im, eps, thicknesses, rough_types, rough_vals, method, exit_eps, periods))]
#[allow(clippy::too_many_arguments)]
pub fn solve_grid_simple(
    py: Python<'_>,
    wls: PyReadonlyArray1<f64>,
    thetas: PyReadonlyArray1<f64>,
    n_entry_re: PyReadonlyArray1<f64>,
    n_entry_im: PyReadonlyArray1<f64>,
    n_exit_re: PyReadonlyArray1<f64>,
    n_exit_im: PyReadonlyArray1<f64>,
    eps: PyReadonlyArray1<f64>,
    thicknesses: PyReadonlyArray1<f64>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    method: i32,
    exit_eps: PyReadonlyArray1<f64>,
    periods: u32,
) -> PyResult<Py<PyDict>> {
    sweep(
        py,
        wls.as_slice()?,
        thetas.as_slice()?,
        n_entry_re.as_slice()?,
        n_entry_im.as_slice()?,
        n_exit_re.as_slice()?,
        n_exit_im.as_slice()?,
        eps.as_slice()?,
        None,
        None,
        None,
        thicknesses.as_slice()?,
        rough_types.as_slice()?,
        rough_vals.as_slice()?,
        method,
        exit_eps.as_slice()?,
        &[],
        &[],
        &[],
        periods,
    )
}

/// Full magneto-optic Berreman spectrum sweep (eps, rho, rhop, mu).
#[pyfunction]
#[pyo3(name = "solve_grid_full")]
#[pyo3(signature = (wls, thetas, n_entry_re, n_entry_im, n_exit_re, n_exit_im, eps, rho, rhop, mu, thicknesses, rough_types, rough_vals, method, exit_eps, exit_rho, exit_rhop, exit_mu, periods))]
#[allow(clippy::too_many_arguments)]
pub fn solve_grid_full(
    py: Python<'_>,
    wls: PyReadonlyArray1<f64>,
    thetas: PyReadonlyArray1<f64>,
    n_entry_re: PyReadonlyArray1<f64>,
    n_entry_im: PyReadonlyArray1<f64>,
    n_exit_re: PyReadonlyArray1<f64>,
    n_exit_im: PyReadonlyArray1<f64>,
    eps: PyReadonlyArray1<f64>,
    rho: PyReadonlyArray1<f64>,
    rhop: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    thicknesses: PyReadonlyArray1<f64>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    method: i32,
    exit_eps: PyReadonlyArray1<f64>,
    exit_rho: PyReadonlyArray1<f64>,
    exit_rhop: PyReadonlyArray1<f64>,
    exit_mu: PyReadonlyArray1<f64>,
    periods: u32,
) -> PyResult<Py<PyDict>> {
    sweep(
        py,
        wls.as_slice()?,
        thetas.as_slice()?,
        n_entry_re.as_slice()?,
        n_entry_im.as_slice()?,
        n_exit_re.as_slice()?,
        n_exit_im.as_slice()?,
        eps.as_slice()?,
        Some(rho.as_slice()?),
        Some(rhop.as_slice()?),
        Some(mu.as_slice()?),
        thicknesses.as_slice()?,
        rough_types.as_slice()?,
        rough_vals.as_slice()?,
        method,
        exit_eps.as_slice()?,
        exit_rho.as_slice()?,
        exit_rhop.as_slice()?,
        exit_mu.as_slice()?,
        periods,
    )
}

// ── internal fields sweep (Phase 2) ───────────────────────────────────────

/// Shared fields sweep. Mirrors sweep()'s stack pre-build and per-wavelength
/// ExitSpec decode (same helpers), but:
///   * no roughness (Route 1 is far-field-only; wrapper rejects it),
///   * no periods (wrapper expands the cell; z needs the unfolded sequence),
///   * `method` selects nothing in propagation — fields_at_depths is always
///     transfer-based (§2.3); the slot keeps the Python method= kwarg meaningful.
/// Returns dict with E_p/H_p/E_s/H_s as re/im [n_wl,n_th,n_z,3] pairs,
/// "z_nm" [n_z] echo, "n_failed".
/// Per-task buffer layout (see fields::field_buf_index — the rule lives there):
///   idx = (((pol*2 + eh) * n_z + iz) * 3 + comp) * 2  (+1 = imag),
/// pol: p = 0, s = 1; eh: E = 0, H = 1.
#[allow(clippy::too_many_arguments)]
fn fields_sweep(
    py: Python<'_>,
    wls: &[f64],
    thetas: &[f64],
    ne_re: &[f64],
    ne_im: &[f64],
    nx_re: &[f64],
    nx_im: &[f64],
    eps: &[f64],
    rho: Option<&[f64]>,
    rhop: Option<&[f64]>,
    mu: Option<&[f64]>,
    thick: &[f64],
    z_nm: &[f64],
    exit_eps: &[f64],
    exit_rho: &[f64],
    exit_rhop: &[f64],
    exit_mu: &[f64],
    method_code: i32,
) -> PyResult<Py<PyDict>> {
    use crate::fields::{field_buf_index, fields_at_depths};
    let n_wl = wls.len();
    let n_th = thetas.len();
    let n_layers = thick.len();
    let n_z = z_nm.len();
    if n_wl == 0 || n_th == 0 || n_layers == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "wls, thetas and thicknesses must all be non-empty",
        ));
    }
    if n_z == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "z_nm must be non-empty",
        ));
    }
    let want = n_wl * n_layers * 18;
    if eps.len() != want {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "eps length {} != n_wl*n_layers*18 = {}",
            eps.len(),
            want
        )));
    }
    let method = match method_code {
        1 => Method::Transfer,
        2 => Method::Exponential,
        _ => Method::Scattering,
    };
    let stacks = decode_stacks(n_wl, n_layers, eps, rho, rhop, mu, thick, None);
    let exits = decode_exits(
        n_wl, nx_re, nx_im, exit_eps, exit_rho, exit_rhop, exit_mu,
        rho.is_some(),
    )?;

    let total = n_wl * n_th;
    // Each task -> flat [n_z*24] f64 + ok flag; indexed collect keeps order.
    let outs: Vec<Option<Vec<f64>>> = py.detach(|| {
        (0..total)
            .into_par_iter()
            .map(|k| {
                let w = k / n_th;
                let th = k % n_th;
                let geom = Geometry {
                    wl_nm: wls[w],
                    theta_in_rad: thetas[th],
                    n_entry: c(ne_re[w], ne_im[w]),
                    n_exit: c(nx_re[w], nx_im[w]),
                    exit: exits[w].clone(),
                    exit_roughness: (0, 0.0), // unused on the fields path
                };
                let pts = fields_at_depths(&geom, &stacks[w], z_nm, method)?;
                let mut buf = vec![0.0; n_z * 24];
                for (iz, p) in pts.iter().enumerate() {
                    for (ji, fv) in p.per_input.iter().enumerate() {
                        for comp in 0..3 {
                            let be = field_buf_index(ji, 0, iz, comp, n_z);
                            buf[be] = fv.e[comp].re;
                            buf[be + 1] = fv.e[comp].im;
                            let bh = field_buf_index(ji, 1, iz, comp, n_z);
                            buf[bh] = fv.h[comp].re;
                            buf[bh + 1] = fv.h[comp].im;
                        }
                    }
                }
                Some(buf)
            })
            .collect()
    });

    // Scatter task buffers into 8 [total, n_z, 3] re/im buffers.
    // None -> NaN row (same convention as sweep).
    let cell = n_z * 3;
    let mut ep_re = vec![f64::NAN; total * cell];
    let mut ep_im = vec![f64::NAN; total * cell];
    let mut hp_re = vec![f64::NAN; total * cell];
    let mut hp_im = vec![f64::NAN; total * cell];
    let mut es_re = vec![f64::NAN; total * cell];
    let mut es_im = vec![f64::NAN; total * cell];
    let mut hs_re = vec![f64::NAN; total * cell];
    let mut hs_im = vec![f64::NAN; total * cell];
    let mut n_fail = 0usize;
    for (k, o) in outs.iter().enumerate() {
        let Some(buf) = o else {
            n_fail += 1;
            continue;
        };
        let base = k * cell;
        for iz in 0..n_z {
            for comp in 0..3 {
                let dst = base + iz * 3 + comp;
                let be = field_buf_index(0, 0, iz, comp, n_z);
                ep_re[dst] = buf[be];
                ep_im[dst] = buf[be + 1];
                let bh = field_buf_index(0, 1, iz, comp, n_z);
                hp_re[dst] = buf[bh];
                hp_im[dst] = buf[bh + 1];
                let be = field_buf_index(1, 0, iz, comp, n_z);
                es_re[dst] = buf[be];
                es_im[dst] = buf[be + 1];
                let bh = field_buf_index(1, 1, iz, comp, n_z);
                hs_re[dst] = buf[bh];
                hs_im[dst] = buf[bh + 1];
            }
        }
    }

    let out = PyDict::new(py);
    let s = [n_wl, n_th, n_z, 3];
    macro_rules! putf {
        ($name:expr, $buf:expr) => {
            out.set_item($name, PyArray::from_vec(py, $buf).reshape(s)?)?;
        };
    }
    putf!("E_p_re", ep_re);
    putf!("E_p_im", ep_im);
    putf!("H_p_re", hp_re);
    putf!("H_p_im", hp_im);
    putf!("E_s_re", es_re);
    putf!("E_s_im", es_im);
    putf!("H_s_re", hs_re);
    putf!("H_s_im", hs_im);
    out.set_item("z_nm", PyArray::from_vec(py, z_nm.to_vec()))?;
    out.set_item("n_failed", n_fail)?;
    Ok(out.into())
}

/// Reduced (eps only) fields sweep.
#[pyfunction]
#[pyo3(name = "fields_grid_simple")]
#[pyo3(signature = (wls, thetas, n_entry_re, n_entry_im, n_exit_re, n_exit_im, eps, thicknesses, z_nm, exit_eps, method))]
#[allow(clippy::too_many_arguments)]
pub fn fields_grid_simple(
    py: Python<'_>,
    wls: PyReadonlyArray1<f64>,
    thetas: PyReadonlyArray1<f64>,
    n_entry_re: PyReadonlyArray1<f64>,
    n_entry_im: PyReadonlyArray1<f64>,
    n_exit_re: PyReadonlyArray1<f64>,
    n_exit_im: PyReadonlyArray1<f64>,
    eps: PyReadonlyArray1<f64>,
    thicknesses: PyReadonlyArray1<f64>,
    z_nm: PyReadonlyArray1<f64>,
    exit_eps: PyReadonlyArray1<f64>,
    method: i32,
) -> PyResult<Py<PyDict>> {
    fields_sweep(
        py,
        wls.as_slice()?,
        thetas.as_slice()?,
        n_entry_re.as_slice()?,
        n_entry_im.as_slice()?,
        n_exit_re.as_slice()?,
        n_exit_im.as_slice()?,
        eps.as_slice()?,
        None,
        None,
        None,
        thicknesses.as_slice()?,
        z_nm.as_slice()?,
        exit_eps.as_slice()?,
        &[],
        &[],
        &[],
        method,
    )
}

/// Full magneto-optic fields sweep (eps, rho, rhop, mu).
#[pyfunction]
#[pyo3(name = "fields_grid_full")]
#[pyo3(signature = (wls, thetas, n_entry_re, n_entry_im, n_exit_re, n_exit_im, eps, rho, rhop, mu, thicknesses, z_nm, exit_eps, exit_rho, exit_rhop, exit_mu, method))]
#[allow(clippy::too_many_arguments)]
pub fn fields_grid_full(
    py: Python<'_>,
    wls: PyReadonlyArray1<f64>,
    thetas: PyReadonlyArray1<f64>,
    n_entry_re: PyReadonlyArray1<f64>,
    n_entry_im: PyReadonlyArray1<f64>,
    n_exit_re: PyReadonlyArray1<f64>,
    n_exit_im: PyReadonlyArray1<f64>,
    eps: PyReadonlyArray1<f64>,
    rho: PyReadonlyArray1<f64>,
    rhop: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    thicknesses: PyReadonlyArray1<f64>,
    z_nm: PyReadonlyArray1<f64>,
    exit_eps: PyReadonlyArray1<f64>,
    exit_rho: PyReadonlyArray1<f64>,
    exit_rhop: PyReadonlyArray1<f64>,
    exit_mu: PyReadonlyArray1<f64>,
    method: i32,
) -> PyResult<Py<PyDict>> {
    fields_sweep(
        py,
        wls.as_slice()?,
        thetas.as_slice()?,
        n_entry_re.as_slice()?, 
        n_entry_im.as_slice()?,
        n_exit_re.as_slice()?,
        n_exit_im.as_slice()?,
        eps.as_slice()?,
        Some(rho.as_slice()?),
        Some(rhop.as_slice()?),
        Some(mu.as_slice()?),
        thicknesses.as_slice()?,
        z_nm.as_slice()?,
        exit_eps.as_slice()?,
        exit_rho.as_slice()?,
        exit_rhop.as_slice()?,
        exit_mu.as_slice()?,
        method,
    )
}

/// Jones (2x2 complex, flat re/im length 4) -> Mueller (4x4 real, flat 16).
#[pyfunction]
#[pyo3(name = "mueller_from_jones")]
pub fn mueller_from_jones_py(
    py: Python<'_>,
    j_re: PyReadonlyArray1<f64>,
    j_im: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray<f64, numpy::Ix2>>> {
    let re = j_re.as_slice()?;
    let im = j_im.as_slice()?;
    if re.len() != 4 || im.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "j_re and j_im must each have length 4 (row-major 2x2)",
        ));
    }
    let j: [[C; 2]; 2] = [
        [c(re[0], im[0]), c(re[1], im[1])],
        [c(re[2], im[2]), c(re[3], im[3])],
    ];
    let m = mueller_from_jones(&j);
    let mut flat = vec![0.0; 16];
    for i in 0..4 {
        for k in 0..4 {
            flat[i * 4 + k] = m[i][k].re;
        }
    }
    Ok(PyArray::from_vec(py, flat).reshape([4, 4])?.into())
}

// ── Mueller suite (scalar flat-16 post-ops over M_refl/M_trans; None → NaN) ──

fn read_m16(re: &[f64]) -> PyResult<[[f64; 4]; 4]> {
    if re.len() != 16 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Mueller matrix must have 16 elements (row-major 4x4)",
        ));
    }
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = re[i * 4 + j];
        }
    }
    Ok(m)
}

/// Scalar depolarization index; NaN when M00 <= 0.
#[pyfunction]
#[pyo3(name = "depolarization_index")]
pub fn depolarization_index_py(
    _py: Python<'_>,
    m_flat: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let m = read_m16(m_flat.as_slice()?)?;
    Ok(crate::mueller::depolarization_index(&m).unwrap_or(f64::NAN))
}

/// Scalar diattenuation (3,) / polarizance (3,) / CD — NaN triple/single.
#[pyfunction]
#[pyo3(name = "diattenuation")]
pub fn diattenuation_py(
    _py: Python<'_>,
    m_flat: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    let m = read_m16(m_flat.as_slice()?)?;
    Ok(crate::mueller::diattenuation(&m)
        .map(|v| v.to_vec())
        .unwrap_or(vec![f64::NAN; 3]))
}

#[pyfunction]
#[pyo3(name = "polarizance")]
pub fn polarizance_py(
    _py: Python<'_>,
    m_flat: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    let m = read_m16(m_flat.as_slice()?)?;
    Ok(crate::mueller::polarizance(&m)
        .map(|v| v.to_vec())
        .unwrap_or(vec![f64::NAN; 3]))
}

#[pyfunction]
#[pyo3(name = "circular_dichroism")]
pub fn circular_dichroism_py(
    _py: Python<'_>,
    m_flat: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let m = read_m16(m_flat.as_slice()?)?;
    Ok(crate::mueller::circular_dichroism(&m).unwrap_or(f64::NAN))
}

/// Cloude -> (lambda[4], entropy); NaNs when non-physical.
#[pyfunction]
#[pyo3(name = "cloude")]
pub fn cloude_py(
    py: Python<'_>,
    m_flat: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyAny>> {
    // Return a (lambda, entropy) tuple; wrapper splits into dict entries.
    let m = read_m16(m_flat.as_slice()?)?;
    let (lam, s) = match crate::mueller::cloude(&m) {
        Some(cc) => (cc.lambda.to_vec(), cc.entropy),
        None => (vec![f64::NAN; 4], f64::NAN),
    };
    use pyo3::types::PyTuple;
    Ok(
        PyTuple::new(py, [lam.into_pyobject(py)?.into_any(), s.into_pyobject(py)?.into_any()])?
            .into(),
    )
}

// ── materials (thin adapters over crate::materials; errors → ValueError) ──

fn cx1(py: Python<'_>, v: Vec<C>) -> Py<PyArray<Complex64, numpy::Ix1>> {
    PyArray::from_vec(py, v).into()
}

fn map_err(e: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e)
}

/// Flat (C-contiguous) view of a 2-D f64 array for osc/ib matrices.
fn flat2(a: PyReadonlyArray2<f64>) -> Vec<f64> {
    a.as_slice().unwrap_or(&[]).to_vec()
}

#[pyfunction]
#[pyo3(name = "materials_konstant")]
pub fn materials_konstant(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    n: f64,
    k: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(py, materials::konstant_nk(wl.as_slice().unwrap_or(&[]), n, k))
}

#[pyfunction]
#[pyo3(name = "materials_table")]
#[pyo3(signature = (wl, grid, n_vals, k_vals, n_factor, k_factor))]
pub fn materials_table(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    grid: PyReadonlyArray1<f64>,
    n_vals: PyReadonlyArray1<f64>,
    k_vals: Option<PyReadonlyArray1<f64>>,
    n_factor: f64,
    k_factor: f64,
) -> PyResult<Py<PyArray<Complex64, numpy::Ix1>>> {
    let kv: Option<Vec<f64>> = k_vals.map(|a| a.as_slice().unwrap_or(&[]).to_vec());
    materials::table_nk(
        wl.as_slice().unwrap_or(&[]),
        grid.as_slice().unwrap_or(&[]),
        n_vals.as_slice().unwrap_or(&[]),
        kv.as_deref(),
        n_factor,
        k_factor,
    )
    .map(|v| cx1(py, v))
    .map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "materials_cauchy")]
pub fn materials_cauchy(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
    c: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(py, materials::cauchy_nk(wl.as_slice().unwrap_or(&[]), a, b, c))
}

#[pyfunction]
#[pyo3(name = "materials_cauchy_urbach")]
pub fn materials_cauchy_urbach(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
    c: f64,
    alpha0: f64,
    eu: f64,
    lambda_g: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(
        py,
        materials::cauchy_urbach_nk(wl.as_slice().unwrap_or(&[]), a, b, c, alpha0, eu, lambda_g),
    )
}

#[pyfunction]
#[pyo3(name = "materials_sellmeier")]
#[pyo3(signature = (wl, b1, c1, b2, c2, b3, c3))]
pub fn materials_sellmeier(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    b1: f64,
    c1: f64,
    b2: f64,
    c2: f64,
    b3: f64,
    c3: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(
        py,
        materials::sellmeier_nk(wl.as_slice().unwrap_or(&[]), [b1, b2, b3], [c1, c2, c3]),
    )
}

#[pyfunction]
#[pyo3(name = "materials_sellmeier_urbach")]
#[pyo3(signature = (wl, b1, c1, b2, c2, b3, c3, alpha0, eu, lambda_g))]
pub fn materials_sellmeier_urbach(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    b1: f64,
    c1: f64,
    b2: f64,
    c2: f64,
    b3: f64,
    c3: f64,
    alpha0: f64,
    eu: f64,
    lambda_g: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(
        py,
        materials::sellmeier_urbach_nk(
            wl.as_slice().unwrap_or(&[]),
            [b1, b2, b3],
            [c1, c2, c3],
            alpha0,
            eu,
            lambda_g,
        ),
    )
}

#[pyfunction]
#[pyo3(name = "materials_lorentz")]
pub fn materials_lorentz(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    osc: PyReadonlyArray2<f64>,
    eps_inf: f64,
) -> PyResult<Py<PyArray<Complex64, numpy::Ix1>>> {
    materials::lorentz_nk(wl.as_slice().unwrap_or(&[]), &flat2(osc), eps_inf)
        .map(|v| cx1(py, v))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "materials_drude")]
pub fn materials_drude(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    omega_p: f64,
    gamma: f64,
    eps_inf: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(py, materials::drude_nk(wl.as_slice().unwrap_or(&[]), omega_p, gamma, eps_inf))
}

#[pyfunction]
#[pyo3(name = "materials_drude_lorentz")]
pub fn materials_drude_lorentz(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    omega_p: f64,
    gamma: f64,
    eps_inf: f64,
    osc: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray<Complex64, numpy::Ix1>>> {
    materials::drude_lorentz_nk(wl.as_slice().unwrap_or(&[]), omega_p, gamma, eps_inf, &flat2(osc))
        .map(|v| cx1(py, v))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "materials_cody_lorentz")]
pub fn materials_cody_lorentz(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    eg: f64,
    et: f64,
    eu: f64,
    osc: PyReadonlyArray2<f64>,
    eps_inf: f64,
) -> PyResult<Py<PyArray<Complex64, numpy::Ix1>>> {
    materials::cody_lorentz_nk(wl.as_slice().unwrap_or(&[]), eg, et, eu, &flat2(osc), eps_inf)
        .map(|v| cx1(py, v))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "materials_fb_interband")]
pub fn materials_fb_interband(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    n_inf: f64,
    ib: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray<Complex64, numpy::Ix1>>> {
    materials::fb_interband_nk(wl.as_slice().unwrap_or(&[]), n_inf, &flat2(ib))
        .map(|v| cx1(py, v))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "materials_fb_metal")]
pub fn materials_fb_metal(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    n_inf: f64,
    fe: PyReadonlyArray1<f64>,
    ib: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray<Complex64, numpy::Ix1>>> {
    let fe_s = fe.as_slice().unwrap_or(&[]);
    if fe_s.len() != 3 {
        return Err(map_err(format!("ForouhiBloomerMetal needs fe (A,B,C), got len {}", fe_s.len())));
    }
    materials::fb_metal_nk(wl.as_slice().unwrap_or(&[]), n_inf, [fe_s[0], fe_s[1], fe_s[2]], &flat2(ib))
        .map(|v| cx1(py, v))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "materials_tauc_lorentz")]
pub fn materials_tauc_lorentz(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    eg: f64,
    osc: PyReadonlyArray2<f64>,
    eps_inf: f64,
) -> PyResult<Py<PyArray<Complex64, numpy::Ix1>>> {
    materials::tauc_lorentz_nk(wl.as_slice().unwrap_or(&[]), eg, &flat2(osc), eps_inf)
        .map(|v| cx1(py, v))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "materials_ubf")]
pub fn materials_ubf(
    py: Python<'_>,
    wl: PyReadonlyArray1<f64>,
    osc: PyReadonlyArray2<f64>,
    eps_inf: f64,
) -> PyResult<Py<PyArray<Complex64, numpy::Ix1>>> {
    materials::ubf_nk(wl.as_slice().unwrap_or(&[]), &flat2(osc), eps_inf)
        .map(|v| cx1(py, v))
        .map_err(map_err)
}

fn cx_slice(a: PyReadonlyArray1<Complex64>) -> Vec<C> {
    a.as_slice().unwrap_or(&[]).to_vec()
}

#[pyfunction]
#[pyo3(name = "materials_ema_lichtenecker")]
pub fn materials_ema_lichtenecker(
    py: Python<'_>,
    n_i: PyReadonlyArray1<Complex64>,
    n_h: PyReadonlyArray1<Complex64>,
    f: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    // EMA returns ε; Python applies eps_to_nk (upstream order).
    cx1(py, materials::ema_lichtenecker(&cx_slice(n_i), &cx_slice(n_h), f))
}

#[pyfunction]
#[pyo3(name = "materials_ema_looyenga")]
pub fn materials_ema_looyenga(
    py: Python<'_>,
    n_i: PyReadonlyArray1<Complex64>,
    n_h: PyReadonlyArray1<Complex64>,
    f: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(py, materials::ema_looyenga(&cx_slice(n_i), &cx_slice(n_h), f))
}

#[pyfunction]
#[pyo3(name = "materials_ema_power_law")]
pub fn materials_ema_power_law(
    py: Python<'_>,
    n_i: PyReadonlyArray1<Complex64>,
    n_h: PyReadonlyArray1<Complex64>,
    f: f64,
    alpha: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(py, materials::ema_power_law(&cx_slice(n_i), &cx_slice(n_h), f, alpha))
}

#[pyfunction]
#[pyo3(name = "materials_ema_maxwell_garnett")]
pub fn materials_ema_maxwell_garnett(
    py: Python<'_>,
    n_i: PyReadonlyArray1<Complex64>,
    n_h: PyReadonlyArray1<Complex64>,
    f: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(py, materials::ema_maxwell_garnett(&cx_slice(n_i), &cx_slice(n_h), f))
}

#[pyfunction]
#[pyo3(name = "materials_ema_mori_tanaka")]
pub fn materials_ema_mori_tanaka(
    py: Python<'_>,
    n_i: PyReadonlyArray1<Complex64>,
    n_h: PyReadonlyArray1<Complex64>,
    f: f64,
    l: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(py, materials::ema_mori_tanaka(&cx_slice(n_i), &cx_slice(n_h), f, l))
}

#[pyfunction]
#[pyo3(name = "materials_ema_bruggeman")]
#[pyo3(signature = (n_i, n_h, f, max_iter, tol))]
pub fn materials_ema_bruggeman(
    py: Python<'_>,
    n_i: PyReadonlyArray1<Complex64>,
    n_h: PyReadonlyArray1<Complex64>,
    f: f64,
    max_iter: usize,
    tol: f64,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(py, materials::ema_bruggeman(&cx_slice(n_i), &cx_slice(n_h), f, max_iter, tol))
}

#[pyfunction]
#[pyo3(name = "materials_ema_roughness")]
pub fn materials_ema_roughness(
    py: Python<'_>,
    n_bottom: PyReadonlyArray1<Complex64>,
    n_top: PyReadonlyArray1<Complex64>,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(py, materials::ema_roughness(&cx_slice(n_bottom), &cx_slice(n_top)))
}

#[pyfunction]
#[pyo3(name = "materials_eps_to_nk")]
pub fn materials_eps_to_nk(
    py: Python<'_>,
    eps: PyReadonlyArray1<Complex64>,
) -> Py<PyArray<Complex64, numpy::Ix1>> {
    cx1(py, materials::eps_to_nk(&cx_slice(eps)))
}

// ── rotations (R·ε·Rᵀ; flat-9 re/im tensor layout, interleaved-18 out) ───

fn rot_apply(r: &rotations::Mat3R, t_re: &[f64], t_im: &[f64]) -> PyResult<Vec<f64>> {
    if t_re.len() != 9 || t_im.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tensor re/im must each have length 9 (row-major 3x3)",
        ));
    }
    let mut t = [[crate::cmatrix::czero(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = c(t_re[i * 3 + j], t_im[i * 3 + j]);
        }
    }
    let o = rotations::apply_rot(r, &t);
    let mut flat = vec![0.0; 18];
    for i in 0..3 {
        for j in 0..3 {
            flat[(i * 3 + j) * 2] = o[i][j].re;
            flat[(i * 3 + j) * 2 + 1] = o[i][j].im;
        }
    }
    Ok(flat)
}

#[pyfunction]
#[pyo3(name = "rot_axis_angle", signature = (ax, ay, az, theta, t_re, t_im))]
pub fn rot_axis_angle_py(
    _py: Python<'_>,
    ax: f64,
    ay: f64,
    az: f64,
    theta: f64,
    t_re: PyReadonlyArray1<f64>,
    t_im: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    let r = rotations::axis_angle([ax, ay, az], theta);
    rot_apply(&r, t_re.as_slice()?, t_im.as_slice()?)
}

#[pyfunction]
#[pyo3(name = "rot_euler", signature = (rx, ry, rz, t_re, t_im))]
pub fn rot_euler_py(
    _py: Python<'_>,
    rx: f64,
    ry: f64,
    rz: f64,
    t_re: PyReadonlyArray1<f64>,
    t_im: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    let r = rotations::euler(rx, ry, rz);
    rot_apply(&r, t_re.as_slice()?, t_im.as_slice()?)
}

#[pyfunction]
#[pyo3(name = "rot_quaternion", signature = (w, x, y, z, t_re, t_im))]
pub fn rot_quaternion_py(
    _py: Python<'_>,
    w: f64,
    x: f64,
    y: f64,
    z: f64,
    t_re: PyReadonlyArray1<f64>,
    t_im: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    let r = rotations::quaternion(w, x, y, z);
    rot_apply(&r, t_re.as_slice()?, t_im.as_slice()?)
}

// ── construction kernels (architecture review: Python keeps marshalling only) ─

fn tensor_to_interleaved(t: &Tensor3) -> Vec<f64> {
    let mut flat = vec![0.0; 18];
    for i in 0..3 {
        for j in 0..3 {
            flat[(i * 3 + j) * 2] = t[i][j].re;
            flat[(i * 3 + j) * 2 + 1] = t[i][j].im;
        }
    }
    flat
}

fn read_flat_tensor(re: &[f64], im: &[f64]) -> PyResult<Tensor3> {
    if re.len() != 9 || im.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tensor re/im must each have length 9 (row-major 3x3)",
        ));
    }
    let mut t = [[c(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = c(re[i * 3 + j], im[i * 3 + j]);
        }
    }
    Ok(t)
}

/// Route-2 graded-tensor kernel (Gaussian CDF profile + volume-weighted
/// mixing). Returns n_sublayers interleaved-18 tensors, or None for the
/// Python-side "no roughness -> no sublayers" contract (sigma <= 0 is a
/// Python semantic: empty list, not an error).
#[pyfunction]
#[pyo3(name = "grade_interface_tensors")]
#[pyo3(signature = (a_re, a_im, b_re, b_im, sigma_nm, n_sublayers, total_width_nm))]
pub fn grade_interface_tensors(
    _py: Python<'_>,
    a_re: PyReadonlyArray1<f64>,
    a_im: PyReadonlyArray1<f64>,
    b_re: PyReadonlyArray1<f64>,
    b_im: PyReadonlyArray1<f64>,
    sigma_nm: f64,
    n_sublayers: usize,
    total_width_nm: Option<f64>,
) -> PyResult<Option<Vec<f64>>> {
    let ea = read_flat_tensor(a_re.as_slice()?, a_im.as_slice()?)?;
    let eb = read_flat_tensor(b_re.as_slice()?, b_im.as_slice()?)?;
    match roughness::graded_tensors(&ea, &eb, sigma_nm, n_sublayers, total_width_nm) {
        Ok(ts) => {
            let mut out = Vec::with_capacity(ts.len() * 18);
            for t in &ts {
                out.extend(tensor_to_interleaved(t));
            }
            Ok(Some(out))
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
    }
}

/// Twist-schedule kernel (grid: 0 = midpoint, 1 = endpoint).
#[pyfunction]
#[pyo3(name = "twisted_tensors")]
#[pyo3(signature = (e_re, e_im, twist_rad, n_slices, grid_code))]
pub fn twisted_tensors(
    _py: Python<'_>,
    e_re: PyReadonlyArray1<f64>,
    e_im: PyReadonlyArray1<f64>,
    twist_rad: f64,
    n_slices: usize,
    grid_code: i32,
) -> PyResult<Option<Vec<f64>>> {
    let e0 = read_flat_tensor(e_re.as_slice()?, e_im.as_slice()?)?;
    let grid = match grid_code {
        0 => rotations::TwistGrid::Midpoint,
        1 => rotations::TwistGrid::Endpoint,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "grid_code must be 0 (midpoint) or 1 (endpoint)",
            ))
        }
    };
    match rotations::twisted_tensors(&e0, twist_rad, n_slices, grid) {
        Ok(ts) => {
            let mut out = Vec::with_capacity(ts.len() * 18);
            for t in &ts {
                out.extend(tensor_to_interleaved(t));
            }
            Ok(Some(out))
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
    }
}

/// The Pasteur–Tellegen mapping (eps, rho, rhop, mu), each flat interleaved-18.
#[pyfunction]
#[pyo3(name = "pasteur_tensors")]
pub fn pasteur_tensors(
    _py: Python<'_>,
    n: f64,
    kappa: f64,
    mu: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let [eps, rho, rhop, mu_t] = crate::berreman::pasteur_tensors(n, kappa, mu)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((
        tensor_to_interleaved(&eps),
        tensor_to_interleaved(&rho),
        tensor_to_interleaved(&rhop),
        tensor_to_interleaved(&mu_t),
    ))
}

/// Batched user-matrix apply R·ε·Rᵀ over n tensors (real 3x3 R). Used by
/// evaluate_tensor(rotate=...) so the tensor contraction stays in Rust.
#[pyfunction]
#[pyo3(name = "rot_apply_matrix")]
pub fn rot_apply_matrix(
    _py: Python<'_>,
    r_re: PyReadonlyArray1<f64>,
    r_im: PyReadonlyArray1<f64>,
    t_re: PyReadonlyArray1<f64>,
    t_im: PyReadonlyArray1<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let rr = read_flat_tensor(r_re.as_slice()?, r_im.as_slice()?)?;
    let r = [[rr[0][0].re, rr[0][1].re, rr[0][2].re],
             [rr[1][0].re, rr[1][1].re, rr[1][2].re],
             [rr[2][0].re, rr[2][1].re, rr[2][2].re]];
    let tre = t_re.as_slice()?;
    let tim = t_im.as_slice()?;
    if tre.len() != tim.len() || tre.len() % 9 != 0 || tre.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "tensor re/im must each have length 9*n (n >= 1), got {} / {}",
            tre.len(),
            tim.len()
        )));
    }
    let n = tre.len() / 9;
    let mut ts = Vec::with_capacity(n);
    for w in 0..n {
        ts.push(read_flat_tensor(&tre[w * 9..(w + 1) * 9], &tim[w * 9..(w + 1) * 9])?);
    }
    let out = rotations::apply_rot_batch(&r, &ts);
    let mut ore = Vec::with_capacity(n * 9);
    let mut oim = Vec::with_capacity(n * 9);
    for t in &out {
        for i in 0..3 {
            for j in 0..3 {
                ore.push(t[i][j].re);
                oim.push(t[i][j].im);
            }
        }
    }
    Ok((ore, oim))
}
