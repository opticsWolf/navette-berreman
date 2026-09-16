// SPDX-License-Identifier: LGPL-3.0-or-later
//! navette-berreman — a Rust port of the 4x4 Berreman/Mueller optical solver for
//! birefringent multilayer systems, styled after the `_smatrix` crate.
//!
//! The physics core (`cmatrix`, `berreman`, `transfer`) is pure Rust over
//! `num_complex` and builds/tests without Python. The optional `python`
//! feature adds PyO3 bindings (`pybind`) that expose the same functionality as
//! a compiled extension module, including a Rayon-parallel spectrum sweep.

pub mod cmatrix;
pub mod berreman;
pub mod expm;
pub mod fields;
pub mod materials;
pub mod mueller;
pub mod rotations;
pub mod roughness;
pub mod transfer;

#[cfg(feature = "python")]
pub mod pybind;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// The compiled extension module. Mirrors the `_smatrix` registration style.
#[cfg(feature = "python")]
#[pymodule]
fn _berreman(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pybind::solve_grid_simple, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::solve_grid_full, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::fields_grid_simple, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::fields_grid_full, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::mueller_from_jones_py, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::depolarization_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::diattenuation_py, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::polarizance_py, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::circular_dichroism_py, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::cloude_py, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_konstant, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_table, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_cauchy, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_cauchy_urbach, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_sellmeier, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_sellmeier_urbach, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_lorentz, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_drude, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_drude_lorentz, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_cody_lorentz, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_fb_interband, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_fb_metal, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_tauc_lorentz, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_ubf, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_ema_lichtenecker, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_ema_looyenga, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_ema_power_law, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_ema_maxwell_garnett, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_ema_mori_tanaka, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_ema_bruggeman, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_ema_roughness, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::materials_eps_to_nk, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::rot_axis_angle_py, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::rot_euler_py, m)?)?;
    m.add_function(wrap_pyfunction!(pybind::rot_quaternion_py, m)?)?;
    Ok(())
}
