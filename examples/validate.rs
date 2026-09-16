// SPDX-License-Identifier: LGPL-3.0-or-later
// Validation harness: solves the same canonical stacks as ref/ref_pyllama.py
// with both methods and prints JSON to stdout for a tolerance diff.

use _berreman::berreman::{diag_eps, Tensor3};
use _berreman::cmatrix::{c, czero, C};
use _berreman::transfer::{solve_stack, Geometry, LayerSpec, Method, SolveResult};

fn rot_z(eps: &Tensor3, angle: f64) -> Tensor3 {
    let cphi = c(angle.cos(), 0.0);
    let sphi = c(angle.sin(), 0.0);
    // Rz = [[c,-s,0],[s,c,0],[0,0,1]]; out = Rz eps Rz^T
    let rz = [
        [cphi, -sphi, czero()],
        [sphi, cphi, czero()],
        [czero(), czero(), c(1.0, 0.0)],
    ];
    let rzt = [
        [cphi, sphi, czero()],
        [-sphi, cphi, czero()],
        [czero(), czero(), c(1.0, 0.0)],
    ];
    let mut tmp = [[czero(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = czero();
            for k in 0..3 {
                s += rz[i][k] * eps[k][j];
            }
            tmp[i][j] = s;
        }
    }
    let mut out = [[czero(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = czero();
            for k in 0..3 {
                s += tmp[i][k] * rzt[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

fn cj(z: C) -> String {
    format!("[{},{}]", z.re, z.im)
}

fn mat2_json(m: &[[C; 2]; 2]) -> String {
    format!(
        "[[{},{}],[{},{}]]",
        cj(m[0][0]),
        cj(m[0][1]),
        cj(m[1][0]),
        cj(m[1][1])
    )
}

fn real2_json(m: &[[C; 2]; 2]) -> String {
    format!(
        "[[{},{}],[{},{}]]",
        m[0][0].re, m[0][1].re, m[1][0].re, m[1][1].re
    )
}

fn emit(res: &SolveResult) -> String {
    format!(
        "{{\"J_refl\":{},\"J_trans\":{},\"R\":{},\"T\":{},\"J_refl_c\":{}}}",
        mat2_json(&res.jones.refl),
        mat2_json(&res.jones.trans),
        real2_json(&res.r_power),
        real2_json(&res.t_power),
        mat2_json(&res.jones_circ.refl),
    )
}

struct Case {
    name: &'static str,
    layers: Vec<LayerSpec>,
    eps_entry: f64,
    eps_exit: f64,
    wl: f64,
    theta: f64,
}

fn cases() -> Vec<Case> {
    let r = |d: f64| d.to_radians();
    let layer = |eps: Tensor3, t: f64| LayerSpec {
        eps,
        thickness_nm: t,
        full: None,
        front_roughness: (0, 0.0),
    };
    vec![
        Case {
            name: "iso_normal",
            layers: vec![layer(diag_eps(c(2.25, 0.0), c(2.25, 0.0), c(2.25, 0.0)), 200.0)],
            eps_entry: 1.0,
            eps_exit: 1.0,
            wl: 550.0,
            theta: 0.0,
        },
        Case {
            name: "uniaxial_normal",
            layers: vec![layer(diag_eps(c(2.25, 0.0), c(2.56, 0.0), c(2.25, 0.0)), 300.0)],
            eps_entry: 1.0,
            eps_exit: 1.0,
            wl: 550.0,
            theta: 0.0,
        },
        Case {
            name: "rot_uniaxial_oblique",
            layers: vec![layer(
                rot_z(&diag_eps(c(2.25, 0.0), c(2.89, 0.0), c(2.25, 0.0)), r(35.0)),
                250.0,
            )],
            eps_entry: 1.0,
            eps_exit: 1.0,
            wl: 633.0,
            theta: r(30.0),
        },
        Case {
            name: "two_layer_oblique",
            layers: vec![
                layer(diag_eps(c(2.56, 0.0), c(2.56, 0.0), c(2.56, 0.0)), 120.0),
                layer(
                    rot_z(&diag_eps(c(2.1, 0.0), c(2.7, 0.0), c(2.1, 0.0)), r(50.0)),
                    180.0,
                ),
            ],
            eps_entry: 1.0,
            eps_exit: 2.25,
            wl: 500.0,
            theta: r(20.0),
        },
        Case {
            name: "absorbing_rot_oblique",
            layers: vec![layer(
                rot_z(
                    &diag_eps(c(2.25, 0.05), c(3.0, 0.2), c(2.25, 0.05)),
                    r(25.0),
                ),
                220.0,
            )],
            eps_entry: 1.0,
            eps_exit: 1.0,
            wl: 600.0,
            theta: r(40.0),
        },
    ]
}

fn main() {
    let cs = cases();
    let mut out = String::from("{");
    for (ci, case) in cs.iter().enumerate() {
        let geom = Geometry::isotropic(
            case.wl,
            case.theta,
            c(case.eps_entry.sqrt(), 0.0),
            c(case.eps_exit.sqrt(), 0.0),
            (0, 0.0),
        );
        let sm = solve_stack(&geom, &case.layers, Method::Scattering).unwrap();
        let tm = solve_stack(&geom, &case.layers, Method::Transfer).unwrap();
        if ci > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "\"{}\":{{\"SM\":{},\"TM\":{}}}",
            case.name,
            emit(&sm),
            emit(&tm)
        ));
    }
    out.push('}');
    println!("{}", out);
}
