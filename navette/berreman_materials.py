# SPDX-License-Identifier: LGPL-3.0-or-later
"""navette.berreman_materials — dispersion models over the Rust core (upstream kernels).

NOTE: named ``berreman_materials`` (not ``materials``) because the upstream
wheel already owns ``navette.materials`` and its portion shadows ours in the
merged namespace — verified during implementation. Same spec vocabulary as
upstream, plus the new tensor layer.

Thin layer over the compiled ``navette._berreman`` materials bindings, which
are themselves thin adapters over the upstream ``navette`` crate (crates.io,
0.7.0) free functions — depend, don't port (IMPLEMENTATION_PLAN.md Phase 9).
Parameter vocabulary (model names + params keys + defaults) is intentionally
identical to upstream ``navette.materials.evaluate`` so specs are
cross-compatible; G14a pins live parity.

Split of labor (mirrors the upstream native/thin-Python shape):
kernels in Rust (``src/materials.rs`` + ``pybind.rs``); dispatch, defaults,
validation and nesting here. Tensor assembly (``evaluate_tensor``) is new —
upstream is scalar-only — and lives here in exact numpy arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from ._berreman import (
        materials_konstant as _mk_konstant,
        materials_table as _mk_table,
        materials_cauchy as _mk_cauchy,
        materials_cauchy_urbach as _mk_cauchy_urbach,
        materials_sellmeier as _mk_sellmeier,
        materials_sellmeier_urbach as _mk_sellmeier_urbach,
        materials_lorentz as _mk_lorentz,
        materials_drude as _mk_drude,
        materials_drude_lorentz as _mk_drude_lorentz,
        materials_cody_lorentz as _mk_cody,
        materials_fb_interband as _mk_fb_ib,
        materials_fb_metal as _mk_fb_metal,
        materials_tauc_lorentz as _mk_tauc,
        materials_ubf as _mk_ubf,
        materials_ema_lichtenecker as _mk_licht,
        materials_ema_looyenga as _mk_looy,
        materials_ema_power_law as _mk_plaw,
        materials_ema_maxwell_garnett as _mk_mg,
        materials_ema_mori_tanaka as _mk_mt,
        materials_ema_bruggeman as _mk_bru,
        materials_ema_roughness as _mk_rough,
        materials_eps_to_nk as _mk_eps_to_nk,
        rot_apply_matrix as _rs_rot_apply_matrix,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import the compiled `navette._berreman` materials bindings. "
        "Rebuild with: VIRTUAL_ENV=<venv> maturin develop --release"
    ) from exc

MODELS: Tuple[str, ...] = (
    "Konstant",
    "Table",
    "Cauchy",
    "CauchyUrbach",
    "Sellmeier",
    "SellmeierUrbach",
    "Lorentz",
    "Drude",
    "DrudeLorentz",
    "CodyLorentz",
    "ForouhiBloomerSingle",
    "ForouhiBloomerMulti",
    "ForouhiBloomerMetal",
    "ForouhiBloomerMetal2021",
    "TaucLorentz",
    "UBF",
    "Bruggeman",
    "MaxwellGarnett",
    "Looyenga",
    "Lichtenecker",
    "MoriTanaka",
    "PowerLaw",
    "Roughness",
)


@dataclass(frozen=True)
class MaterialSpec:
    """A material as data: ``model`` name plus plain ``params``.

    Oscillator conventions (match upstream, and the native layouts):

    - Lorentz / DrudeLorentz ``osc``: ``(E0, Gamma, f0)`` (N,3)
    - CodyLorentz ``osc``: ``(E0, A, Gamma, Ep)`` (N,4)
    - TaucLorentz ``osc``: ``(A, E0, C)`` (N,3)
    - UBF ``osc``: dicts ``{Eg, Ec, Eu, A, Gamma, gamma}``
    - Forouhi-Bloomer ``ib``: ``(Eg, A, B, C)`` (N,4); metal ``fe``: ``(A,B,C)``
    - EMA composites nest specs: ``host`` / ``inclusion`` are MaterialSpec
    - Table: ``n_data`` / ``k_data`` as ``(wavelengths, values)`` pairs
    """

    model: str
    params: Dict[str, Any] = field(default_factory=dict)


def _as_osc_array(osc, width: int, what: str) -> np.ndarray:
    arr = np.asarray(osc, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{what} needs an (N, {width}) array, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{what} needs at least one oscillator")
    return np.ascontiguousarray(arr)


def _ubf_array(oscs: Sequence[Mapping[str, float]]) -> np.ndarray:
    rows = []
    for i, o in enumerate(oscs):
        try:
            eu = float(o["Eu"])
            rows.append([
                float(o["Eg"]),
                float(o["Ec"]),
                1.0 / eu,  # beta = 1/Eu kernel convention (upstream)
                float(o["A"]),
                float(o["Gamma"]),
                float(o["gamma"]),
            ])
        except KeyError as exc:
            raise ValueError(f"UBF oscillator {i} missing key {exc}") from exc
        if eu <= 0:
            raise ValueError(f"UBF oscillator {i}: Eu must be > 0")
    return np.ascontiguousarray(rows, dtype=np.float64)


def _eval_nested(spec, wl: np.ndarray, what: str) -> np.ndarray:
    if spec is None:
        raise ValueError(f"EMA/Roughness composite missing '{what}' spec")
    return evaluate(spec, wl)


def evaluate(spec: MaterialSpec | Mapping[str, Any], wavelength_nm) -> np.ndarray:
    """Evaluate a material spec to complex ``n + ik`` on ``wavelength_nm``.

    Parameter-for-parameter compatible with upstream
    ``navette.materials.evaluate`` (same model names, keys, defaults).
    """
    if isinstance(spec, Mapping):
        spec = MaterialSpec(model=spec["model"], params=dict(spec.get("params", {})))
    wl = np.ascontiguousarray(np.asarray(wavelength_nm, dtype=np.float64))
    if wl.ndim != 1 or wl.size == 0:
        raise ValueError("wavelength_nm must be a non-empty 1-D array")
    p = spec.params
    get = p.get

    def req(*names: str) -> List[float]:
        missing = [n for n in names if n not in p]
        if missing:
            raise ValueError(f"{spec.model} missing params: {missing}")
        return [float(p[n]) for n in names]

    model = spec.model
    if model == "Konstant":
        n = float(req("n")[0])
        k = float(p.get("k", 0.0))
        return np.ascontiguousarray(_mk_konstant(wl, n, k))
    if model == "Table":
        n_data = p.get("n_data")
        if n_data is None:
            raise ValueError("Table missing 'n_data' ((wavelengths, values) pair)")
        gw, nv = (np.asarray(a, dtype=np.float64) for a in n_data)
        k_data = p.get("k_data")
        kv = None
        if k_data is not None:
            _, kv = (np.asarray(a, dtype=np.float64) for a in k_data)
            kv = np.ascontiguousarray(kv)
        for key in ("interpolation_type_n", "interpolation_type_k"):
            if key in p and p[key] != "linear":
                raise ValueError(
                    f"Table {key}={p[key]!r} unsupported: native core is linear-only, "
                    "resample the table offline"
                )
        return np.ascontiguousarray(
            _mk_table(wl, np.ascontiguousarray(gw), np.ascontiguousarray(nv), kv,
                      float(p.get("n_factor", 1.0)), float(p.get("k_factor", 1.0)))
        )
    if model == "Cauchy":
        a, b, c = req("A", "B", "C")
        return np.ascontiguousarray(_mk_cauchy(wl, a, b, c))
    if model == "CauchyUrbach":
        a, b, c, alpha0, eu, lg = req("A", "B", "C", "alpha0", "Eu", "lambda_g")
        return np.ascontiguousarray(_mk_cauchy_urbach(wl, a, b, c, alpha0, eu, lg))
    if model == "Sellmeier":
        b1, c1, b2, c2, b3, c3 = req("B1", "C1", "B2", "C2", "B3", "C3")
        return np.ascontiguousarray(_mk_sellmeier(wl, b1, c1, b2, c2, b3, c3))
    if model == "SellmeierUrbach":
        vals = req("B1", "C1", "B2", "C2", "B3", "C3", "alpha0", "Eu", "lambda_g")
        return np.ascontiguousarray(_mk_sellmeier_urbach(wl, *vals))
    if model == "Lorentz":
        osc = _as_osc_array(p.get("osc", []), 3, "Lorentz osc")
        return np.ascontiguousarray(_mk_lorentz(wl, osc, float(get("epsilon_inf", 1.0))))
    if model == "Drude":
        wp, gamma, eps = req("omega_p", "gamma", "epsilon_inf")
        return np.ascontiguousarray(_mk_drude(wl, wp, gamma, eps))
    if model == "DrudeLorentz":
        wp = float(p.get("omega_p", p.get("wp", 0.0)))
        gamma = float(p.get("gamma_drude", p.get("gamma", 0.0)))
        eps = float(get("epsilon_inf", 1.0))
        osc = _as_osc_array(p.get("osc", []), 3, "DrudeLorentz osc")
        return np.ascontiguousarray(_mk_drude_lorentz(wl, wp, gamma, eps, osc))
    if model == "CodyLorentz":
        eg, et, eu = req("Eg", "Et", "Eu")
        osc = _as_osc_array(p.get("osc", []), 4, "CodyLorentz osc")
        return np.ascontiguousarray(_mk_cody(wl, eg, et, eu, osc, float(get("epsilon_inf", 1.0))))
    if model in ("ForouhiBloomerSingle", "ForouhiBloomerMulti"):
        ib = _as_osc_array(p.get("ib", []), 4, "ForouhiBloomer ib")
        return np.ascontiguousarray(_mk_fb_ib(wl, float(get("n_inf", 1.0)), ib))
    if model in ("ForouhiBloomerMetal", "ForouhiBloomerMetal2021"):
        ib = _as_osc_array(p.get("ib", []), 4, "ForouhiBloomer ib")
        if "A_fe" in p:
            fe = np.ascontiguousarray(
                [float(p["A_fe"]), float(p["B_fe"]), float(p["C_fe"])], dtype=np.float64
            )
        else:
            fe = np.ascontiguousarray(p.get("fe", []), dtype=np.float64)
        if fe.shape != (3,):
            raise ValueError("ForouhiBloomerMetal needs fe (A_fe, B_fe, C_fe)")
        return np.ascontiguousarray(_mk_fb_metal(wl, float(get("n_inf", 1.0)), fe, ib))
    if model == "TaucLorentz":
        eg = float(req("Eg")[0])
        osc = _as_osc_array(p.get("osc", []), 3, "TaucLorentz osc")
        return np.ascontiguousarray(_mk_tauc(wl, eg, osc, float(get("epsilon_inf", 1.0))))
    if model == "UBF":
        osc = _ubf_array(p.get("osc", []))
        return np.ascontiguousarray(_mk_ubf(wl, osc, float(get("epsilon_inf", 1.0))))
    if model in ("Bruggeman", "MaxwellGarnett", "Looyenga", "Lichtenecker",
                 "MoriTanaka", "PowerLaw"):
        host = _eval_nested(p.get("host"), wl, "host")
        incl = _eval_nested(p.get("inclusion"), wl, "inclusion")
        f = float(req("fraction")[0])
        if model == "Bruggeman":
            eps = _mk_bru(incl, host, f, int(p.get("max_iter", 100)), float(p.get("tol", 1e-9)))
        elif model == "MaxwellGarnett":
            eps = _mk_mg(incl, host, f)
        elif model == "Looyenga":
            eps = _mk_looy(incl, host, f)
        elif model == "Lichtenecker":
            eps = _mk_licht(incl, host, f)
        elif model == "MoriTanaka":
            eps = _mk_mt(incl, host, f, float(p.get("L", 1.0 / 3.0)))
        else:
            eps = _mk_plaw(incl, host, f, float(p.get("alpha", 0.5)))
        return np.ascontiguousarray(_mk_eps_to_nk(np.ascontiguousarray(eps)))
    if model == "Roughness":
        bottom = _eval_nested(p.get("bottom"), wl, "bottom")
        top = _eval_nested(p.get("top"), wl, "top")
        return np.ascontiguousarray(_mk_eps_to_nk(np.ascontiguousarray(_mk_rough(bottom, top))))
    raise ValueError(f"Unknown material model {model!r}. Available: {MODELS}")


def _axis_nk(axis, wl: np.ndarray) -> np.ndarray:
    if isinstance(axis, MaterialSpec) or (isinstance(axis, Mapping) and "model" in axis):
        return evaluate(axis, wl)
    raise ValueError(
        "tensor axis must be a MaterialSpec (or {'model':..., 'params':...} mapping)"
    )


def evaluate_tensor(spec, wavelength_nm, rotate: Optional[np.ndarray] = None) -> np.ndarray:
    """Birefringent constructor over :func:`evaluate`.

    ``spec`` is either a single (isotropic) spec or one of::

        {"uniaxial": {"ordinary": spec_o, "extraordinary": spec_e}}
        {"biaxial": {"x": spec_x, "y": spec_y, "z": spec_z}}

    Returns ``(n_wl, 3, 3)`` complex128 diagonal ``eps = n̂²`` — ready for
    ``bl.Layer`` / ``graded_stack`` (which accept ``(n_wl,3,3)``). EMA
    composites stay scalar per axis (per-axis mixing is the defined
    semantic); ``Roughness(bottom, top)`` maps to the 50:50 Looyenga
    ``roughness_interface`` per axis. Orientation is applied downstream:
    pass ``rotate`` as a REAL 3×3 orientation matrix for ``R·ε·Rᵀ`` per
    wavelength (applied in Rust via ``apply_rot_batch``), or rotate
    afterwards with the ``rot`` helpers; materials never rotate internally.
    """
    wl = np.ascontiguousarray(np.asarray(wavelength_nm, dtype=np.float64))
    if isinstance(spec, MaterialSpec) or (isinstance(spec, Mapping) and "model" in spec):
        n = _axis_nk(spec, wl)
        eps_diag = n * n
        out = np.zeros((wl.size, 3, 3), dtype=np.complex128)
        for a in range(3):
            out[:, a, a] = eps_diag
    elif isinstance(spec, Mapping) and "uniaxial" in spec:
        u = spec["uniaxial"]
        no = _axis_nk(u["ordinary"], wl)
        ne = _axis_nk(u["extraordinary"], wl)
        out = np.zeros((wl.size, 3, 3), dtype=np.complex128)
        out[:, 0, 0] = no * no
        out[:, 1, 1] = no * no
        out[:, 2, 2] = ne * ne
    elif isinstance(spec, Mapping) and "biaxial" in spec:
        b = spec["biaxial"]
        nx = _axis_nk(b["x"], wl)
        ny = _axis_nk(b["y"], wl)
        nz = _axis_nk(b["z"], wl)
        out = np.zeros((wl.size, 3, 3), dtype=np.complex128)
        out[:, 0, 0] = nx * nx
        out[:, 1, 1] = ny * ny
        out[:, 2, 2] = nz * nz
    else:
        raise ValueError(
            "evaluate_tensor needs a MaterialSpec or "
            "{'uniaxial': {'ordinary':.., 'extraordinary':..}} / "
            "{'biaxial': {'x':.., 'y':.., 'z':..}}"
        )
    if rotate is not None:
        # R·ε·Rᵀ per wavelength — delegated to Rust (rotations::apply_rot_batch
        # via rot_apply_matrix) so the tensor contraction stays out of Python.
        r = np.asarray(rotate)
        if r.shape != (3, 3):
            raise ValueError(f"rotate must be (3,3), got {r.shape}")
        if np.any(np.asarray(r, dtype=complex).imag != 0):
            raise ValueError(
                "rotate must be a REAL 3x3 orientation matrix "
                "(R·ε·Rᵀ; the wrapper applies it in Rust, which is real-only)")
        r = np.ascontiguousarray(r, dtype=np.float64)
        n = wl.size
        tre = np.ascontiguousarray(out.real.reshape(n * 9))
        tim = np.ascontiguousarray(out.imag.reshape(n * 9))
        ore, oim = _rs_rot_apply_matrix(
            r.reshape(9).copy(), np.zeros(9), tre, tim)
        out = (np.asarray(ore, dtype=float).reshape(n, 3, 3)
               + 1j * np.asarray(oim, dtype=float).reshape(n, 3, 3))
    return out
