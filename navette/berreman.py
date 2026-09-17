# SPDX-License-Identifier: LGPL-3.0-or-later
"""navette.berreman — a clean Python interface to the ``_berreman`` Rust extension.

A thin, self-contained wrapper around the compiled Rust Berreman/Mueller core.
It owns no physics: every number comes from the Rust ``solve_grid_*`` sweep. The
wrapper's job is to

  * hold a birefringent layer stack + a (wavelength, angle) grid,
  * marshal them into the exact flat array layouts the Rust functions expect,
  * return a dict of named, correctly-shaped results (Jones, R/T, Mueller),
  * optionally expose the full magneto-optic (rho, rho', mu) Berreman matrix.

Conventions (faithful to pyllama)
---------------------------------
* The field 4-vector is ψ = [E_x, H_y, E_y, -H_x].
* ``eps`` per layer is a 3x3 complex tensor; pass shape ``(3, 3)`` for a
  non-dispersive layer or ``(n_wl, 3, 3)`` for a dispersive one.
* ``n_entry`` / ``n_exit`` are complex refractive indices, scalar or length
  ``n_wl``.
* Angles are in degrees by default (``angles_in_radians=True`` to override).
* The reduced ("simple") Berreman matrix is used unless a layer supplies
  ``rho``/``rhop``/``mu`` (then the full matrix is used for the whole stack).
* Circular-polarization convention follows pyllama: the columns of the linear
  Jones matrices are ordered (p, s); ``J_*_c`` are in the (L, R) circular basis
  via F = [[1, 1], [-1j, 1j]], B = [[1, 1], [1j, -1j]].
* Results are shaped ``[n_wl, n_angles, ...]``; singleton axes are squeezed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

try:
    from ._berreman import (
        solve_grid_simple as _rs_solve_simple,
        solve_grid_full as _rs_solve_full,
        mueller_from_jones as _rs_mueller_from_jones,
        rot_axis_angle as _rs_rot_aa,
        rot_euler as _rs_rot_eu,
        rot_quaternion as _rs_rot_q,
        fields_grid_simple as _rs_fields_simple,
        fields_grid_full as _rs_fields_full,
        depolarization_index as _rs_di,
        diattenuation as _rs_diat,
        polarizance as _rs_pol,
        circular_dichroism as _rs_cd,
        cloude as _rs_cloude,
        grade_interface_tensors as _rs_grade_tensors,
        twisted_tensors as _rs_twisted_tensors,
        pasteur_tensors as _rs_pasteur_tensors,
        rot_apply_matrix as _rs_rot_apply_matrix,
    )
except ImportError:  # pragma: no cover - allow flat-module import too
    from _berreman import (
        solve_grid_simple as _rs_solve_simple,
        solve_grid_full as _rs_solve_full,
        mueller_from_jones as _rs_mueller_from_jones,
        rot_axis_angle as _rs_rot_aa,
        rot_euler as _rs_rot_eu,
        rot_quaternion as _rs_rot_q,
        fields_grid_simple as _rs_fields_simple,
        fields_grid_full as _rs_fields_full,
        depolarization_index as _rs_di,
        diattenuation as _rs_diat,
        polarizance as _rs_pol,
        circular_dichroism as _rs_cd,
        cloude as _rs_cloude,
        grade_interface_tensors as _rs_grade_tensors,
        twisted_tensors as _rs_twisted_tensors,
        pasteur_tensors as _rs_pasteur_tensors,
        rot_apply_matrix as _rs_rot_apply_matrix,
    )

_METHOD = {"scattering": 0, "sm": 0, "transfer": 1, "tm": 1,
           "exponential": 2, "em": 2, "exp": 2}


def _lookup_method(name: str) -> int:
    """Method-name lookup with a ValueError (not a bare KeyError) on typos."""
    try:
        return _METHOD[name.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown method {name!r}; valid: "
            "'scattering'/'sm', 'transfer'/'tm', 'exponential'/'em'/'exp'") from None


@dataclass
class Layer:
    """One birefringent layer.

    Parameters
    ----------
    eps : array_like
        Permittivity tensor, shape ``(3, 3)`` (non-dispersive) or
        ``(n_wl, 3, 3)`` (dispersive), complex.
    thickness_nm : float
        Physical thickness in nanometres.
    rho, rhop, mu : array_like, optional
        Magneto-electric (ρ, ρ') and magnetic (μ) tensors, same shape rules as
        ``eps``. If any layer in a stack supplies these, the full Berreman
        matrix is used for the whole stack.
    roughness : (int, float), optional
        Specular surface roughness ``(rough_type, sigma_nm)`` of the interface
        at the *front* of this layer (between the previous medium and this
        layer). ``rough_type`` matches the smatrix integer codes:
        0 none, 1 uniform/box, 2 two-delta, 3 Lorentzian, 4 Gaussian
        (Debye–Waller), 5 Névot–Croce. Roughness requires ``method="scattering"``.
    """

    eps: np.ndarray
    thickness_nm: float
    rho: Optional[np.ndarray] = None
    rhop: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    roughness: Optional[tuple] = None


# smatrix-compatible roughness type codes.
ROUGH_NONE, ROUGH_UNIFORM, ROUGH_TWODELTA, ROUGH_LORENTZIAN, ROUGH_GAUSSIAN, ROUGH_NEVOT_CROCE = range(6)

_ROUGH_TYPES = frozenset((0, 1, 2, 3, 4, 5))


def _norm_rough(spec, where):
    """Validate + normalize one (type, sigma_nm) spec. None/degenerate → None."""
    if spec is None:
        return None
    t, s = int(spec[0]), float(spec[1])
    if t not in _ROUGH_TYPES:
        raise ValueError(f"{where}: unknown roughness type {t} (valid 0–5)")
    if not np.isfinite(s) or s < 0:
        raise ValueError(f"{where}: sigma_nm must be finite and >= 0, got {spec[1]}")
    if t == 0 or s == 0.0:
        return None  # smooth: Rust identity either way; unblocks transfer
    return (t, s)


class BerremanStack:
    """A multilayer stack plus a (wavelength, angle) grid, solved in Rust.

    ``periods=N`` repeats the layer list N times (unit cell; ``periods=1`` is
    a plain stack). Identical to passing ``layers * N`` — including
    roughness: each cell boundary is dressed like the expanded stack
    (cell[0] front at entry + every wrap, internal fronts per copy).
    """

    def __init__(
        self,
        layers: Sequence[Layer],
        wavelengths_nm: Sequence[float],
        angles: Sequence[float],
        n_entry: Union[complex, Sequence[complex]] = 1.0,
        n_exit: Union[complex, Sequence[complex]] = 1.0,
        method: str = "scattering",
        angles_in_radians: bool = False,
        exit_roughness: Optional[tuple] = None,
        exit_eps=None, exit_rho=None, exit_rhop=None, exit_mu=None,
        periods: int = 1,
    ):
        self.layers = list(layers)
        self.wl = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
        ang = np.atleast_1d(np.asarray(angles, dtype=float))
        self.theta = ang if angles_in_radians else np.radians(ang)
        self.n_wl = self.wl.size
        self.n_theta = self.theta.size
        self.method = _lookup_method(method)
        self.n_entry = self._broadcast_index(n_entry)
        self.n_exit = self._broadcast_index(n_exit)
        self.exit_roughness = exit_roughness
        self.exit_eps = self._tensor_per_wl_opt(exit_eps, "exit_eps")
        self.exit_rho = self._tensor_per_wl_opt(exit_rho, "exit_rho")
        self.exit_rhop = self._tensor_per_wl_opt(exit_rhop, "exit_rhop")
        self.exit_mu = self._tensor_per_wl_opt(exit_mu, "exit_mu")
        mo_given = [x is not None for x in (exit_rho, exit_rhop, exit_mu)]
        if any(mo_given) and not all(mo_given):
            raise ValueError("exit_rho/rhop/mu must be given together")
        if exit_eps is None and any(mo_given):
            raise ValueError("exit MO tensors need exit_eps")
        if exit_eps is not None and not np.all(
                np.asarray(n_exit, dtype=complex) == 1.0):
            raise ValueError("exit_eps and a non-unity n_exit are mutually "
                             "exclusive (pass n_exit=1.0 with exit_eps)")
        if not isinstance(periods, (int, np.integer)) or int(periods) < 1:
            raise ValueError("periods must be an integer >= 1")
        self.periods = int(periods)
        self._use_full = any(
            l.rho is not None or l.rhop is not None or l.mu is not None
            for l in self.layers
        )
        # normalized per-layer fronts + exit (parallel arrays); degenerate
        # (0,*)/(*,0) specs become None so the method gate sees real dressing only
        self._rough_norm = [_norm_rough(l.roughness, f"layers[{i}].roughness")
                            for i, l in enumerate(self.layers)]
        self._exit_rough_norm = _norm_rough(exit_roughness, "exit_roughness")
        self._has_rough = self._exit_rough_norm is not None or \
            any(r is not None for r in self._rough_norm)
        if self._has_rough and self.method == 1:
            raise ValueError(
                "surface roughness (Route 1) is only supported with "
                "method='scattering'; for the transfer method, expand the rough "
                "interface into graded sublayers with grade_interface()."
            )

    # ── marshalling helpers ────────────────────────────────────────────────
    def _broadcast_index(self, n) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(n, dtype=complex))
        if arr.size == 1:
            arr = np.full(self.n_wl, arr[0], dtype=complex)
        if arr.size != self.n_wl:
            raise ValueError("refractive index must be scalar or length n_wl")
        return arr

    def _tensor_per_wl(self, t: Optional[np.ndarray], name: str) -> np.ndarray:
        """Return a (n_wl, 3, 3) complex array from a (3,3) or (n_wl,3,3) input."""
        if t is None:
            return np.zeros((self.n_wl, 3, 3), dtype=complex)
        t = np.asarray(t, dtype=complex)
        if t.shape == (3, 3):
            return np.broadcast_to(t, (self.n_wl, 3, 3)).copy()
        if t.shape == (self.n_wl, 3, 3):
            return t
        raise ValueError(
            f"{name} must have shape (3,3) or (n_wl,3,3), got {t.shape}"
        )

    def _tensor_per_wl_opt(self, t, name):
        """Like _tensor_per_wl but passes None through."""
        if t is None:
            return None
        return self._tensor_per_wl(np.asarray(t, dtype=complex), name)

    def _flatten_opt(self, per_wl) -> np.ndarray:
        """Flatten an (n_wl,3,3) exit tensor field to [n_wl*18] (re,im) pairs.
        Vectorized: interleave re/im once over all rows (same values as the
        former per-wavelength loop — pure copies)."""
        if per_wl is None:
            return np.empty(0, dtype=float)
        a = np.asarray(per_wl, dtype=complex)
        n = a.shape[0]
        out = np.empty((n, 18), dtype=float)
        out[:, 0::2] = a.real.reshape(n, 9)
        out[:, 1::2] = a.imag.reshape(n, 9)
        return out.reshape(-1)

    def _flatten(self, picker, layers=None) -> np.ndarray:
        """Flatten one tensor field across (wl, layer) into the Rust layout:
        [n_wl * n_layers * 18] as 9 (re, im) pairs row-major.
        layers defaults to self.layers; fields() passes the expanded cell.
        Vectorized: stack the per-layer (n_wl,3,3) blocks, put the layer axis
        after wl (offset order w*n_layers + li, matching Rust), interleave
        re/im once — same values as the former double Python loop."""
        layers = self.layers if layers is None else layers
        stacked = np.stack([self._tensor_per_wl(picker(l), "tensor")
                            for l in layers])       # (n_layers, n_wl, 3, 3)
        stacked = np.moveaxis(stacked, 0, 1).reshape(-1, 9)  # (n_wl*L, 9)
        out = np.empty((stacked.shape[0], 18), dtype=float)
        out[:, 0::2] = stacked.real
        out[:, 1::2] = stacked.imag
        return out.reshape(-1)

    # ── solve ───────────────────────────────────────────────────────────────
    def solve(self) -> Dict[str, np.ndarray]:
        thick = np.asarray([l.thickness_nm for l in self.layers], dtype=float)
        eps = self._flatten(lambda l: l.eps)
        # one (rough_type, sigma) per interface: [entry|l0, l0|l1, …, l_{n-1}|exit]
        n_if = len(self.layers) + 1
        rtypes = np.zeros(n_if, dtype=np.int32)
        rvals = np.zeros(n_if, dtype=float)
        for i, spec in enumerate(self._rough_norm):
            if spec is not None:
                rtypes[i], rvals[i] = spec[0], spec[1]
        if self._exit_rough_norm is not None:
            rtypes[-1], rvals[-1] = self._exit_rough_norm[0], self._exit_rough_norm[1]
        args = (
            self.wl, self.theta,
            self.n_entry.real.copy(), self.n_entry.imag.copy(),
            self.n_exit.real.copy(), self.n_exit.imag.copy(),
        )
        flat = lambda a: self._flatten_opt(a) if a is not None else np.empty(0, dtype=float)
        e_eps, e_rho, e_rhop, e_mu = (flat(x) for x in
            (self.exit_eps, self.exit_rho, self.exit_rhop, self.exit_mu))
        if self._use_full:
            ones = np.broadcast_to(np.eye(3), (3, 3))
            rho = self._flatten(lambda l: l.rho)
            rhop = self._flatten(lambda l: l.rhop)
            mu = self._flatten(lambda l: l.mu if l.mu is not None else ones)
            raw = _rs_solve_full(*args, eps, rho, rhop, mu, thick, rtypes, rvals,
                                  self.method, e_eps, e_rho, e_rhop, e_mu,
                                  self.periods)
        else:
            if e_rho.size or e_rhop.size or e_mu.size:
                raise ValueError("exit MO tensors force the full path; give "
                                 "any layer rho/rhop/mu or pass exit tensors "
                                 "only via the full entry point")
            raw = _rs_solve_simple(*args, eps, thick, rtypes, rvals,
                                    self.method, e_eps, self.periods)
        return self._assemble(raw)

    def _assemble(self, raw: dict) -> Dict[str, np.ndarray]:
        def cx(re, im):
            return np.asarray(raw[re]) + 1j * np.asarray(raw[im])

        res = {
            "J_refl": cx("J_refl_re", "J_refl_im"),
            "J_trans": cx("J_trans_re", "J_trans_im"),
            "J_refl_circ": cx("J_refl_c_re", "J_refl_c_im"),
            "J_trans_circ": cx("J_trans_c_re", "J_trans_c_im"),
            "R": np.asarray(raw["R"]),
            "T": np.asarray(raw["T"]),
            "M_refl": np.asarray(raw["M_refl"]),
            "M_trans": np.asarray(raw["M_trans"]),
            "power_method": np.asarray(raw["power_method"]),
            "n_failed": int(raw["n_failed"]),
        }
        res = self._squeeze(res)
        res["power_method"] = np.where(np.asarray(res["power_method"]) == 1,
                                         "flux", "jones_factor")
        return res

    def _squeeze(self, res: dict) -> dict:
        """Squeeze singleton wl / angle axes (leading two) for ergonomics.
        Shared by solve() and fields(): later axes (z, 3-vectors) are never
        touched, so n_z == 1 stays."""
        for k, v in list(res.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                if self.n_theta == 1:
                    v = v[:, 0, ...]
                if self.n_wl == 1:
                    v = v[0, ...]
                res[k] = v
        return res

    # ── near fields ──────────────────────────────────────────────────────
    def fields(self, z_nm, method=None) -> Dict[str, np.ndarray]:
        """Near fields at depths z_nm (nm, 0 = entry plane; negatives = entry
        side, beyond total thickness = exit side).

        Returns dict with E_p/H_p/E_s/H_s: complex [n_wl, n_th, n_z, 3]
        (full 6-vector per input polarization) + "z_nm" echo + "n_failed".
        Longitudinal-component note: Ez/Hz follow the LOCAL medium (normal-D
        continuity ⇒ Ez jumps by eps_ratio across interfaces — physical).
        z == total thickness returns EXIT-side values; absorption integrals
        must use interior-side limits (drop the exactly-total endpoint).
        method defaults to self.method; propagation is always transfer-based
        (same P/Q as the TM path — matches pyllama's get_in_plane_fields),
        the slot only tags which far-field engine the asymptotes must match.
        periods > 1 expands the cell (z spans the unfolded stack).
        Roughness is rejected: Route-1 dressing is a far-field power
        redistribution with no local-field meaning — expand the interface
        with graded_stack()/grade_interface() first, then take fields."""
        if self._has_rough:
            raise ValueError(
                "fields() does not support Route-1 roughness (no local-field "
                "meaning); expand rough interfaces into graded sublayers with "
                "graded_stack()/grade_interface() first")
        if isinstance(method, str):
            m = _lookup_method(method)
        elif method is None:
            m = self.method
        else:
            m = int(method)
            if m not in (0, 1, 2):
                raise ValueError(f"unknown method code {m}")
        z = np.atleast_1d(np.asarray(z_nm, dtype=float))
        if z.size == 0:
            raise ValueError("z_nm must be non-empty")
        layers = self.layers if self.periods == 1 else self.layers * self.periods
        thick = np.asarray([l.thickness_nm for l in layers], dtype=float)
        eps = self._flatten(lambda l: l.eps, layers)
        args = (
            self.wl, self.theta,
            self.n_entry.real.copy(), self.n_entry.imag.copy(),
            self.n_exit.real.copy(), self.n_exit.imag.copy(),
        )
        flat = lambda a: self._flatten_opt(a) if a is not None else np.empty(0, dtype=float)
        e_eps, e_rho, e_rhop, e_mu = (flat(x) for x in
            (self.exit_eps, self.exit_rho, self.exit_rhop, self.exit_mu))
        if self._use_full:
            ones = np.broadcast_to(np.eye(3), (3, 3))
            rho = self._flatten(lambda l: l.rho, layers)
            rhop = self._flatten(lambda l: l.rhop, layers)
            mu = self._flatten(lambda l: l.mu if l.mu is not None else ones, layers)
            raw = _rs_fields_full(*args, eps, rho, rhop, mu, thick, z,
                                   e_eps, e_rho, e_rhop, e_mu, m)
        else:
            if e_rho.size or e_rhop.size or e_mu.size:
                raise ValueError("exit MO tensors force the full path; call "
                                 "fields() on a stack with layer rho/rhop/mu")
            raw = _rs_fields_simple(*args, eps, thick, z, e_eps, m)

        def cx(re, im):
            return np.asarray(raw[re]) + 1j * np.asarray(raw[im])
        res = {
            "E_p": cx("E_p_re", "E_p_im"), "H_p": cx("H_p_re", "H_p_im"),
            "E_s": cx("E_s_re", "E_s_im"), "H_s": cx("H_s_re", "H_s_im"),
            "z_nm": np.asarray(raw["z_nm"]), "n_failed": int(raw["n_failed"]),
        }
        return self._squeeze(res)


def mueller_from_jones(jones: np.ndarray) -> np.ndarray:
    """Mueller matrix (4x4 real) from a 2x2 complex Jones matrix."""
    j = np.asarray(jones, dtype=complex).reshape(2, 2)
    return np.asarray(
        _rs_mueller_from_jones(j.real.reshape(4).copy(), j.imag.reshape(4).copy())
    )


def _m16_flat(M) -> tuple:
    """(...,4,4) Mueller stack -> (flat row-major (n,16) contiguous, leading
    shape). One batched Rust call per metric (Rayon over rows) — the former
    per-row Python loop is gone (architecture review follow-up)."""
    M = np.asarray(M, dtype=float)
    if M.shape[-2:] != (4, 4):
        raise ValueError(f"expected (...,4,4), got {M.shape}")
    return np.ascontiguousarray(M.reshape(-1, 16)), M.shape[:-2]


def depolarization_index(M):
    """DI over (...,4,4) Mueller matrices -> (...) float; 1 == non-depolarizing."""
    flat, sh = _m16_flat(M)
    return np.asarray(_rs_di(flat.reshape(-1)), dtype=float).reshape(sh)


def diattenuation(M):
    """Diattenuation vector (first row / M00) over (...,4,4) -> (...,3)."""
    flat, sh = _m16_flat(M)
    return np.asarray(_rs_diat(flat.reshape(-1)), dtype=float).reshape(sh + (3,))


def polarizance(M):
    """Polarizance vector (first column / M00) over (...,4,4) -> (...,3)."""
    flat, sh = _m16_flat(M)
    return np.asarray(_rs_pol(flat.reshape(-1)), dtype=float).reshape(sh + (3,))


def circular_dichroism(M):
    """CD = M03/M00 over (...,4,4) -> (...) float (dimensionless)."""
    flat, sh = _m16_flat(M)
    return np.asarray(_rs_cd(flat.reshape(-1)), dtype=float).reshape(sh)


def cloude(M):
    """Cloude coherency eigendecomposition over (...,4,4), one batched call.

    Returns ``{"lambda": (...,4) descending eigenvalues (sum == M00),
    "entropy": (...) base-4 Shannon entropy}``.
    """
    flat, sh = _m16_flat(M)
    lam, ent = _rs_cloude(flat.reshape(-1))
    return {"lambda": np.asarray(lam, dtype=float).reshape(sh + (4,)),
            "entropy": np.asarray(ent, dtype=float).reshape(sh)}


def _rot_apply(rs_fn, *angle_args, eps) -> np.ndarray:
    """Single choke point for Rust-backed rotations (flat-9 re/im out)."""
    e = np.asarray(eps, dtype=complex)
    if e.shape != (3, 3):
        raise ValueError(f"eps must be (3,3), got {e.shape}")
    flat = np.asarray(rs_fn(*angle_args, e.real.reshape(9).copy(),
                             e.imag.reshape(9).copy()),
                      dtype=float).reshape(9, 2)
    return flat[:, 0].reshape(3, 3) + 1j * flat[:, 1].reshape(3, 3)


def rot_axis(axis, theta_rad: float, eps: np.ndarray) -> np.ndarray:
    """Rotate a 3x3 tensor about an arbitrary axis (Rodrigues).

    Rust ``axis_angle`` path; matches pyllama ``rot_mat`` (which raises on a
    zero axis — so do we, with ``ValueError``).
    """
    ax = np.atleast_1d(np.asarray(axis, dtype=float))
    if ax.shape != (3,):
        raise ValueError(f"axis must have 3 components, got shape {ax.shape}")
    if float(np.dot(ax, ax)) < 1e-300:
        raise ValueError("rotation axis must be non-zero (like pyllama rot_mat)")
    return _rot_apply(_rs_rot_aa, float(ax[0]), float(ax[1]), float(ax[2]),
                      float(theta_rad), eps=eps)


def rot_euler(rx_rad: float, ry_rad: float, rz_rad: float,
              eps: np.ndarray) -> np.ndarray:
    """Extrinsic Euler rotation, R = Rx·Ry·Rz.

    Matches ``dielectric_tensor.euler_rotation_matrix(rx, ry, rz)`` (whose
    docstring says "the transformation is z, then y, then x").
    """
    return _rot_apply(_rs_rot_eu, float(rx_rad), float(ry_rad), float(rz_rad),
                      eps=eps)


def rot_quat(w: float, x: float, y: float, z: float,
             eps: np.ndarray) -> np.ndarray:
    """Quaternion rotation, scalar-first ``(w, x, y, z)``.

    Same matrix as scipy ``Rotation.from_quat([x, y, z, w])`` (scalar-last
    at the live call site — the order is explicit in our arg names). Zero
    norm raises ``ValueError``.
    """
    if float(w*w + x*x + y*y + z*z) < 1e-300:
        raise ValueError("quaternion must have non-zero norm")
    return _rot_apply(_rs_rot_q, float(w), float(x), float(y), float(z),
                      eps=eps)


def rot_z(eps: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a 3x3 tensor about z by ``angle_rad`` (R eps R^T).

    Kept name/signature; now delegates to the Rust axis-angle path.
    """
    return rot_axis([0.0, 0.0, 1.0], angle_rad, eps)


# ──────────── Phase 10: chiral media + twist staircase ─────────────────────
# TIME_CONVENTION record (operational, §10.2): the engine's (rho, rhop) slots
# couple with opposite sign to the e^{-iwt} SI Pasteur derivation — i.e. as
# if e^{+iwt} — so the RECORDED Pasteur mapping is rho = -iκI, rhop = +iκI.
# Pinned by berreman::tests::pasteur_task0_* (spectrum {±(n±κ)} + handedness
# q = n+κ ↔ F-col0 = RCP channel). Achiral observables cannot pin this;
# every formula below is written in κ plus this assignment.

def chiral_layer(n, kappa, thickness_nm, mu=1.0):
    """Pasteur–Tellegen chiral slab (reciprocal, isotropic background).

    SI definition (e^{-iwt}): ``D = εE + iξH``, ``B = -iξE + μH`` with
    ``ξ = κ/c₀`` (κ dimensionless; Lindell–Sihvola constitutive form).
    Normalization chain: code units carry impedance-scaled H (``h = Z₀H``),
    so ``D̃ = εE + iκh``, ``B̃ = μh - iκE`` (c₀ε₀Z₀ = 1, c₀μ₀ = Z₀) — and the
    TIME_CONVENTION record above flips the slots: ``rho = -iκI₃``,
    ``rhop = +iκI₃``. Eigen-indices ``n ± κ`` with ``n + κ`` on the RCP
    (channel-0) circular basis vector; κ → 0 reduces to an isotropic slab.

    Parameters ``n``/``mu`` may be real scalars (positive, finite) or
    length-``n_wl`` arrays only via a manual Layer build — this helper is
    non-dispersive; use ``kappa_table`` + Phase-9 Table models for
    dispersive backgrounds. Warns for ``|κ| ≥ 0.2n`` (natural-media regime;
    chiral nihility κ → n makes det = n²−κ² singular → NaN/n_failed).

    The tensor mapping itself lives in Rust (``berreman::pasteur_tensors``,
    shared with the Task-0 convention tests — single source of truth); this
    wrapper only validates the warning regime and materializes the Layer.
    """
    n = float(n)
    kappa = float(kappa)
    mu = float(mu)
    if not (np.isfinite(n) and n > 0):
        raise ValueError(f"chiral_layer: n must be finite and > 0, got {n}")
    if abs(kappa) >= 0.2 * n and np.isfinite(kappa):
        warnings.warn(
            f"chiral_layer: |kappa|={abs(kappa):.4g} >= 0.2*n ({0.2*n:.4g}); "
            "outside the natural-media regime, det = n^2-k^2 nears singular",
            stacklevel=2)
    eps18, rho18, rhop18, mu18 = _rs_pasteur_tensors(n, kappa, mu)

    def to_t(flat):
        pair = np.asarray(flat, dtype=float).reshape(9, 2)
        return pair[:, 0].reshape(3, 3) + 1j * pair[:, 1].reshape(3, 3)

    return Layer(eps=to_t(eps18), thickness_nm=float(thickness_nm),
                 rho=to_t(rho18), rhop=to_t(rhop18), mu=to_t(mu18))


def kappa_table(wavelengths_nm, kappas):
    """Dispersive chirality κ(λ) as (wavelengths, values) for table lookup.

    Thin validation wrapper (finite, matching lengths, ascending λ ≥ 1
    point): returns ``(np.asarray(wavelengths, float), np.asarray(kappas,
    float))`` for use with Phase-9 Table-model interpolation by the caller.
    For a named physical model see ``condon_kappa`` (§10.5,
    CHIRAL_BRIDGE.md; gates G16).
    """
    wl = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    ka = np.atleast_1d(np.asarray(kappas, dtype=float))
    if wl.shape != ka.shape or wl.size == 0:
        raise ValueError("kappa_table: wavelengths and kappas must be "
                         "non-empty with matching shape")
    if not (np.all(np.isfinite(wl)) and np.all(np.isfinite(ka))):
        raise ValueError("kappa_table: wavelengths and kappas must be finite")
    if wl.size > 1 and not np.all(np.diff(wl) > 0):
        raise ValueError("kappa_table: wavelengths must be strictly ascending")
    return wl, ka


_C0_NM_S = 299792458e9  # c₀ in nm/s


def dbf_beta_to_kappa(beta, n, wavelength_nm):
    """Drude–Born–Fedorov β → Pasteur κ (weak-chirality bridge, docs §10.9).

    DBF constitutive relations ``D = ε(E + β∇×E)``, ``B = μ(H + β∇×H)``
    (e^{-iωt}; Cho, arXiv:1501.01078 eqs (2)–(3)) have the EXACT circular
    eigen-indices ``n± = n/(1 ∓ x)`` with ``x = β·k₀·n`` (derived from
    plane-wave algebra in §10.9; asymmetric about ``n`` — no single κ
    matches both).  This helper returns the half-circular-birefringence
    mapping ``κ_sym = (n⁺−n⁻)/2 = n·x/(1−x²)``, which reproduces the DBF
    circular birefringence Δn EXACTLY and each absolute index to O(x²)
    (the mean-index and per-channel-impedance residues are O(x²)/O(x)
    respectively — documented, see §10.9).  Weak-chirality limit
    ``κ ≈ β·k₀·n²``.  Inverse: ``kappa_to_dbf_beta``.
    """
    beta = float(beta)
    n = float(n)
    lam = float(wavelength_nm)
    if not (np.isfinite(beta) and np.isfinite(n) and np.isfinite(lam)):
        raise ValueError("dbf_beta_to_kappa: all arguments must be finite")
    if n <= 0 or lam <= 0:
        raise ValueError(f"dbf_beta_to_kappa: need n > 0 and wavelength > 0, "
                         f"got n={n}, lambda={lam}")
    x = beta * (2 * np.pi / lam) * n
    if abs(x) >= 1.0:
        raise ValueError(
            f"dbf_beta_to_kappa: |beta*k0*n|={abs(x):.4g} >= 1 hits the DBF "
            "eigenvalue pole n+=n/(1-x); DBF itself breaks down there")
    if abs(x) >= 0.2:
        warnings.warn(
            f"dbf_beta_to_kappa: |beta*k0*n|={abs(x):.4g} >= 0.2; outside "
            "the natural-media weak-chirality regime (same boundary as "
            "chiral_layer's |kappa| >= 0.2n warning)", stacklevel=2)
    return n * x / (1.0 - x * x)


def kappa_to_dbf_beta(kappa, n, wavelength_nm):
    """Exact inverse of ``dbf_beta_to_kappa`` (κ_sym ↔ β roundtrip).

    From ``κ/n = x/(1−x²)``: ``x = (√(1+4(κ/n)²) − 1)/(2(κ/n))``,
    ``β = x/(k₀·n)``.  Requires ``|κ| < n`` (det = n²−κ² > 0).
    """
    kappa = float(kappa)
    n = float(n)
    lam = float(wavelength_nm)
    if not (np.isfinite(kappa) and np.isfinite(n) and np.isfinite(lam)):
        raise ValueError("kappa_to_dbf_beta: all arguments must be finite")
    if n <= 0 or lam <= 0:
        raise ValueError(f"kappa_to_dbf_beta: need n > 0 and wavelength > 0, "
                         f"got n={n}, lambda={lam}")
    if abs(kappa) >= n:
        raise ValueError(
            f"kappa_to_dbf_beta: |kappa|={abs(kappa):.4g} >= n={n:.4g}; "
            "det = n^2-k^2 <= 0 (singular chiral medium)")
    a = kappa / n
    # x = (sqrt(1+4a^2)-1)/(2a) rationalized: no small-a cancellation
    x = 2.0 * a / (1.0 + np.sqrt(1.0 + 4.0 * a * a)) if a != 0.0 else 0.0
    return x * lam / (2 * np.pi * n)


def condon_kappa(wavelengths_nm, R, lambda0_nm, gamma=0.0):
    """Condon single-oscillator chirality dispersion κ(λ) (docs §10.9).

    Condon–Altar–Eyring one-electron rotatory power (1937; Lindell 1994
    bi-isotropic form; Akyurtlu & Werner 2004 as the FDTD-standard):

        κ(ω) = ω·R / (ω₀² − ω² − i·ω·Γ),   ω = 2πc₀/λ (vacuum),

    with R the rotational-strength amplitude (units of angular frequency
    in our normalization; a per-material fit parameter — its absolute
    scale is convention-dependent across the literature, its dispersion
    SHAPE is not), ω₀ the resonant angular frequency (``lambda0_nm``), Γ
    the damping.  Returned κ is in OUR normalization: eigen-indices
    ``n± = n ± κ`` (Task-0-pinned; matches the Lindell eigenvalue
    structure ``n± = √(εμ) ± κ``).  Positive R below resonance gives
    κ > 0 ⇒ ``arg(t_R) − arg(t_L) = +k₀κd`` (G15c sense, channel-0=RCP).

    ``gamma=0`` (lossless ORD) returns a real array; ``gamma > 0``
    returns complex κ (Im κ > 0 ⇒ R-channel absorbs; feed via the tensor
    path, see §10.9 — ``rho = -1j·κ[:,None,None]·I₃``).  ω → 0 gives
    κ → 0 linearly (the ω numerator: no static chirality); far from
    resonance κ ≈ ωR/(ω₀² − ω²) real.
    """
    lam = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    R = float(R)
    lam0 = float(lambda0_nm)
    gam = float(gamma)
    if lam.size == 0 or not np.all(np.isfinite(lam)) or np.any(lam <= 0):
        raise ValueError("condon_kappa: wavelengths must be non-empty, "
                         "finite and > 0")
    if not (np.isfinite(R) and np.isfinite(lam0) and np.isfinite(gam)):
        raise ValueError("condon_kappa: R, lambda0_nm, gamma must be finite")
    if lam0 <= 0:
        raise ValueError(f"condon_kappa: lambda0_nm must be > 0, got {lam0}")
    if gam < 0:
        raise ValueError(f"condon_kappa: gamma must be >= 0, got {gam}")
    om = 2 * np.pi * _C0_NM_S / lam
    om0 = 2 * np.pi * _C0_NM_S / lam0
    kappa = om * R / (om0 * om0 - om * om - 1j * om * gam)
    return np.real(kappa) if gam == 0.0 else kappa


def twisted_stack(eps_uniaxial, total_nm, twist_rad, n_slices, grid="midpoint"):
    """Discretized in-plane twist staircase (cf. B44 TwistedMaterial and
    pyllama's CholestericModel).

    ``eps_uniaxial``: (3, 3) tensor at zero twist; slice k carries
    ``rot_z(eps, twist * frac)`` with ``frac = (k+0.5)/n`` for
    ``grid="midpoint"`` (default, 2nd-order accurate in slice thickness) or
    ``frac = k/(n-1)`` for ``grid="endpoint"`` (matches B44 ``getSlices``
    ``linspace(0, d, div+1)`` sampling so discretization drops out of
    solver-vs-solver comparisons — G15d). Returns ``list[Layer]`` with equal
    thicknesses summing to ``total_nm``.
    """
    e0 = np.asarray(eps_uniaxial, dtype=complex)
    if e0.shape != (3, 3):
        raise ValueError(f"twisted_stack: eps_uniaxial must be (3,3), got {e0.shape}")
    n_slices = int(n_slices)
    if n_slices < 1:
        raise ValueError("twisted_stack: n_slices >= 1")
    if grid not in ("midpoint", "endpoint"):
        raise ValueError(f"twisted_stack: grid must be 'midpoint' or 'endpoint', got {grid!r}")
    # Schedule + per-slice rotation live in Rust (rotations::twisted_tensors;
    # same rot_z code path the old Python loop used — bit-identical output).
    grid_code = 0 if grid == "midpoint" else 1
    flat = _rs_twisted_tensors(e0.real.reshape(9).copy(), e0.imag.reshape(9).copy(),
                               float(twist_rad), n_slices, grid_code)
    pair = np.asarray(flat, dtype=float).reshape(-1, 9, 2)
    dz = float(total_nm) / n_slices
    return [Layer(p[:, 0].reshape(3, 3) + 1j * p[:, 1].reshape(3, 3), dz)
            for p in pair]


def cholesteric_stack(n_o, n_e, pitch_nm, n_periods, n_per_pitch=48,
                      grid="midpoint"):
    """Full-pitch cholesteric helix: uniaxial tensor rotating 2π per pitch
    (cf. B44 TwistedMaterial; pyllama cholesteric.py / CholestericModel).
    Director starts along x; default 48 slices/pitch (G15d convergence
    triple: 24/48/96). Returns ``list[Layer]``.

    SAMPLING (P10 correction — the draft had this backwards): B44's
    InhomogeneousLayer midpoint evaluation samples the tensor at slice
    midpoints ((k+0.5)*d/div) even though getSlices() returns endpoint
    edges linspace(0,d,div+1). grid="midpoint" (default) reproduces B44's
    sampling EXACTLY (same angles, same widths) so discretization drops out
    of solver-vs-solver comparisons; "endpoint" is kept for explicit
    edge-matched studies only.
    """
    e0 = np.diag([n_e ** 2, n_o ** 2, n_o ** 2]).astype(complex)
    return twisted_stack(e0, pitch_nm * n_periods, 2 * np.pi * n_periods,
                         n_periods * n_per_pitch, grid=grid)


# ─────────────────────── Route 2: graded effective-medium roughness ──────────
def _as_tensor(eps) -> np.ndarray:
    """Promote a permittivity spec to a 3x3 complex tensor.

    Accepts a scalar permittivity, a length-3 diagonal, or a full 3x3 tensor.
    """
    e = np.asarray(eps, dtype=complex)
    if e.ndim == 0:
        return np.eye(3, dtype=complex) * e
    if e.shape == (3,):
        return np.diag(e).astype(complex)
    if e.shape == (3, 3):
        return e
    raise ValueError("eps must be scalar, shape (3,), or (3, 3)")


def grade_interface(eps_a, eps_b, sigma_nm: float, n_sublayers: int = 7,
                    total_width_nm: Optional[float] = None,
                    mixing: str = "linear") -> List["Layer"]:
    """Graded effective-medium sublayers bridging medium A → medium B (Route 2).

    Models a Gaussian rough interface of RMS roughness ``sigma_nm`` as a stack of
    thin homogeneous sublayers whose permittivity *tensor* interpolates from
    ``eps_a`` to ``eps_b``. The volume fraction of medium B at depth ``z`` (from
    the nominal interface) follows the cumulative Gaussian
    ``f(z) = ½(1 + erf(z / (σ√2)))``; the transition region spans
    ``±total_width/2`` (default ``total_width = 6σ``).

    Unlike Route 1 this makes no perturbative (Névot–Croce) assumption, handles
    anisotropic tensors directly, and works in *both* the scattering and
    transfer methods (each sublayer is an ordinary Berreman layer).

    Parameters
    ----------
    eps_a, eps_b : scalar | (3,) | (3, 3)
        Permittivities of the incident- and transmitted-side media.
    sigma_nm : float
        RMS interface roughness.
    n_sublayers : int
        Number of graded slabs (≈5–10 is usually plenty).
    total_width_nm : float, optional
        Total transition thickness. Default ``6 * sigma_nm`` (±3σ).
    mixing : {"linear"}
        Tensor mixing rule. ``"linear"`` is volume-weighted (Wiener / graded
        index): ``ε_eff = (1−f)·ε_a + f·ε_b``.

    Returns
    -------
    list[Layer]
        Sublayers (front→back), total thickness ``total_width_nm``. Empty if
        ``sigma_nm <= 0``.
    """
    if sigma_nm is None or sigma_nm <= 0 or n_sublayers < 1:
        return []
    try:
        ea, eb = _as_tensor(eps_a), _as_tensor(eps_b)
    except ValueError:
        raise ValueError("grade_interface needs scalar/(3,)/(3,3) eps; dispersive "
                         "(n_wl,3,3) + roughness is unsupported — build sublayers "
                         "per wavelength instead") from None
    if mixing != "linear":
        raise ValueError("unknown mixing '%s' (only 'linear')" % mixing)
    # Gaussian-CDF profile + volume-weighted tensor mixing live in Rust
    # (roughness::graded_tensors); the wrapper materializes Layers.
    flat = _rs_grade_tensors(ea.real.reshape(9).copy(), ea.imag.reshape(9).copy(),
                             eb.real.reshape(9).copy(), eb.imag.reshape(9).copy(),
                             float(sigma_nm), int(n_sublayers),
                             None if total_width_nm is None else float(total_width_nm))
    pair = np.asarray(flat, dtype=float).reshape(-1, 9, 2)
    return [Layer(p[:, 0].reshape(3, 3) + 1j * p[:, 1].reshape(3, 3),
                  float(total_width_nm if total_width_nm is not None
                        else 6.0 * sigma_nm) / n_sublayers)
            for p in pair]


def graded_stack(layers: Sequence["Layer"], interface_sigma_nm: Sequence[float],
                 n_entry: complex = 1.0, n_exit: complex = 1.0,
                 n_sublayers: int = 7, total_width_nm: Optional[float] = None,
                 mixing: str = "linear") -> List["Layer"]:
    """Expand a smooth stack into a graded (Route-2) stack.

    For each rough interface a graded transition region is inserted and half its
    width is shaved from each adjacent *finite* layer (the semi-infinite entry /
    exit half-spaces are not shaved). The returned plain ``Layer`` list carries
    no Route-1 dressing, so it can be solved with either method.

    Parameters
    ----------
    layers : sequence of Layer
        The interior layers (length N).
    interface_sigma_nm : sequence of float
        RMS roughness per interface, length N+1, ordered
        ``[entry|l0, l0|l1, …, l_{N-1}|exit]``.
    n_entry, n_exit : complex
        Half-space refractive indices (used as the outer media for the first /
        last graded regions).
    """
    layers = list(layers)
    n = len(layers)
    if len(interface_sigma_nm) != n + 1:
        raise ValueError("interface_sigma_nm must have length len(layers)+1 = %d" % (n + 1))
    entry_eps = np.eye(3, dtype=complex) * (complex(n_entry) ** 2)
    exit_eps = np.eye(3, dtype=complex) * (complex(n_exit) ** 2)
    if any(s and s > 0 for s in interface_sigma_nm):
        mo = [j for j, lyr in enumerate(layers)
              if lyr.rho is not None or lyr.rhop is not None or lyr.mu is not None]
        if mo:
            warnings.warn(
                f"graded_stack: transition sublayers are non-magnetic; MO tensors "
                f"on layers {mo} are dropped inside the graded region", UserWarning)

    def width(i):
        s = interface_sigma_nm[i]
        if s is None or s <= 0:
            return 0.0
        return float(6.0 * s if total_width_nm is None else total_width_nm)

    out: List[Layer] = []
    for j, lyr in enumerate(layers):
        a_eps = entry_eps if j == 0 else layers[j - 1].eps
        if interface_sigma_nm[j] and interface_sigma_nm[j] > 0:
            out += grade_interface(a_eps, lyr.eps, interface_sigma_nm[j],
                                   n_sublayers, total_width_nm, mixing)
        shave = 0.5 * width(j) + 0.5 * width(j + 1)
        if shave > 0 and lyr.thickness_nm - shave < 1e-9:
            warnings.warn(
                f"graded_stack: layer {j} thickness {lyr.thickness_nm}nm < "
                f"3σ shave {shave:.2f}nm; clamped to 1e-9nm "
                f"(thicken the layer or reduce sigma)", UserWarning)
        t_new = max(lyr.thickness_nm - shave, 1e-9)
        out.append(Layer(lyr.eps, t_new, rho=lyr.rho, rhop=lyr.rhop, mu=lyr.mu))
    # final interface (last layer | exit)
    if interface_sigma_nm[n] and interface_sigma_nm[n] > 0:
        out += grade_interface(layers[-1].eps, exit_eps, interface_sigma_nm[n],
                               n_sublayers, total_width_nm, mixing)
    return out
