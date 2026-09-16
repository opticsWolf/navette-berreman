"""Minimal, faithful NumPy reference for the pyllama 4x4 Berreman solver.

Transcribed directly from the uploaded pyllama.py (Layer with the reduced
"simple" Delta, HalfSpace, build_P_Q, transfer-matrix and scattering-matrix
assembly, Fresnel extraction, R/T, and the linear->circular conversion). Used
only as an oracle to validate the Rust port. No external package needed.
"""
import numpy as np
from numpy import linalg as la

thr = 1e-7


class Wave:
    def __init__(self, eps, Kx, Ex, Ey, Hx, Hy):
        Ez = -(eps[2, 0] / eps[2, 2]) * Ex - (eps[2, 1] / eps[2, 2]) * Ey - (Kx / eps[2, 2]) * Hy
        Hz = Kx * Ey
        self.elec = [Ex, Ey, Ez]
        self.magnet = [Hx, Hy, Hz]
        Sx = self.elec[1] * self.magnet[2] - self.elec[2] * self.magnet[1]
        Sy = self.elec[2] * self.magnet[0] - self.elec[0] * self.magnet[2]
        Sz = self.elec[0] * self.magnet[1] - self.elec[1] * self.magnet[0]
        self.poynting = [Sx, Sy, Sz]

    @staticmethod
    def _cp(x, y):
        d = np.abs(x) ** 2 + np.abs(y) ** 2
        return 0 if d == 0 else np.abs(x) ** 2 / d

    def calc_cp_poynting(self):
        return Wave._cp(self.poynting[0], self.poynting[1])

    def calc_cp_elec(self):
        return Wave._cp(self.elec[0], self.elec[1])

    @staticmethod
    def matrix_to_waves(mat, eps, Kx):
        return [Wave(eps, Kx, mat[0, k], mat[2, k], -mat[3, k], mat[1, k]) for k in range(4)]


class Layer:
    def __init__(self, eps, thickness_nm, Kx, k0):
        self.eps = np.array(eps, dtype=complex)
        self.thickness = thickness_nm
        self.Kx = Kx
        self.k0 = k0
        self.D = self._build_D()
        p, q, _ = self._calc_p_q_sorted()
        self.eigenvectors = p
        self.eigenvalues = q
        self.P, self.Q = self.build_P_Q()

    def _build_D(self):
        e = self.eps
        Kx = self.Kx
        return np.array([
            [-Kx * e[2, 0] / e[2, 2], 1 - Kx ** 2 / e[2, 2], -Kx * e[2, 1] / e[2, 2], 0],
            [e[0, 0] - e[0, 2] * e[2, 0] / e[2, 2], -Kx * e[0, 2] / e[2, 2],
             e[0, 1] - e[0, 2] * e[2, 1] / e[2, 2], 0],
            [0, 0, 0, 1],
            [e[1, 0] - e[1, 2] * e[2, 0] / e[2, 2], -Kx * e[1, 2] / e[2, 2],
             -Kx ** 2 + e[1, 1] - e[1, 2] * e[2, 1] / e[2, 2], 0],
        ], dtype=complex)

    def _calc_p_q_sorted(self):
        q_unsorted, p_unsorted = la.eig(self.D)
        pw = Wave.matrix_to_waves(p_unsorted, self.eps, self.Kx)
        return self._sort(p_unsorted, q_unsorted, pw)

    def _sort(self, p_unsorted, q_unsorted, pw):
        id_refl, id_trans = [], []
        for k in range(4):
            v = np.real(q_unsorted[k]) + np.imag(q_unsorted[k]).round(decimals=10)
            test = np.real(v) if np.isreal(v) else np.imag(v)
            (id_trans if test > 0 else id_refl).append(k)
        if not (len(id_trans) == 2 and len(id_refl) == 2):
            # fall back: descending split (matches Rust fallback)
            idx = sorted(range(4), key=lambda k: -(np.real(q_unsorted[k]) +
                                                   np.imag(q_unsorted[k]).round(10)))
            order = idx
        else:
            Cp0 = pw[id_trans[0]].calc_cp_poynting()
            Cp1 = pw[id_trans[1]].calc_cp_poynting()
            if np.abs(Cp0 - Cp1) > thr:
                if Cp1 < Cp0:
                    id_trans = [id_trans[1], id_trans[0]]
                Cp0 = pw[id_refl[0]].calc_cp_poynting()
                Cp1 = pw[id_refl[1]].calc_cp_poynting()
                if Cp1 < Cp0:
                    id_refl = [id_refl[1], id_refl[0]]
            else:
                Cp0 = pw[id_trans[0]].calc_cp_elec()
                Cp1 = pw[id_trans[1]].calc_cp_elec()
                if (Cp1 - Cp0) < thr:
                    id_trans = [id_trans[1], id_trans[0]]
                Cp0 = pw[id_refl[0]].calc_cp_elec()
                Cp1 = pw[id_refl[1]].calc_cp_elec()
                if (Cp1 - Cp0) < thr:
                    id_refl = [id_refl[1], id_refl[0]]
            order = [id_trans[1], id_trans[0], id_refl[1], id_refl[0]]
        q_sorted = np.array([q_unsorted[order[k]] for k in range(4)])
        p_sorted = np.stack([p_unsorted[:, order[k]].T for k in range(4)], axis=1)
        pws = [pw[order[k]] for k in range(4)]
        return p_sorted, q_sorted, pws

    def build_P_Q(self):
        P = self.eigenvectors
        Q = np.diag([np.exp(1j * self.k0 * self.eigenvalues[k] * self.thickness) for k in range(4)])
        return P, Q


class HalfSpace(Layer):
    def __init__(self, eps, Kx, Kz, k0):
        self.eps = np.array(eps, dtype=complex)
        self.thickness = 0
        self.Kx = Kx
        self.Kz = Kz
        self.k0 = k0
        self.D = self._build_D()
        p, q, _ = self._calc_p_q_sorted()
        self.eigenvectors = p
        self.eigenvalues = q
        self.P, self.Q = self.build_P_Q()

    def _calc_p_q_sorted(self):
        n = np.sqrt(self.eps[0, 0])
        sin_phi = self.Kx / n
        cos_phi = np.sqrt(1 - sin_phi ** 2 + 0j)
        q_sorted = [n * cos_phi, n * cos_phi, -n * cos_phi, -n * cos_phi]
        p = np.array([
            [cos_phi, 0, cos_phi, 0],
            [n, 0, -n, 0],
            [0, 1, 0, 1],
            [0, n * cos_phi, 0, -n * cos_phi],
        ], dtype=complex)
        return p, q_sorted, None


def s_to_next(a, b):
    Qf = np.array([[a.Q[0, 0], a.Q[0, 1], 0, 0], [a.Q[1, 0], a.Q[1, 1], 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    Qb = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                   [0, 0, a.Q[2, 2], a.Q[2, 3]], [0, 0, a.Q[3, 2], a.Q[3, 3]]])
    Pout = np.array([[a.P[i, 0], a.P[i, 1], -b.P[i, 2], -b.P[i, 3]] for i in range(4)])
    Pin = np.array([[b.P[i, 0], b.P[i, 1], -a.P[i, 2], -a.P[i, 3]] for i in range(4)])
    return la.multi_dot((la.inv(Qb), la.inv(Pin), Pout, Qf))


def s_combine(ab, bc):
    def blk(s, r, c):
        return np.array([[s[2 * r, 2 * c], s[2 * r, 2 * c + 1]],
                         [s[2 * r + 1, 2 * c], s[2 * r + 1, 2 * c + 1]]])
    ab00, ab01, ab10, ab11 = blk(ab, 0, 0), blk(ab, 0, 1), blk(ab, 1, 0), blk(ab, 1, 1)
    bc00, bc01, bc10, bc11 = blk(bc, 0, 0), blk(bc, 0, 1), blk(bc, 1, 0), blk(bc, 1, 1)
    C = la.inv(np.identity(2) - ab01 @ bc10)
    ac00 = la.multi_dot((bc00, C, ab00))
    ac01 = bc01 + la.multi_dot((bc00, C, ab01, bc11))
    ac10 = ab10 + la.multi_dot((ab11, bc10, C, ab00))
    ac11 = la.multi_dot((ab11, (np.identity(2) + la.multi_dot((bc10, C, ab01))), bc11))
    return np.array([
        [ac00[0, 0], ac00[0, 1], ac01[0, 0], ac01[0, 1]],
        [ac00[1, 0], ac00[1, 1], ac01[1, 0], ac01[1, 1]],
        [ac10[0, 0], ac10[0, 1], ac11[0, 0], ac11[0, 1]],
        [ac10[1, 0], ac10[1, 1], ac11[1, 0], ac11[1, 1]],
    ])


def solve(layers, eps_entry, eps_exit, wl, theta_in, method="SM"):
    n_entry = np.sqrt(eps_entry)
    n_exit = np.sqrt(eps_exit)
    k0 = 2 * np.pi / wl
    Kx = n_entry * np.sin(theta_in)
    Kz_entry = n_entry * np.cos(theta_in)
    theta_out = np.arcsin((n_entry / n_exit) * np.sin(theta_in + 0j))
    Kz_exit = n_exit * np.cos(theta_out)

    entry = HalfSpace(np.diag([eps_entry] * 3), Kx, Kz_entry, k0)
    exit_ = HalfSpace(np.diag([eps_exit] * 3), Kx, Kz_exit, k0)
    L = [Layer(eps, t, Kx, k0) for (eps, t) in layers]

    if method == "TM":
        T = np.identity(4, dtype=complex)
        for l in L:
            Tl = la.multi_dot((l.P, l.Q, la.inv(l.P)))
            T = Tl @ T
        TM = la.multi_dot((la.inv(exit_.P), T, entry.P))
        deno = TM[2, 2] * TM[3, 3] - TM[3, 2] * TM[2, 3]
        r_pp = (TM[3, 0] * TM[2, 3] - TM[2, 0] * TM[3, 3]) / deno
        r_ps = (TM[2, 0] * TM[3, 2] - TM[3, 0] * TM[2, 2]) / deno
        r_sp = (TM[3, 1] * TM[2, 3] - TM[2, 1] * TM[3, 3]) / deno
        r_ss = (TM[2, 1] * TM[3, 2] - TM[3, 1] * TM[2, 2]) / deno
        t_pp = TM[0, 0] + TM[0, 2] * r_pp + TM[0, 3] * r_ps
        t_ps = TM[1, 0] + TM[1, 2] * r_pp + TM[1, 3] * r_ps
        t_sp = TM[0, 1] + TM[0, 2] * r_sp + TM[0, 3] * r_ss
        t_ss = TM[1, 1] + TM[1, 2] * r_sp + TM[1, 3] * r_ss
        J_refl = np.array([[r_pp, r_sp], [r_ps, r_ss]])
        J_trans = np.array([[t_pp, t_sp], [t_ps, t_ss]])
    else:
        n = len(L)
        S = np.identity(4, dtype=complex)
        if n >= 2:
            for kl in range(n - 2, -1, -1):
                S = s_combine(s_to_next(L[kl], L[kl + 1]), S)
        S_entry = s_to_next(entry, L[0])
        S_exit = s_to_next(L[-1], exit_)
        S = s_combine(S, S_exit)
        S = s_combine(S_entry, S)
        J_refl = np.array([[S[2, 0], S[2, 1]], [S[3, 0], S[3, 1]]])
        J_trans = np.array([[S[0, 0], S[0, 1]], [S[1, 0], S[1, 1]]])

    factor = (Kz_exit / Kz_entry).real
    R = np.abs(J_refl) ** 2
    T = factor * np.abs(J_trans) ** 2
    # circular
    F = np.array([[1, 1], [-1j, 1j]])
    B = np.array([[1, 1], [1j, -1j]])
    J_refl_c = la.multi_dot((la.inv(B), J_refl, F))
    J_trans_c = la.multi_dot((la.inv(F), J_trans, F))
    return dict(J_refl=J_refl, J_trans=J_trans, R=R, T=T, factor=factor,
                J_refl_c=J_refl_c, J_trans_c=J_trans_c)


def rot_z(eps, angle):
    cphi, sphi = np.cos(angle), np.sin(angle)
    Rz = np.array([[cphi, -sphi, 0], [sphi, cphi, 0], [0, 0, 1]])
    return Rz @ eps @ Rz.T


# ── canonical test cases (shared with the Rust validator) ──────────────────
def test_cases():
    cases = []
    # 1: single isotropic layer, normal incidence
    cases.append(("iso_normal",
                  [(np.diag([2.25, 2.25, 2.25]).astype(complex), 200.0)],
                  1.0, 1.0, 550.0, 0.0))
    # 2: uniaxial layer, normal incidence
    cases.append(("uniaxial_normal",
                  [(np.diag([2.25, 2.56, 2.25]).astype(complex), 300.0)],
                  1.0, 1.0, 550.0, 0.0))
    # 3: rotated uniaxial, oblique 30 deg
    e = np.diag([2.25, 2.89, 2.25]).astype(complex)
    cases.append(("rot_uniaxial_oblique",
                  [(rot_z(e, np.radians(35.0)), 250.0)],
                  1.0, 1.0, 633.0, np.radians(30.0)))
    # 4: two-layer stack, oblique 20 deg, asymmetric media
    cases.append(("two_layer_oblique",
                  [(np.diag([2.56, 2.56, 2.56]).astype(complex), 120.0),
                   (rot_z(np.diag([2.1, 2.7, 2.1]).astype(complex), np.radians(50.0)), 180.0)],
                  1.0, 2.25, 500.0, np.radians(20.0)))
    # 5: absorbing rotated uniaxial, oblique
    ea = np.diag([2.25 + 0.05j, 3.0 + 0.2j, 2.25 + 0.05j]).astype(complex)
    cases.append(("absorbing_rot_oblique",
                  [(rot_z(ea, np.radians(25.0)), 220.0)],
                  1.0, 1.0, 600.0, np.radians(40.0)))
    return cases


if __name__ == "__main__":
    import json
    out = {}
    for name, layers, ee, ex, wl, th in test_cases():
        res = {}
        for m in ("SM", "TM"):
            r = solve(layers, ee, ex, wl, th, method=m)
            res[m] = dict(
                J_refl=[[ [v.real, v.imag] for v in row] for row in r["J_refl"]],
                J_trans=[[ [v.real, v.imag] for v in row] for row in r["J_trans"]],
                R=r["R"].real.tolist(), T=r["T"].real.tolist(),
                J_refl_c=[[ [v.real, v.imag] for v in row] for row in r["J_refl_c"]],
            )
        out[name] = res
    print(json.dumps(out))
