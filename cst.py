"""Class-Shape Transformation (Kulfan, 2008): parametrização CST de aerofólios.

   y(ψ) = C(ψ) · S(ψ) + ψ · y_TE
   C(ψ) = ψ^N1 · (1−ψ)^N2          (LE/TE)
   S(ψ) = Σ A_i · K_i · ψ^i · (1−ψ)^(n−i)   (Bernstein)

Os pesos A_i (extradorso e intradorso) são as variáveis de design.
"""

from __future__ import annotations

import numpy as np
from scipy.special import comb
from pathlib import Path
from typing import Tuple


def naca4_coords(naca: str = "0012", n: int = 160) -> Tuple[np.ndarray, np.ndarray]:
    """Coordenadas analíticas de NACA 4 dígitos. Formato Selig (TE→LE→TE)."""
    m = int(naca[0]) / 100.0
    p = int(naca[1]) / 10.0
    t = int(naca[2:4]) / 100.0

    # Espaçamento cosenoidal 
    beta = np.linspace(0.0, np.pi, n)
    x = (1.0 - np.cos(beta)) / 2.0

    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4   
    )

    if m == 0.0:
        xu, xl = x, x
        yu, yl = yt, -yt
    else:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc_dx = np.where(x < p, 2 * m / p ** 2 * (p - x),
                          2 * m / (1 - p) ** 2 * (p - x))
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta); yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta); yl = yc - yt * np.cos(theta)

    x_full = np.concatenate([xu[::-1], xl[1:]])
    y_full = np.concatenate([yu[::-1], yl[1:]])
    return x_full, y_full


def save_dat_file(path: Path, x: np.ndarray, y: np.ndarray, name: str = "AIRFOIL") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{name}\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi: .6f}  {yi: .6f}\n")


def load_dat_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    coords = []
    with open(path, "r") as f:
        lines = f.readlines()
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) >= 2:
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    arr = np.array(coords)
    return arr[:, 0], arr[:, 1]


class CSTAirfoil:
    """Aerofólio CST: gera coords, fita pesos a partir de .dat, valida geometria."""

    def __init__(
        self,
        n_upper: int = 8,
        n_lower: int = 8,
        N1: float = 0.5,
        N2: float = 1.0,
    ) -> None:
        self.n_upper = n_upper
        self.n_lower = n_lower
        self.N1 = N1
        self.N2 = N2

    def _class_function(self, psi: np.ndarray) -> np.ndarray:
        return psi ** self.N1 * (1.0 - psi) ** self.N2

    @staticmethod
    def _bernstein_basis(psi: np.ndarray, n: int) -> np.ndarray:
        B = np.zeros((len(psi), n + 1))
        for i in range(n + 1):
            B[:, i] = comb(n, i) * psi ** i * (1.0 - psi) ** (n - i)
        return B

    def _shape_function(self, psi: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n = len(weights) - 1
        B = self._bernstein_basis(psi, n)
        return B @ np.asarray(weights)

    def coords(
        self,
        w_upper: np.ndarray,
        w_lower: np.ndarray,
        n_points: int = 160,
        yTE_upper: float = 0.0,
        yTE_lower: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Coordenadas (x,y) no formato Selig: TE → extradorso → LE → intradorso → TE."""
        beta = np.linspace(0.0, np.pi, n_points)
        psi = (1.0 - np.cos(beta)) / 2.0

        C = self._class_function(psi)
        S_u = self._shape_function(psi, w_upper)
        S_l = self._shape_function(psi, w_lower)

        # Convenção: pesos do extradorso positivos, intradorso negativos
        y_u = C * S_u + psi * yTE_upper
        y_l = C * S_l + psi * yTE_lower

        x = np.concatenate([psi[::-1], psi[1:]])
        y = np.concatenate([y_u[::-1], y_l[1:]])
        return x, y

    def fit_to_coords(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Fita pesos CST a um perfil (x,y) Selig. Retorna (w_u, w_l, yTE_u, yTE_l)."""
        le = int(np.argmin(x))
        x_u, y_u = x[:le + 1][::-1], y[:le + 1][::-1]
        x_l, y_l = x[le:], y[le:]

        yTE_u = float(y_u[-1])
        yTE_l = float(y_l[-1])

        w_u = self._lstsq_fit(x_u, y_u, self.n_upper, yTE_u)
        w_l = self._lstsq_fit(x_l, y_l, self.n_lower, yTE_l)
        return w_u, w_l, yTE_u, yTE_l

    def _lstsq_fit(
        self, psi: np.ndarray, y: np.ndarray, n_w: int, yTE: float
    ) -> np.ndarray:
        # y - ψ·yTE = C(ψ) · Σ A_i B_i(ψ)  →  resolve por Moore-Penrose
        psi = np.asarray(psi, dtype=float)
        y = np.asarray(y, dtype=float)
        C = self._class_function(psi)
        n = n_w - 1
        B = self._bernstein_basis(psi, n)
        M = B * C[:, None]
        b = y - psi * yTE
        weights, *_ = np.linalg.lstsq(M, b, rcond=None)
        return weights

    @staticmethod
    def is_valid_geometry(
        x: np.ndarray,
        y: np.ndarray,
        min_thickness: float = 0.06,
        max_thickness: float = 0.25,
        max_te_thickness: float = 0.01,
        local_thickness_constraints: list | None = None,
        enforce_upper_convexity: bool = False,
        convexity_tolerance: float = 0.001,
        convexity_check_range: tuple = (0.05, 0.95),
        max_curvature_roughness: float = None,
    ) -> Tuple[bool, str]:
        """Filtro de viabilidade física. Retorna (ok, motivo)."""
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return False, "coords_non_finite"

        le = int(np.argmin(x))
        x_u, y_u = x[:le + 1][::-1], y[:le + 1][::-1]
        x_l, y_l = x[le:], y[le:]

        # Reamostra na mesma malha pra poder comparar
        psi_grid = np.linspace(0.001, 0.999, 100)
        y_u_i = np.interp(psi_grid, x_u, y_u)
        y_l_i = np.interp(psi_grid, x_l, y_l)

        if np.any(y_u_i <= y_l_i):
            return False, "self_intersection"

        thickness = y_u_i - y_l_i
        t_max = float(np.max(thickness))
        if t_max < min_thickness:
            return False, f"too_thin({t_max:.3f})"
        if t_max > max_thickness:
            return False, f"too_thick({t_max:.3f})"

        te_thickness = abs(float(y_u[-1] - y_l[-1]))
        if te_thickness > max_te_thickness:
            return False, f"te_too_open({te_thickness:.3f})"

        if local_thickness_constraints:
            for x_c, t_min in local_thickness_constraints:
                idx = int(np.argmin(np.abs(psi_grid - x_c)))
                t_local = float(thickness[idx])
                if t_local < t_min:
                    return False, f"local_thin@x={x_c:.2f}_t={t_local:.3f}<{t_min:.3f}"

        if enforce_upper_convexity:
            psi_dense = np.linspace(
                convexity_check_range[0], convexity_check_range[1], 80
            )
            y_u_dense = np.interp(psi_dense, x_u, y_u)
            d2y = np.gradient(np.gradient(y_u_dense, psi_dense), psi_dense)
            d2y_max = float(np.max(d2y))
            if d2y_max > convexity_tolerance:
                idx_bad = int(np.argmax(d2y))
                x_bad = psi_dense[idx_bad]
                return False, f"upper_inflection@x={x_bad:.2f}_d2y={d2y_max:.4f}"

            # Intradorso: tolerância 3x mais frouxa (perfis cambrados são
            # naturalmente côncavos no intradorso; só barra patológico)
            y_l_dense = np.interp(psi_dense, x_l, y_l)
            d2y_l = np.gradient(np.gradient(y_l_dense, psi_dense), psi_dense)
            d2y_l_min = float(np.min(d2y_l))
            lower_tol = 3.0 * convexity_tolerance
            if d2y_l_min < -lower_tol:
                idx_bad = int(np.argmin(d2y_l))
                x_bad = psi_dense[idx_bad]
                return False, f"lower_concavity@x={x_bad:.2f}_d2y={d2y_l_min:.4f}"

        if max_curvature_roughness is not None:
            psi_dense = np.linspace(0.05, 0.95, 80)
            y_u_dense = np.interp(psi_dense, x_u, y_u)
            d2y_u = np.gradient(np.gradient(y_u_dense, psi_dense), psi_dense)
            tv_d2y = float(np.sum(np.abs(np.diff(d2y_u))))
            if tv_d2y > max_curvature_roughness:
                return False, f"too_rough({tv_d2y:.2f}>{max_curvature_roughness:.1f})"

        return True, "ok"
