"""AirfoilProblem: conecta CST + XFoil ao pymoo.

x = [w_upper_0..w_upper_{n_u-1}, w_lower_0..w_lower_{n_l-1}]
Single: maximizar Cl/Cd → minimizar -Cl/Cd
Multi:  [-Cl, +Cd] → NSGA-II
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple
from pymoo.core.problem import ElementwiseProblem

from cst import CSTAirfoil
from xfoil_runner import XFoilRunner
from data_logger import DataLogger


class AirfoilProblem(ElementwiseProblem):
    def __init__(
        self,
        cst: CSTAirfoil,
        xfoil: XFoilRunner,
        logger: DataLogger,
        upper_bounds: Tuple[float, float] = (-0.20, 0.60),
        lower_bounds: Tuple[float, float] = (-0.60, 0.20),
        explicit_bounds: Tuple[np.ndarray, np.ndarray] | None = None,
        reynolds: float = 500_000.0,
        mach: float = 0.0,
        eval_mode: str = "fixed_alpha",
        alpha_fixed: float = 5.0,
        alpha_sweep: Tuple[float, float, float] = (0.0, 10.0, 1.0),
        min_thickness: float = 0.06,
        max_thickness: float = 0.25,
        max_te_thickness: float = 0.005,
        local_thickness_constraints: list | None = None,
        n_airfoil_points: int = 160,
        multi_objective: bool = False,
        algorithm_name: str = "DE",
        penalty: float = 1e6,
        verbose_eval: bool = False,
        pop_size: int = 30,
        max_realistic_cl_cd: float = 180.0,
        min_realistic_cd: float = 0.0045,
        spike_detection_ratio: float = 1.4,
        use_robust_fitness: bool = True,
        robust_fitness_top_k: int = 3,
        enforce_upper_convexity: bool = True,
        convexity_tolerance: float = 0.001,
        convexity_check_range: tuple = (0.05, 0.95),
        max_curvature_roughness: float = 50.0,
    ) -> None:
        self.cst = cst
        self.xfoil = xfoil
        self.logger = logger

        self.reynolds = reynolds
        self.mach = mach
        self.eval_mode = eval_mode
        self.alpha_fixed = alpha_fixed
        self.alpha_sweep = alpha_sweep

        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.max_te_thickness = max_te_thickness
        self.local_thickness_constraints = local_thickness_constraints
        self.n_airfoil_points = n_airfoil_points

        self.multi_objective = multi_objective
        self.algorithm_name = algorithm_name
        self.penalty = penalty
        self.verbose_eval = verbose_eval
        self.pop_size = pop_size

        self.max_realistic_cl_cd = max_realistic_cl_cd
        self.min_realistic_cd = min_realistic_cd
        self.spike_detection_ratio = spike_detection_ratio
        self.use_robust_fitness = use_robust_fitness
        self.robust_fitness_top_k = robust_fitness_top_k

        self.enforce_upper_convexity = enforce_upper_convexity
        self.convexity_tolerance = convexity_tolerance
        self.convexity_check_range = convexity_check_range
        self.max_curvature_roughness = max_curvature_roughness

        n_var = cst.n_upper + cst.n_lower
        if explicit_bounds is not None:
            xl, xu = explicit_bounds
            xl = np.asarray(xl, dtype=float)
            xu = np.asarray(xu, dtype=float)
            assert xl.shape == (n_var,) and xu.shape == (n_var,), \
                f"explicit_bounds devem ter shape ({n_var},), got {xl.shape}/{xu.shape}"
        else:
            xl = np.array([upper_bounds[0]] * cst.n_upper + [lower_bounds[0]] * cst.n_lower)
            xu = np.array([upper_bounds[1]] * cst.n_upper + [lower_bounds[1]] * cst.n_lower)

        n_obj = 2 if multi_objective else 1
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        # Geração é incrementada externamente pelo callback
        self.current_generation: int = 0
        self._gen_eval_idx: int = 0
        self._last_gen_seen: int = 0

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        if self.current_generation != self._last_gen_seen:
            self._gen_eval_idx = 0
            self._last_gen_seen = self.current_generation
        self._gen_eval_idx += 1

        airfoil_id = self.logger.next_id()
        w_upper = x[: self.cst.n_upper]
        w_lower = x[self.cst.n_upper :]

        try:
            xc, yc = self.cst.coords(
                w_upper, w_lower, n_points=self.n_airfoil_points
            )
        except Exception:
            self._set_penalty(out, airfoil_id, x, "geom_error")
            return

        ok, reason = self.cst.is_valid_geometry(
            xc, yc,
            min_thickness=self.min_thickness,
            max_thickness=self.max_thickness,
            max_te_thickness=self.max_te_thickness,
            local_thickness_constraints=self.local_thickness_constraints,
            enforce_upper_convexity=self.enforce_upper_convexity,
            convexity_tolerance=self.convexity_tolerance,
            convexity_check_range=self.convexity_check_range,
            max_curvature_roughness=self.max_curvature_roughness,
        )
        if not ok:
            # Salva mesmo assim, ajuda na análise depois
            self.logger.save_geometry(airfoil_id, xc, yc)
            self._set_penalty(out, airfoil_id, x, f"invalid:{reason}")
            return

        dat_path = self.logger.save_geometry(airfoil_id, xc, yc)

        polar_data = []
        try:
            if self.eval_mode == "fixed_alpha":
                cl_, cd_ = self.xfoil.run_alpha(
                    dat_path, self.alpha_fixed, self.reynolds, self.mach
                )
                if cl_ is not None:
                    polar_data = [(self.alpha_fixed, cl_, cd_)]
            else:
                a0, a1, da = self.alpha_sweep
                polar_data = self.xfoil.run_alpha_sweep(
                    dat_path, a0, a1, da, self.reynolds, self.mach
                )
        except Exception:
            polar_data = []

        if polar_data:
            self.logger.save_polar(airfoil_id, polar_data)

        cl, cd, alpha_best, sanity_reason = self._compute_robust_fitness(polar_data)
        if cl is None:
            self._set_penalty(out, airfoil_id, x, sanity_reason)
            return

        fitness_log = cl / cd

        if self.multi_objective:
            out["F"] = np.array([-cl, cd])
        else:
            out["F"] = np.array([-fitness_log])

        self.logger.log_evaluation(
            airfoil_id=airfoil_id,
            algorithm=self.algorithm_name,
            generation=self.current_generation,
            cst_weights=list(map(float, x)),
            cl=float(cl),
            cd=float(cd),
            fitness=float(fitness_log),
            status="ok",
        )
        self._print_progress(airfoil_id, ok=True, fitness=fitness_log,
                             alpha=alpha_best, cl=cl, cd=cd)

    def _print_progress(
        self,
        airfoil_id: str,
        ok: bool,
        reason: str = "",
        fitness: float = 0.0,
        alpha: float | None = None,
        cl: float = 0.0,
        cd: float = 0.0,
    ) -> None:
        if not self.verbose_eval:
            return
        gen = self.current_generation
        idx = self._gen_eval_idx
        pop = self.pop_size
        prefix = f"  [Gen {gen:>2} | {self.algorithm_name:<3}] eval {idx:>3}/{pop} {airfoil_id}"
        if ok:
            a_str = f"alpha={alpha:+5.1f}deg" if alpha is not None else ""
            print(f"{prefix} -> Cl={cl:+.4f} Cd={cd:.5f} Cl/Cd={fitness:7.2f}  {a_str}  [OK]")
        else:
            print(f"{prefix} -> [FAIL] {reason}")

    def _compute_robust_fitness(
        self, polar_data: list
    ) -> tuple:
        """Aplica sanity checks ao polar. Retorna (Cl, Cd, alpha, motivo) ou (None,...,motivo)."""
        if not polar_data:
            return None, None, None, "xfoil_no_data"

        arr = np.array(polar_data, dtype=float)
        if arr.shape[0] == 0:
            return None, None, None, "xfoil_no_data"

        # Filtra Cd minimamente crível (descarta bolha laminar idealizada do XFoil)
        valid_mask = (arr[:, 2] >= self.min_realistic_cd) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
        arr_valid = arr[valid_mask]
        if arr_valid.shape[0] == 0:
            return None, None, None, f"all_cd_below_{self.min_realistic_cd:.4f}"

        cl_cd = arr_valid[:, 1] / arr_valid[:, 2]

        order = np.argsort(-cl_cd)
        cl_cd_sorted = cl_cd[order]
        arr_sorted = arr_valid[order]

        # Detecção de spike: top1 muito acima do top3 = artefato numérico
        if len(cl_cd_sorted) >= 3 and cl_cd_sorted[2] > 0:
            ratio = cl_cd_sorted[0] / cl_cd_sorted[2]
            if ratio > self.spike_detection_ratio:
                return None, None, None, f"spike(ratio={ratio:.2f})"

        # Polares com zigzag em Cl(alpha) antes do stall = XFoil oscilando
        # entre soluções diferentes -> geometria patológica
        if len(arr) >= 5:
            arr_sorted_alpha = arr[arr[:, 0].argsort()]
            alphas = arr_sorted_alpha[:, 0]
            cls = arr_sorted_alpha[:, 1]
            i_pico = int(np.argmax(cls))
            if i_pico >= 3:
                dcl = np.diff(cls[:i_pico + 1])
                n_decresc = int(np.sum(dcl < -0.01))
                frac_decresc = n_decresc / max(1, len(dcl))
                if frac_decresc > 0.25:
                    return None, None, None, f"polar_zigzag(frac={frac_decresc:.2f})"

        # Fitness robusto: média dos top-K (preferir pico amplo a pico fino)
        if self.use_robust_fitness:
            k = min(self.robust_fitness_top_k, len(cl_cd_sorted))
            fitness = float(np.mean(cl_cd_sorted[:k]))
            cl_rep = float(np.mean(arr_sorted[:k, 1]))
            cd_rep = float(np.mean(arr_sorted[:k, 2]))
            alpha_rep = float(arr_sorted[0, 0])
        else:
            fitness = float(cl_cd_sorted[0])
            cl_rep = float(arr_sorted[0, 1])
            cd_rep = float(arr_sorted[0, 2])
            alpha_rep = float(arr_sorted[0, 0])

        if fitness > self.max_realistic_cl_cd:
            return None, None, None, f"cl_cd_too_high({fitness:.0f}>{self.max_realistic_cl_cd:.0f})"

        if cd_rep <= 0 or not np.isfinite(fitness):
            return None, None, None, "fitness_invalid"

        return cl_rep, cd_rep, alpha_rep, "ok"

    def _set_penalty(self, out: dict, airfoil_id: str, x: np.ndarray, reason: str) -> None:
        if self.multi_objective:
            out["F"] = np.array([self.penalty, self.penalty])
        else:
            out["F"] = np.array([self.penalty])
        self.logger.log_evaluation(
            airfoil_id=airfoil_id,
            algorithm=self.algorithm_name,
            generation=self.current_generation,
            cst_weights=list(map(float, x)),
            cl=0.0,
            cd=0.0,
            fitness=0.0,
            status=reason,
        )
        self._print_progress(airfoil_id, ok=False, reason=reason)
