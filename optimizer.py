"""Wrapper das metaheurísticas via pymoo (DE, GA, CMA-ES, NSGA-II).
Inicialização por LHS (ou warm start + LHS).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.termination import get_termination

from problem import AirfoilProblem


class WarmStartLHSSampling(Sampling):
    """Híbrido: parte da pop começa perto do baseline, restante via LHS.

    LHS puro num espaço amplo gera muito indivíduo patológico que o XFoil
    não converge — a evolução queima avaliações sem informação útil.
    O warm start ancora a busca a uma região fisicamente razoável.
    """

    def __init__(self, baseline: np.ndarray, n_warm: int, noise_std: float = 0.02):
        super().__init__()
        self.baseline = np.asarray(baseline, dtype=float)
        self.n_warm = max(0, int(n_warm))
        self.noise_std = float(noise_std)
        self._lhs = LHS()

    def _do(self, problem, n_samples: int, **kwargs) -> np.ndarray:
        n_var = len(self.baseline)
        n_warm = min(self.n_warm, n_samples)
        n_lhs = n_samples - n_warm

        if n_warm > 0:
            warm = self.baseline[None, :] + np.random.normal(
                0.0, self.noise_std, size=(n_warm, n_var)
            )
            warm = np.clip(warm, problem.xl, problem.xu)
        else:
            warm = np.empty((0, n_var))

        if n_lhs > 0:
            lhs_samples = self._lhs._do(problem, n_lhs, **kwargs)
        else:
            lhs_samples = np.empty((0, n_var))

        return np.vstack([warm, lhs_samples])


@dataclass
class OptimizationResult:
    algorithm: str
    best_x: np.ndarray
    best_f: float
    history_best_per_eval: List[float] = field(default_factory=list)
    n_evaluations: int = 0
    wall_time_s: float = 0.0
    pareto_front: np.ndarray = None
    seed: int | None = None


class ConvergenceCallback(Callback):
    """Atualiza generation no Problem e registra o melhor histórico."""

    def __init__(self, problem: AirfoilProblem) -> None:
        super().__init__()
        self.problem = problem
        self.history: List[float] = []
        self._best_so_far: float = -np.inf

    def notify(self, algorithm) -> None:
        self.problem.current_generation = algorithm.n_gen

        F = algorithm.pop.get("F")
        if F is None or len(F) == 0:
            return

        # Detecção precoce: se 100% da gen 1 falhou, XFoil provavelmente
        # não está respondendo. Avisa mas não interrompe.
        if algorithm.n_gen == 1:
            penalty = self.problem.penalty
            n_failed = int(np.sum(F[:, 0] >= penalty * 0.99))
            n_total = len(F)
            if n_failed == n_total:
                print("\n" + "!" * 72)
                print("!  ATENCAO: 100% das avaliacoes da geracao 1 falharam.")
                print("!  Provavelmente o XFoil nao esta convergindo / nao foi chamado.")
                print("!  Recomendacao: ative XFOIL_DEBUG=True em config.py e re-execute,")
                print("!  ou rode o diagnostico verboso no inicio do main.py.")
                print("!" * 72 + "\n")
            elif n_failed > 0.7 * n_total:
                print(f"\n[aviso] {n_failed}/{n_total} avaliacoes falharam na gen 1 — taxa alta.\n")

        if self.problem.multi_objective:
            cl_arr = -F[:, 0]
            cd_arr = F[:, 1]
            valid = cd_arr > 1e-9
            if np.any(valid):
                ratio = np.full_like(cd_arr, -np.inf)
                ratio[valid] = cl_arr[valid] / cd_arr[valid]
                gen_best = float(np.max(ratio))
            else:
                gen_best = -np.inf
        else:
            gen_best = float(-np.min(F[:, 0]))

        self._best_so_far = max(self._best_so_far, gen_best)
        # Replica o melhor pra cada avaliação da geração (facilita plotar)
        n_new = len(F)
        self.history.extend([self._best_so_far] * n_new)


class AirfoilOptimizer:
    def __init__(
        self,
        problem: AirfoilProblem,
        pop_size: int = 30,
        n_max_eval: int = 750,
        seed: int = 42,
        verbose: bool = True,
        baseline_weights: np.ndarray | None = None,
        warm_start_fraction: float = 0.0,
        warm_noise_std: float = 0.02,
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.n_max_eval = n_max_eval
        self.seed = seed
        self.verbose = verbose
        self.baseline_weights = baseline_weights
        self.warm_start_fraction = float(warm_start_fraction)
        self.warm_noise_std = float(warm_noise_std)

    def _build_sampling(self):
        if (
            self.baseline_weights is not None
            and self.warm_start_fraction > 0.0
        ):
            n_warm = max(1, int(round(self.pop_size * self.warm_start_fraction)))
            return WarmStartLHSSampling(
                baseline=self.baseline_weights,
                n_warm=n_warm,
                noise_std=self.warm_noise_std,
            )
        return LHS()

    def _build_algorithm(self, name: str, hp: Dict):
        sampling = self._build_sampling()

        if name == "DE":
            return DE(
                pop_size=self.pop_size,
                sampling=sampling,
                variant=hp.get("variant", "DE/rand/1/bin"),
                CR=hp.get("CR", 0.9),
                F=hp.get("F", 0.7),
            )
        if name == "GA":
            return GA(
                pop_size=self.pop_size,
                sampling=sampling,
                eliminate_duplicates=True,
            )
        if name == "CMAES":
            # CMA-ES tem sampling próprio (matriz de covariância adaptativa).
            # Warm start é via x0 = baseline e sigma0 = passo inicial.
            x0 = (self.baseline_weights
                  if self.baseline_weights is not None
                  else None)
            return CMAES(
                x0=x0,
                sigma=hp.get("sigma0", 0.10),
                popsize=self.pop_size,
                restarts=hp.get("restarts", 0),
                bipop=hp.get("bipop", False),
                normalize=False,
            )
        if name == "NSGA2":
            return NSGA2(
                pop_size=self.pop_size,
                sampling=sampling,
                eliminate_duplicates=True,
            )
        raise ValueError(f"Algoritmo desconhecido: {name}")

    def run(self, algorithm_name: str, hp: Dict | None = None) -> OptimizationResult:
        hp = hp or {}
        self.problem.algorithm_name = algorithm_name

        algo = self._build_algorithm(algorithm_name, hp)
        callback = ConvergenceCallback(self.problem)
        termination = get_termination("n_eval", self.n_max_eval)

        t0 = time.perf_counter()
        res = minimize(
            self.problem,
            algo,
            termination=termination,
            seed=self.seed,
            callback=callback,
            verbose=self.verbose,
            save_history=False,
        )
        elapsed = time.perf_counter() - t0

        if self.problem.multi_objective:
            pf = res.F.copy()
            pf[:, 0] = -pf[:, 0]
            best_idx = np.argmax(pf[:, 0] / np.maximum(pf[:, 1], 1e-9))
            best_x = res.X[best_idx]
            best_f = float(pf[best_idx, 0] / pf[best_idx, 1])
            pareto = pf
        else:
            best_x = res.X
            best_f = float(-res.F[0])
            pareto = None

        return OptimizationResult(
            algorithm=algorithm_name,
            best_x=np.asarray(best_x),
            best_f=best_f,
            history_best_per_eval=callback.history,
            n_evaluations=self.problem.logger.total_evaluations,
            wall_time_s=elapsed,
            pareto_front=pareto,
            seed=self.seed,
        )
