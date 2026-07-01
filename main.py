"""Roda DE/GA/CMA-ES por seed e gera os plots."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import config as CFG
from cst import CSTAirfoil, naca4_coords, load_dat_file, save_dat_file
from xfoil_runner import XFoilRunner
from data_logger import DataLogger
from problem import AirfoilProblem
from optimizer import AirfoilOptimizer, OptimizationResult
import plotter


def banner(msg: str) -> None:
    print("\n" + "=" * 76)
    print(f"  {msg}")
    print("=" * 76)


def load_or_create_baseline(cst: CSTAirfoil) -> tuple:
    """Carrega o .dat baseline (ou gera um NACA 0012 analítico se não achar)."""
    dat_path = CFG.BASELINE_AIRFOIL_DAT
    if dat_path.exists():
        x, y = load_dat_file(dat_path)
        print(f"[baseline] Carregado de {dat_path}")
    else:
        print("[baseline] .dat não encontrado, gerando NACA 0012 analiticamente.")
        x, y = naca4_coords("0012", n=CFG.N_AIRFOIL_POINTS)
        save_dat_file(dat_path, x, y, name="NACA0012")

    w_u, w_l, yTE_u, yTE_l = cst.fit_to_coords(x, y)
    print(f"[baseline] Pesos CST extradorso: {np.round(w_u, 4)}")
    print(f"[baseline] Pesos CST intradorso: {np.round(w_l, 4)}")
    print(f"[baseline] Espessura BF: {abs(yTE_u - yTE_l):.5f}")
    return x, y, w_u, w_l


def run_single_optimization(
    alg_name: str,
    seed: int,
    cst: CSTAirfoil,
    xfoil: XFoilRunner,
    baseline_weights: np.ndarray,
    explicit_bounds: tuple | None,
    seed_dir: Path,
) -> OptimizationResult:
    algo_log_dir = seed_dir / alg_name
    logger = DataLogger(
        geometry_dir=algo_log_dir / "geometries",
        polar_dir=algo_log_dir / "polars",
        csv_path=algo_log_dir / "evaluation_log.csv",
    )

    problem = AirfoilProblem(
        cst=cst,
        xfoil=xfoil,
        logger=logger,
        upper_bounds=CFG.CST_UPPER_BOUNDS,
        lower_bounds=CFG.CST_LOWER_BOUNDS,
        explicit_bounds=explicit_bounds,
        reynolds=CFG.REYNOLDS,
        mach=CFG.MACH,
        eval_mode=CFG.EVAL_MODE,
        alpha_fixed=CFG.ALPHA_FIXED,
        alpha_sweep=CFG.ALPHA_SWEEP_RANGE,
        min_thickness=CFG.MIN_THICKNESS,
        max_thickness=CFG.MAX_THICKNESS,
        max_te_thickness=CFG.MAX_TE_THICKNESS,
        local_thickness_constraints=CFG.LOCAL_THICKNESS_CONSTRAINTS,
        n_airfoil_points=CFG.N_AIRFOIL_POINTS,
        multi_objective=(CFG.OPT_MODE == "multi"),
        algorithm_name=alg_name,
        penalty=CFG.PENALTY_VALUE,
        verbose_eval=CFG.VERBOSE_EVAL,
        pop_size=CFG.POP_SIZE,
        max_realistic_cl_cd=CFG.MAX_REALISTIC_CL_CD,
        min_realistic_cd=CFG.MIN_REALISTIC_CD,
        spike_detection_ratio=CFG.SPIKE_DETECTION_RATIO,
        use_robust_fitness=CFG.USE_ROBUST_FITNESS,
        robust_fitness_top_k=CFG.ROBUST_FITNESS_TOP_K,
        enforce_upper_convexity=CFG.ENFORCE_UPPER_CONVEXITY,
        convexity_tolerance=CFG.CONVEXITY_TOLERANCE,
        convexity_check_range=CFG.CONVEXITY_CHECK_RANGE,
        max_curvature_roughness=CFG.MAX_CURVATURE_ROUGHNESS,
    )

    opt = AirfoilOptimizer(
        problem=problem,
        pop_size=CFG.POP_SIZE,
        n_max_eval=CFG.N_MAX_EVAL,
        seed=seed,
        verbose=False,
        baseline_weights=baseline_weights,
        warm_start_fraction=CFG.WARM_START_FRACTION,
        warm_noise_std=CFG.WARM_START_NOISE_STD,
    )

    if alg_name == "DE":
        hp = dict(variant=CFG.DE_VARIANT, F=CFG.DE_F, CR=CFG.DE_CR)
    elif alg_name == "CMAES":
        hp = dict(sigma0=CFG.CMAES_SIGMA0,
                  restarts=CFG.CMAES_RESTARTS,
                  bipop=CFG.CMAES_BIPOP)
    else:
        hp = {}

    run_alg = "NSGA2" if CFG.OPT_MODE == "multi" else alg_name
    result = opt.run(run_alg, hp)
    result.algorithm = alg_name
    return result


def main() -> None:
    CFG.ensure_dirs()
    banner(f"OTIMIZAÇÃO AERODINÂMICA 2D — DE / GA / CMA-ES  (TCC)")
    print(f"Re = {CFG.REYNOLDS:.0f}  |  M = {CFG.MACH}")
    print(f"Pesos CST: {CFG.N_CST_UPPER} (sup) + {CFG.N_CST_LOWER} (inf)  =  {CFG.N_CST_UPPER+CFG.N_CST_LOWER} variáveis")
    print(f"Pop = {CFG.POP_SIZE} | N_max_eval = {CFG.N_MAX_EVAL}")
    print(f"Modo de avaliação: {CFG.EVAL_MODE}  "
          f"(sweep alpha = {CFG.ALPHA_SWEEP_RANGE})" if CFG.EVAL_MODE == "alpha_sweep"
          else f"Modo de avaliação: {CFG.EVAL_MODE}  (alpha fixo = {CFG.ALPHA_FIXED}°)")
    if CFG.XFOIL_VISIBLE:
        print("Modo XFOIL_VISIBLE = True (lento; use False p/ rodada de produção)")
    if CFG.USE_RELATIVE_BOUNDS:
        print(f"Bounds RELATIVOS ao baseline (delta=±{CFG.CST_BOUNDS_DELTA})")
    if CFG.WARM_START_FRACTION > 0:
        print(f"Warm start: {int(CFG.WARM_START_FRACTION*100)}% da pop iniciada perto do baseline")
    print(f"Sanity checks anti-overfit: max Cl/Cd={CFG.MAX_REALISTIC_CL_CD:.0f}, "
          f"min Cd={CFG.MIN_REALISTIC_CD:.4f}, "
          f"fitness=top-{CFG.ROBUST_FITNESS_TOP_K} medio")

    seeds_to_run = CFG.SEEDS[: CFG.N_SEEDS]
    print(f"\nSeeds a executar: {seeds_to_run}  ({CFG.N_SEEDS} corridas por algoritmo)")
    n_total_runs = CFG.N_SEEDS * len(CFG.ALGORITHMS_TO_RUN)
    n_total_evals = n_total_runs * CFG.N_MAX_EVAL
    print(f"Total: {n_total_runs} corridas, ~{n_total_evals} avaliações XFoil")

    cst = CSTAirfoil(
        n_upper=CFG.N_CST_UPPER, n_lower=CFG.N_CST_LOWER,
        N1=CFG.CST_N1, N2=CFG.CST_N2,
    )
    xfoil = XFoilRunner(
        xfoil_path=CFG.DEFAULT_XFOIL_PATH,
        work_dir=CFG.GEOMETRY_DIR,
        timeout=CFG.XFOIL_TIMEOUT_S,
        n_iter=CFG.XFOIL_ITER,
        ncrit=CFG.NCRIT,
        xtr_top=CFG.XTR_TOP, xtr_bot=CFG.XTR_BOT,
        debug=CFG.XFOIL_DEBUG,
        visible=CFG.XFOIL_VISIBLE, visible_delay=CFG.XFOIL_VISIBLE_DELAY,
    )

    x_base, y_base, w_u0, w_l0 = load_or_create_baseline(cst)
    baseline_weights = np.concatenate([w_u0, w_l0])

    # Avaliação aerodinâmica do baseline
    banner("Avaliando o aerofólio BASELINE no XFoil")
    cl0, cd0, alpha_best0 = None, None, None
    if CFG.EVAL_MODE == "alpha_sweep":
        a0, a1, da = CFG.ALPHA_SWEEP_RANGE
        polar_b = xfoil.run_alpha_sweep(
            CFG.BASELINE_AIRFOIL_DAT, a0, a1, da, CFG.REYNOLDS, CFG.MACH
        )
        if polar_b:
            best = max(polar_b, key=lambda r: r[1] / r[2] if r[2] > 0 else -1e9)
            alpha_best0, cl0, cd0 = best
    else:
        cl0, cd0 = xfoil.run_alpha(
            CFG.BASELINE_AIRFOIL_DAT, CFG.ALPHA_FIXED, CFG.REYNOLDS, CFG.MACH
        )
        alpha_best0 = CFG.ALPHA_FIXED

    if cl0 is not None and cd0 > 0:
        print(f"  Baseline NACA 0012: alpha={alpha_best0:.1f}°  "
              f"Cl={cl0:.4f}  Cd={cd0:.5f}  Cl/Cd={cl0/cd0:.2f}")
    else:
        print("  Baseline: XFoil falhou. Acionando diagnóstico...")
        ok = xfoil.diagnose(CFG.BASELINE_AIRFOIL_DAT,
                            alpha=5.0, reynolds=CFG.REYNOLDS, mach=CFG.MACH)
        if not ok:
            print("\n!! Diagnóstico falhou. Verifique caminho do XFoil em config.py.")
            sys.exit(1)

    if CFG.USE_RELATIVE_BOUNDS:
        delta = CFG.CST_BOUNDS_DELTA
        explicit_bounds = (baseline_weights - delta, baseline_weights + delta)
    else:
        explicit_bounds = None

    # --- Loop principal: cada combinação (seed, algoritmo) ---
    all_results: Dict[str, List[OptimizationResult]] = {
        alg: [] for alg in CFG.ALGORITHMS_TO_RUN
    }

    for i_seed, seed in enumerate(seeds_to_run):
        banner(f"SEED {seed}  ({i_seed+1}/{CFG.N_SEEDS})")
        seed_dir = CFG.OUTPUT_DIR / f"seed_{seed:04d}"

        for alg_name in CFG.ALGORITHMS_TO_RUN:
            print(f"\n  >>> Executando {alg_name} (seed={seed})...")
            result = run_single_optimization(
                alg_name=alg_name, seed=seed,
                cst=cst, xfoil=xfoil,
                baseline_weights=baseline_weights,
                explicit_bounds=explicit_bounds,
                seed_dir=seed_dir,
            )
            all_results[alg_name].append(result)

            w_u_b = result.best_x[: cst.n_upper]
            w_l_b = result.best_x[cst.n_upper :]
            xb_, yb_ = cst.coords(w_u_b, w_l_b, n_points=CFG.N_AIRFOIL_POINTS)
            save_dat_file(
                seed_dir / alg_name / f"BEST_{alg_name}_seed{seed}.dat",
                xb_, yb_, name=f"BEST_{alg_name}_seed{seed}",
            )

            print(f"      [{alg_name}/seed{seed}]  best Cl/Cd = {result.best_f:.3f}  "
                  f"|  evals = {result.n_evaluations}  "
                  f"|  tempo = {result.wall_time_s:.1f}s")

    # --- Resumo estatístico ---
    banner("RESUMO ESTATÍSTICO POR ALGORITMO")
    if cl0 is not None and cd0 > 0:
        ref = cl0 / cd0
        print(f"  Baseline NACA 0012:  Cl/Cd = {ref:.2f}\n")
    else:
        ref = None

    summary_rows = []
    for alg, results_list in all_results.items():
        fitnesses = np.array([r.best_f for r in results_list])
        times = np.array([r.wall_time_s for r in results_list])
        print(f"  {alg:<6}  Cl/Cd  best={np.max(fitnesses):7.2f}  "
              f"mean={np.mean(fitnesses):7.2f}  std={np.std(fitnesses):6.2f}")
        print(f"          tempo  mean={np.mean(times):.1f}s  "
              f"({len(fitnesses)} corridas)")
        summary_rows.append(dict(
            Algoritmo=alg,
            Best=float(np.max(fitnesses)),
            Mean=float(np.mean(fitnesses)),
            Std=float(np.std(fitnesses)),
            Min=float(np.min(fitnesses)),
            Median=float(np.median(fitnesses)),
            TempoMedio=float(np.mean(times)),
            NSeeds=len(fitnesses),
        ))

    pd.DataFrame(summary_rows).to_csv(
        CFG.OUTPUT_DIR / "summary_stats.csv", index=False
    )

    # Melhor seed por algoritmo (usada nos plots de geometria)
    best_per_alg = {
        alg: max(rs, key=lambda r: r.best_f)
        for alg, rs in all_results.items()
    }

    # --- Re-avaliação dos ótimos com sweep fino ---
    banner("Re-avaliando os perfis ótimos com varredura fina de alpha")
    sweep_fine = (-3.0, 14.0, 0.5)
    a0f, a1f, daf = sweep_fine
    print(f"  Sweep fino: alpha de {a0f}° a {a1f}° (passo {daf}°)")

    print("  Calculando polar fino do BASELINE...")
    polar_baseline = xfoil.run_alpha_sweep(
        CFG.BASELINE_AIRFOIL_DAT, a0f, a1f, daf, CFG.REYNOLDS, CFG.MACH
    )
    print(f"    -> {len(polar_baseline)} pontos validos")

    polars_optimized: Dict[str, list] = {}
    polars_per_seed: Dict[str, list] = {}
    geometries_per_seed: Dict[str, list] = {}
    for alg, results_list in all_results.items():
        polars_per_seed[alg] = []
        geometries_per_seed[alg] = []
        for r in results_list:
            w_u_b = r.best_x[: cst.n_upper]
            w_l_b = r.best_x[cst.n_upper :]
            xb_, yb_ = cst.coords(w_u_b, w_l_b, n_points=CFG.N_AIRFOIL_POINTS)
            tmp_dat = CFG.OUTPUT_DIR / f"_eval_{alg}_seed{r.seed}.dat"
            save_dat_file(tmp_dat, xb_, yb_, name=f"{alg}_seed{r.seed}")
            print(f"  Polar fino de {alg}/seed{r.seed} (Cl/Cd={r.best_f:.1f})...")
            polar = xfoil.run_alpha_sweep(tmp_dat, a0f, a1f, daf,
                                           CFG.REYNOLDS, CFG.MACH)
            print(f"    -> {len(polar)} pontos validos")
            polars_per_seed[alg].append((polar, r.seed, r.best_f))
            geometries_per_seed[alg].append((xb_, yb_, r.seed, r.best_f))
            try: tmp_dat.unlink()
            except Exception: pass

        best_r = best_per_alg[alg]
        for polar, seed, fit in polars_per_seed[alg]:
            if seed == best_r.seed:
                polars_optimized[alg] = polar
                break

    # --- Gráficos ---
    banner("Gerando gráficos comparativos")
    plot_dir = CFG.PLOT_DIR

    # Logs da melhor seed de cada algoritmo (coerente com plots de geometria)
    eval_logs: Dict[str, pd.DataFrame] = {}
    for alg, r in best_per_alg.items():
        seed_used = r.seed
        csv_path = (CFG.OUTPUT_DIR / f"seed_{seed_used:04d}"
                    / alg / "evaluation_log.csv")
        if csv_path.exists():
            eval_logs[alg] = pd.read_csv(csv_path)
            print(f"  [eval_logs] {alg}: usando log da seed {seed_used} "
                  f"(best Cl/Cd = {r.best_f:.2f})")

    plotter.plot_convergence_scatter(eval_logs, plot_dir / "01_convergence.png")
    print(f"  -> 01_convergence.png            (nuvem + melhor historico, melhor seed)")

    runtimes_mean = {alg: float(np.mean([r.wall_time_s for r in rs]))
                     for alg, rs in all_results.items()}
    n_evals_mean = {alg: int(np.mean([r.n_evaluations for r in rs]))
                    for alg, rs in all_results.items()}
    best_fitness_max = {alg: float(np.max([r.best_f for r in rs]))
                        for alg, rs in all_results.items()}
    plotter.plot_comparative_runtime(
        runtimes=runtimes_mean, n_evals=n_evals_mean,
        best_fitness=best_fitness_max,
        save_path=plot_dir / "02_comparative_runtime.png",
    )
    print(f"  -> 02_comparative_runtime.png    (custo medio + melhor Cl/Cd)")

    optimized_geoms = {}
    for alg, r in best_per_alg.items():
        w_u_b = r.best_x[: cst.n_upper]
        w_l_b = r.best_x[cst.n_upper :]
        xo, yo = cst.coords(w_u_b, w_l_b, n_points=CFG.N_AIRFOIL_POINTS)
        optimized_geoms[alg] = (xo, yo)
    plotter.plot_geometry_comparison(
        baseline=(x_base, y_base), optimized=optimized_geoms,
        save_path=plot_dir / "03_geometry.png",
        baseline_label="NACA 0012 (baseline)",
    )
    print(f"  -> 03_geometry.png               (perfis baseline x melhores otimizados)")

    plotter.plot_polar_comparison(
        polars=polars_optimized, baseline_polar=polar_baseline,
        save_path=plot_dir / "04_polar.png",
    )
    print(f"  -> 04_polar.png                  (polar Cl x Cd completa)")

    plotter.plot_efficiency_curves(
        polars=polars_optimized, baseline_polar=polar_baseline,
        save_path=plot_dir / "05_efficiency.png",
    )
    print(f"  -> 05_efficiency.png             (Cl-alpha e Cl/Cd-alpha)")

    plotter.plot_population_boxplot(eval_logs, plot_dir / "06_population_boxplot.png")
    print(f"  -> 06_population_boxplot.png     (distribuicao por geracao)")
    plotter.plot_success_rate(eval_logs, plot_dir / "07_success_rate.png")
    print(f"  -> 07_success_rate.png           (% avals validas)")
    plotter.plot_population_diversity(eval_logs, plot_dir / "08_diversity.png")
    print(f"  -> 08_diversity.png              (diversidade dos pesos CST)")

    # Análise estatística só faz sentido com mais de uma seed
    if CFG.N_SEEDS > 1:
        plotter.plot_seeds_boxplot(
            all_results,
            save_path=plot_dir / "09_seeds_boxplot.png",
            baseline_value=ref,
        )
        print(f"  -> 09_seeds_boxplot.png          (robustez entre seeds)")

        plotter.plot_convergence_with_uncertainty(
            all_results,
            save_path=plot_dir / "10_convergence_uncertainty.png",
        )
        print(f"  -> 10_convergence_uncertainty.png  (media +/- std entre seeds)")

        plotter.plot_statistical_comparison(
            all_results,
            save_path=plot_dir / "11_statistical_comparison.png",
        )
        print(f"  -> 11_statistical_comparison.png   (Mann-Whitney p-values)")

        plotter.plot_geometries_per_seed(
            geometries_per_alg=geometries_per_seed,
            baseline=(x_base, y_base),
            save_path=plot_dir / "12_geometries_per_seed.png",
        )
        print(f"  -> 12_geometries_per_seed.png       (overlay de geometrias por seed)")

        plotter.plot_polars_per_seed(
            polars_per_alg=polars_per_seed,
            baseline_polar=polar_baseline,
            save_path=plot_dir / "13_polars_per_seed.png",
        )
        print(f"  -> 13_polars_per_seed.png           (polar Cl x Cd por seed)")

        plotter.plot_convergence_per_seed(
            all_results,
            save_path=plot_dir / "14_convergence_per_seed.png",
        )
        print(f"  -> 14_convergence_per_seed.png      (curvas de convergencia por seed)")

        plotter.plot_effect_size_analysis(
            all_results,
            save_path=plot_dir / "15_effect_size.png",
        )
        print(f"  -> 15_effect_size.png               (Cliff's delta + IC bootstrap)")
    else:
        print("\n  [info] N_SEEDS=1: gráficos estatísticos pulados.")
        print("         Rode com N_SEEDS=5 em config.py p/ análise estatística.")

    if CFG.OPT_MODE == "multi":
        pareto_data = {alg: r.pareto_front for alg, r in best_per_alg.items()
                       if r.pareto_front is not None}
        if pareto_data:
            plotter.plot_pareto(pareto_data, plot_dir / "12_pareto.png")
            print(f"  -> 12_pareto.png                 (fronteira de Pareto)")

    # --- Resumo final ---
    banner("RESUMO FINAL")
    if ref is not None:
        print(f"  Baseline NACA 0012:  Cl/Cd = {ref:.2f}\n")
    for alg, rs in all_results.items():
        best_run = max(rs, key=lambda r: r.best_f)
        msg = f"  {alg:<6}  best Cl/Cd = {best_run.best_f:7.2f}  "
        if ref is not None and best_run.best_f > 0:
            gain = (best_run.best_f - ref) / ref * 100
            msg += f"|  ganho = {gain:+5.1f}%"
        print(msg)
    print(f"\nTodos os resultados em: {CFG.OUTPUT_DIR}")
    print(f"Plots em: {CFG.PLOT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrompido pelo usuário]")
        sys.exit(130)
