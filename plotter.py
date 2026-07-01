"""Geração de gráficos do projeto."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Estilo (sem LaTeX pra não exigir instalação extra)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "savefig.dpi": 200,
        "figure.dpi": 100,
    }
)

# Paleta colorblind-friendly (Wong, 2011)
ALG_COLORS: Dict[str, str] = {
    "DE":     "#0072B2",
    "GA":     "#D55E00",
    "CMAES":  "#009E73",
    "PSO":    "#009E73",
    "NSGA2":  "#CC79A7",
}
BASELINE_COLOR: str = "#000000"


def plot_convergence_scatter(
    eval_logs: Dict[str, pd.DataFrame],
    save_path: Path,
    log_y: bool = False,
) -> None:
    """Scatter de avaliações + linha do melhor histórico (uma curva por algoritmo)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for alg, df in eval_logs.items():
        if df is None or len(df) == 0:
            continue
        color = ALG_COLORS.get(alg, "gray")

        mask_ok = (df["Status"] == "ok") & (df["Fitness"] > 0)
        df_ok = df[mask_ok].copy()
        if len(df_ok) == 0:
            continue
        df_ok["eval_idx"] = np.arange(1, len(df) + 1)[mask_ok.values]

        ax.scatter(
            df_ok["eval_idx"], df_ok["Fitness"],
            color=color, s=10, alpha=0.25,
            edgecolors="none", zorder=1,
        )

        best_so_far = df_ok["Fitness"].cummax()
        ax.plot(
            df_ok["eval_idx"], best_so_far,
            color=color, linewidth=2.0,
            label=f"{alg} (melhor histórico)",
            drawstyle="steps-post", zorder=3,
        )

    ax.set_xlabel("Número da avaliação")
    ax.set_ylabel(r"$C_\ell / C_d$")
    ax.set_title("Convergência dos Algoritmos")
    if log_y:
        ax.set_yscale("log")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_comparative_runtime(
    runtimes: Dict[str, float],
    n_evals: Dict[str, int],
    best_fitness: Dict[str, float],
    save_path: Path,
) -> None:
    """Barras: tempo, nº de avaliações e melhor Cl/Cd lado a lado."""
    algs = list(runtimes.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    metrics = [
        (runtimes, "Tempo de execução [s]", "Custo computacional", "{:.1f}"),
        (n_evals, "Nº de avaliações XFoil", "Total de chamadas ao solver", "{:.0f}"),
        (best_fitness, r"Melhor $C_\ell / C_d$", "Qualidade da solução final", "{:.2f}"),
    ]
    for ax, (data, ylabel, title, fmt) in zip(axes, metrics):
        bars = ax.bar(
            algs, [data[a] for a in algs],
            color=[ALG_COLORS.get(a, "gray") for a in algs],
            edgecolor="black", linewidth=0.6,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    fmt.format(b.get_height()),
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("Análise comparativa entre metaheurísticas", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_geometry_comparison(
    baseline: Tuple[np.ndarray, np.ndarray],
    optimized: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Path,
    baseline_label: str = "Baseline",
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.6))

    xb, yb = baseline
    ax.plot(xb, yb, "k--", linewidth=1.6, label=baseline_label)

    for alg, (xo, yo) in optimized.items():
        ax.plot(
            xo, yo,
            color=ALG_COLORS.get(alg, None),
            linewidth=1.6,
            label=f"Otimizado ({alg})",
        )

    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$y/c$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Comparação geométrica: baseline × otimizados")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_polar_comparison(
    polars: Dict[str, List[Tuple[float, float, float]]],
    save_path: Path,
    baseline_polar: List[Tuple[float, float, float]] | None = None,
) -> None:
    """Curva polar (Cl × Cd) de cada perfil ótimo + baseline."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if baseline_polar:
        bp = np.array(baseline_polar)
        ax.plot(bp[:, 2], bp[:, 1], "k--",
                linewidth=1.6, label="NACA 0012 (baseline)",
                marker="s", markersize=3)

    for alg, polar in polars.items():
        if not polar:
            continue
        p = np.array(polar)
        ax.plot(
            p[:, 2], p[:, 1],
            color=ALG_COLORS.get(alg, None),
            linewidth=1.6, marker="o", markersize=3.5,
            label=f"Otimizado ({alg})",
        )

    ax.set_xlabel(r"$C_d$")
    ax.set_ylabel(r"$C_\ell$")
    ax.set_title(r"Polar $C_\ell \times C_d$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_curves(
    polars: Dict[str, List[Tuple[float, float, float]]],
    save_path: Path,
    baseline_polar: List[Tuple[float, float, float]] | None = None,
) -> None:
    """Cl × α e Cl/Cd × α — análise dos perfis ótimos."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))

    def _plot(ax, data, color, label, ls="-", marker="o"):
        a = data[:, 0]
        cl = data[:, 1]
        cd = data[:, 2]
        if ax is ax1:
            ax.plot(a, cl, color=color, linewidth=1.6, marker=marker,
                    markersize=3.5, label=label, linestyle=ls)
        else:
            valid = cd > 1e-6
            ax.plot(a[valid], cl[valid] / cd[valid], color=color,
                    linewidth=1.6, marker=marker, markersize=3.5,
                    label=label, linestyle=ls)

    if baseline_polar:
        _plot(ax1, np.array(baseline_polar), BASELINE_COLOR,
              "NACA 0012 (baseline)", ls="--", marker="s")
        _plot(ax2, np.array(baseline_polar), BASELINE_COLOR,
              "NACA 0012 (baseline)", ls="--", marker="s")

    for alg, polar in polars.items():
        if not polar:
            continue
        color = ALG_COLORS.get(alg, "gray")
        _plot(ax1, np.array(polar), color, f"Otimizado ({alg})")
        _plot(ax2, np.array(polar), color, f"Otimizado ({alg})")

    ax1.set_xlabel(r"$\alpha$ [deg]")
    ax1.set_ylabel(r"$C_\ell$")
    ax1.set_title(r"Curva $C_\ell \times \alpha$")

    ax2.set_xlabel(r"$\alpha$ [deg]")
    ax2.set_ylabel(r"$C_\ell / C_d$")
    ax2.set_title(r"Eficiência aerodinâmica $C_\ell/C_d \times \alpha$")

    # Legenda unica na base, fora da area de dados (as series sao iguais nos dois paineis)
    handles, labels = ax1.get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Análise dos perfis otimizados", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_polar_and_efficiency(
    polars: Dict[str, List[Tuple[float, float, float]]],
    save_path: Path,
    baseline_polar: List[Tuple[float, float, float]] | None = None,
) -> None:
    """Combina 04 e 05 numa figura unica com 3 paineis:
    polar Cl x Cd, curva Cl x alpha e eficiencia Cl/Cd x alpha."""
    fig, (axp, axl, axe) = plt.subplots(1, 3, figsize=(16.5, 7.0))

    # Painel 1 -- polar Cl x Cd
    if baseline_polar:
        bp = np.array(baseline_polar)
        axp.plot(bp[:, 2], bp[:, 1], "--", color=BASELINE_COLOR, linewidth=1.6,
                 marker="s", markersize=3, label="NACA 0012 (baseline)")
    for alg, polar in polars.items():
        if not polar:
            continue
        p = np.array(polar)
        axp.plot(p[:, 2], p[:, 1], color=ALG_COLORS.get(alg, "gray"),
                 linewidth=1.6, marker="o", markersize=3.5,
                 label=f"Otimizado ({alg})")
    axp.set_xlabel(r"$C_d$")
    axp.set_ylabel(r"$C_\ell$")
    axp.set_title(r"Polar $C_\ell \times C_d$")

    # Paineis 2 e 3 -- Cl x alpha e Cl/Cd x alpha
    def _curve(ax, data, color, label, ls="-", marker="o", eff=False):
        a, cl, cd = data[:, 0], data[:, 1], data[:, 2]
        if eff:
            valid = cd > 1e-6
            ax.plot(a[valid], cl[valid] / cd[valid], color=color, linewidth=1.6,
                    marker=marker, markersize=3.5, label=label, linestyle=ls)
        else:
            ax.plot(a, cl, color=color, linewidth=1.6, marker=marker,
                    markersize=3.5, label=label, linestyle=ls)

    if baseline_polar:
        _curve(axl, np.array(baseline_polar), BASELINE_COLOR,
               "NACA 0012 (baseline)", ls="--", marker="s")
        _curve(axe, np.array(baseline_polar), BASELINE_COLOR,
               "NACA 0012 (baseline)", ls="--", marker="s", eff=True)
    for alg, polar in polars.items():
        if not polar:
            continue
        color = ALG_COLORS.get(alg, "gray")
        _curve(axl, np.array(polar), color, f"Otimizado ({alg})")
        _curve(axe, np.array(polar), color, f"Otimizado ({alg})", eff=True)

    axl.set_xlabel(r"$\alpha$ [deg]")
    axl.set_ylabel(r"$C_\ell$")
    axl.set_title(r"Curva $C_\ell \times \alpha$")
    axe.set_xlabel(r"$\alpha$ [deg]")
    axe.set_ylabel(r"$C_\ell / C_d$")
    axe.set_title(r"Eficiência aerodinâmica $C_\ell/C_d \times \alpha$")

    # Legenda unica embaixo (as series sao as mesmas nos tres paineis)
    handles, labels = axp.get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Análise dos perfis otimizados", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_population_boxplot(
    eval_logs: Dict[str, pd.DataFrame],
    save_path: Path,
    max_gen_to_plot: int | None = None,
) -> None:
    """Distribuição de Cl/Cd por geração (um painel por algoritmo)."""
    n_alg = len(eval_logs)
    fig, axes = plt.subplots(n_alg, 1, figsize=(11, 3.0 * n_alg), sharex=False)
    if n_alg == 1:
        axes = [axes]

    for ax, (alg, df) in zip(axes, eval_logs.items()):
        if df is None or len(df) == 0:
            continue
        color = ALG_COLORS.get(alg, "gray")

        df_ok = df[(df["Status"] == "ok") & (df["Fitness"] > 0)]
        if len(df_ok) == 0:
            ax.text(0.5, 0.5, f"{alg}: sem avaliações válidas",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        gens = sorted(df_ok["Generation"].unique())
        if max_gen_to_plot:
            gens = gens[:max_gen_to_plot]
        data = [df_ok[df_ok["Generation"] == g]["Fitness"].values for g in gens]

        bp = ax.boxplot(
            data, positions=gens, widths=0.6,
            patch_artist=True, showfliers=True,
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)

        ax.set_ylabel(r"$C_\ell / C_d$")
        ax.set_title(f"Distribuição da população por geração — {alg}")
        ax.set_xlim(min(gens) - 0.5, max(gens) + 0.5)

    axes[-1].set_xlabel("Geração")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_success_rate(
    eval_logs: Dict[str, pd.DataFrame],
    save_path: Path,
) -> None:
    """% de avaliações válidas por geração — robustez do algoritmo."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    for alg, df in eval_logs.items():
        if df is None or len(df) == 0:
            continue
        color = ALG_COLORS.get(alg, "gray")

        gens = sorted(df["Generation"].unique())
        rates = []
        for g in gens:
            sub = df[df["Generation"] == g]
            if len(sub) == 0:
                rates.append(0.0)
            else:
                ok = (sub["Status"] == "ok").sum()
                rates.append(100.0 * ok / len(sub))

        ax.plot(gens, rates, marker="o", linewidth=1.6,
                color=color, label=alg, markersize=4)

    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Geração")
    ax.set_ylabel("Taxa de avaliações bem-sucedidas [%]")
    ax.set_ylim(-2, 102)
    ax.set_title("Robustez do algoritmo: % de avaliações válidas por geração")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_population_diversity(
    eval_logs: Dict[str, pd.DataFrame],
    save_path: Path,
) -> None:
    """Diversidade = desvio-padrão médio dos pesos CST por geração.
    σ alto e estável = exploração; σ caindo cedo = convergência prematura.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    for alg, df in eval_logs.items():
        if df is None or len(df) == 0:
            continue
        color = ALG_COLORS.get(alg, "gray")

        gens = sorted(df["Generation"].unique())
        diversities = []
        for g in gens:
            sub = df[df["Generation"] == g]
            if len(sub) == 0:
                diversities.append(0.0)
                continue
            # Pesos CST estão como string "w1;w2;w3;..."
            weights_matrix = np.array(
                [list(map(float, w.split(";")))
                 for w in sub["CST_Weights"].values]
            )
            diversities.append(float(np.mean(np.std(weights_matrix, axis=0))))

        ax.plot(gens, diversities, marker="o", linewidth=1.6,
                color=color, label=alg, markersize=4)

    ax.set_xlabel("Geração")
    ax.set_ylabel(r"Diversidade média $\bar{\sigma}$ dos pesos CST")
    ax.set_title("Exploração vs. explotação ao longo das gerações")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(
    pareto_fronts: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Fronteira de Pareto Cl × Cd (multi-objetivo)."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for alg, pf in pareto_fronts.items():
        if pf is None or len(pf) == 0:
            continue
        idx = np.argsort(pf[:, 1])
        ax.plot(
            pf[idx, 1], pf[idx, 0],
            "o-", markersize=4,
            color=ALG_COLORS.get(alg, "black"),
            label=alg, linewidth=1.4,
        )

    ax.set_xlabel(r"$C_d$  (arrasto)")
    ax.set_ylabel(r"$C_\ell$  (sustentação)")
    ax.set_title(r"Fronteira de Pareto: $C_\ell \times C_d$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_seeds_boxplot(
    all_results: Dict[str, list],
    save_path: Path,
    baseline_value: float | None = None,
) -> None:
    """Boxplot do Cl/Cd final por algoritmo (uma seed por ponto)."""
    algs = list(all_results.keys())
    data = [[r.best_f for r in all_results[a]] for a in algs]
    n_seeds = len(data[0]) if data else 0

    fig, ax = plt.subplots(figsize=(9, 6))

    positions = np.arange(len(algs))

    bp = ax.boxplot(
        data, positions=positions, widths=0.50,
        patch_artist=True, showmeans=True, meanline=True,
        medianprops=dict(color="black", linewidth=2.0),
        meanprops=dict(color="#B22222", linewidth=1.4, linestyle="--"),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker="D", markersize=6,
                        markerfacecolor="white", markeredgecolor="black"),
    )
    for patch, alg in zip(bp["boxes"], algs):
        patch.set_facecolor(ALG_COLORS.get(alg, "gray"))
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    # Pontos de cada seed com jitter
    rng = np.random.default_rng(0)
    for i, (alg, d) in enumerate(zip(algs, data)):
        x_jitter = positions[i] + rng.uniform(-0.09, 0.09, size=len(d))
        ax.scatter(x_jitter, d, color="#222222", s=32,
                   zorder=5, alpha=0.85,
                   edgecolors="white", linewidths=0.7)

    all_vals = [v for d in data for v in d]
    if baseline_value is not None:
        all_vals.append(baseline_value)
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_range = y_max - y_min if y_max > y_min else 1.0

    # Anotações abaixo das caixas
    txt_y = y_min - 0.12 * y_range
    for i, (alg, d) in enumerate(zip(algs, data)):
        med = float(np.median(d))
        mean = float(np.mean(d))
        std = float(np.std(d))
        ax.text(positions[i], txt_y,
                f"med = {med:.1f}\n$\\bar{{x}}$ = {mean:.1f}\n$\\sigma$ = {std:.1f}",
                ha="center", va="top", fontsize=10,
                color="#1a1a1a",
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="white", edgecolor="lightgray",
                          alpha=0.95))

    if baseline_value is not None:
        ax.axhline(baseline_value, color="black", linestyle="--",
                   linewidth=1.3, alpha=0.7, zorder=2)
        ax.text(positions[-1] + 0.55, baseline_value,
                f"baseline NACA 0012 = {baseline_value:.1f}",
                va="center", ha="left", fontsize=10, color="black",
                bbox=dict(boxstyle="round,pad=0.30",
                          facecolor="white", edgecolor="lightgray", alpha=0.95))

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", lw=2.0, label="mediana"),
        Line2D([0], [0], color="#B22222", lw=1.4, linestyle="--", label="média"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#222222", markeredgecolor="white",
               markersize=8, label=f"{n_seeds} seeds individuais", linestyle=""),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor="white", markeredgecolor="black",
               markersize=8, label="outlier", linestyle=""),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=10, framealpha=0.95, edgecolor="lightgray")

    ax.set_xticks(positions)
    ax.set_xticklabels(algs, fontsize=12)
    ax.set_ylabel(r"Melhor $C_\ell / C_d$ obtido por execução", fontsize=11)
    ax.set_title(f"Consistência entre seeds — n = {n_seeds} execuções independentes por algoritmo",
                 fontsize=12.5, pad=12)

    ax.set_ylim(txt_y - 0.18 * y_range, y_max + 0.10 * y_range)
    ax.set_xlim(positions[0] - 0.7, positions[-1] + 1.6)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_with_uncertainty(
    all_results: Dict[str, list],
    save_path: Path,
) -> None:
    """Mediana + IQR + min-max do melhor histórico entre seeds.
    Mediana/IQR é mais robusta a outliers que média ± std com poucas seeds.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    n_seeds_total = None
    for alg, results_list in all_results.items():
        if not results_list:
            continue
        color = ALG_COLORS.get(alg, "gray")

        # Padroniza tamanho cortando no menor histórico
        histories = [np.asarray(r.history_best_per_eval, dtype=float)
                     for r in results_list if r.history_best_per_eval]
        if not histories:
            continue
        n_min = min(len(h) for h in histories)
        H = np.stack([h[:n_min] for h in histories])
        n_seeds_total = H.shape[0]

        x = np.arange(1, n_min + 1)
        median = np.median(H, axis=0)
        q25 = np.quantile(H, 0.25, axis=0)
        q75 = np.quantile(H, 0.75, axis=0)
        vmin = H.min(axis=0)
        vmax = H.max(axis=0)

        ax.fill_between(x, vmin, vmax, color=color, alpha=0.10, zorder=1)
        ax.fill_between(x, q25, q75, color=color, alpha=0.32, zorder=2)
        ax.plot(x, median, color=color, linewidth=2.2, label=alg,
                drawstyle="steps-post", zorder=4)

    ax.set_xlabel("Número de avaliações", fontsize=11)
    ax.set_ylabel(r"Melhor $C_\ell / C_d$ histórico", fontsize=11)
    title = (f"Convergência mediana entre seeds — n = {n_seeds_total} corridas independentes\n"
             "(linha = mediana;  faixa escura = IQR 25–75%;  faixa clara = mín–máx)")
    ax.set_title(title, fontsize=12, pad=10)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    alg_handles = [Line2D([0], [0], color=ALG_COLORS.get(a, "gray"),
                          lw=2.2, label=a)
                   for a in all_results.keys()]
    band_handles = [
        Patch(facecolor="gray", alpha=0.32, label="IQR (25%–75%)"),
        Patch(facecolor="gray", alpha=0.10, label="mín–máx"),
    ]
    leg1 = ax.legend(handles=alg_handles, loc="upper left",
                     title="Algoritmo", fontsize=10, title_fontsize=10,
                     framealpha=0.95, edgecolor="lightgray")
    ax.add_artist(leg1)
    ax.legend(handles=band_handles, loc="lower right",
              title="Dispersão", fontsize=10, title_fontsize=10,
              framealpha=0.95, edgecolor="lightgray")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_statistical_comparison(
    all_results: Dict[str, list],
    save_path: Path,
    alpha_level: float = 0.05,
) -> None:
    """Heatmap de p-values do Mann-Whitney U pareado entre algoritmos."""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        print("[plot_statistical_comparison] scipy não disponível — pulando.")
        return

    algs = list(all_results.keys())
    n = len(algs)

    pvals = np.full((n, n), np.nan)
    medians = {alg: float(np.median([r.best_f for r in all_results[alg]]))
               for alg in algs}

    for i, a1 in enumerate(algs):
        for j, a2 in enumerate(algs):
            if i == j:
                continue
            f1 = [r.best_f for r in all_results[a1]]
            f2 = [r.best_f for r in all_results[a2]]
            try:
                _, p = mannwhitneyu(f1, f2, alternative="two-sided")
            except Exception:
                p = np.nan
            pvals[i, j] = p

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Verde abaixo de alpha, vermelho acima; alpha = ponto neutro
    from matplotlib.colors import TwoSlopeNorm
    cmap = plt.get_cmap("RdYlGn_r")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=alpha_level, vmax=min(2 * alpha_level, 1.0))

    pvals_display = pvals.copy()
    im = ax.imshow(pvals_display, cmap=cmap, norm=norm, aspect="equal")

    # Diagonal pintada de cinza
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    facecolor="#e8e8e8", edgecolor="white",
                                    linewidth=2, zorder=2))
        ax.text(i, i, "—", ha="center", va="center",
                fontsize=22, color="#999999", zorder=3)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = pvals[i, j]
            if np.isnan(p):
                txt = "n/d"
                color = "#444444"
                weight = "normal"
            elif p < alpha_level:
                txt = f"p = {p:.3f} *"
                color = "white"
                weight = "bold"
            else:
                txt = f"p = {p:.3f}"
                color = "#333333"
                weight = "normal"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=12, color=color, weight=weight, zorder=3)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [f"{a}\n(mediana = {medians[a]:.1f})" for a in algs]
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.tick_params(axis="both", which="both", length=0)

    for i in range(n + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2.5, zorder=4)
        ax.axvline(i - 0.5, color="white", linewidth=2.5, zorder=4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.10)
    cbar.set_label("p-value (Mann-Whitney U)", fontsize=10)
    cbar.ax.axhline(alpha_level, color="black", linewidth=1.2, linestyle="--")
    # Anotacao a esquerda da barra para nao colidir com o rotulo vertical p-value
    cbar.ax.text(-0.45, alpha_level, f"α = {alpha_level}",
                 va="center", ha="right", fontsize=9, color="black",
                 transform=cbar.ax.get_yaxis_transform(),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor="none", alpha=0.85))

    ax.set_title(f"Teste Mann–Whitney U: comparação par-a-par entre algoritmos\n"
                 f"H₀: as distribuições de $C_\\ell/C_d$ final são equivalentes  "
                 f"(* indica p < α = {alpha_level})",
                 fontsize=12, pad=14)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_geometries_per_seed(
    geometries_per_alg: Dict[str, list],
    baseline: Tuple[np.ndarray, np.ndarray],
    save_path: Path,
    baseline_label: str = "NACA 0012 (baseline)",
) -> None:
    """Overlay de todas as geometrias ótimas por algoritmo (uma por seed)."""
    n_alg = len(geometries_per_alg)
    fig, axes = plt.subplots(n_alg, 1, figsize=(11, 3.0 * n_alg), sharex=True)
    if n_alg == 1:
        axes = [axes]

    xb, yb = baseline

    for ax, (alg, items) in zip(axes, geometries_per_alg.items()):
        color = ALG_COLORS.get(alg, "gray")
        ax.plot(xb, yb, "k--", linewidth=1.3, alpha=0.7, label=baseline_label)

        # Melhor sólido em cima, demais em transparência
        items_sorted = sorted(items, key=lambda r: -r[3])
        best_x, best_y, best_seed, best_f = items_sorted[0]
        for x_, y_, seed_, f_ in items_sorted[1:]:
            ax.plot(x_, y_, color=color, linewidth=1.0, alpha=0.45)
        ax.plot(best_x, best_y, color=color, linewidth=2.0, alpha=1.0,
                label=f"melhor (seed {best_seed}, $C_\\ell/C_d$={best_f:.1f})")

        ax.set_ylabel(r"$y/c$")
        n_seeds = len(items)
        ax.set_title(f"{alg} — {n_seeds} geometrias ótimas (uma por seed)",
                     fontsize=11)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    axes[-1].set_xlabel(r"$x/c$")
    fig.suptitle("Consistência das geometrias entre seeds", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_polars_per_seed(
    polars_per_alg: Dict[str, list],
    baseline_polar: list,
    save_path: Path,
) -> None:
    """Polar Cl × Cd de cada seed agrupado por algoritmo."""
    n_alg = len(polars_per_alg)
    fig, axes = plt.subplots(1, n_alg, figsize=(5.5 * n_alg, 5.2),
                              sharey=True)
    if n_alg == 1:
        axes = [axes]

    bp = np.array(baseline_polar) if baseline_polar else None

    for ax, (alg, items) in zip(axes, polars_per_alg.items()):
        color = ALG_COLORS.get(alg, "gray")

        if bp is not None and len(bp) > 0:
            ax.plot(bp[:, 2], bp[:, 1], "k--", linewidth=1.2,
                    alpha=0.6, label="NACA 0012", marker="s", markersize=2)

        items_sorted = sorted(items, key=lambda r: -r[2])
        best_seed = items_sorted[0][1]
        for polar, seed, fit in items_sorted:
            if not polar:
                continue
            p = np.array(polar)
            is_best = (seed == best_seed)
            ax.plot(p[:, 2], p[:, 1],
                    color=color,
                    linewidth=2.0 if is_best else 1.0,
                    alpha=1.0 if is_best else 0.40,
                    marker="o" if is_best else None,
                    markersize=3 if is_best else 0,
                    label=f"seed {seed} (Cl/Cd={fit:.1f})" if is_best
                          else None,
                    zorder=5 if is_best else 2)

        ax.set_xlabel(r"$C_d$")
        if ax is axes[0]:
            ax.set_ylabel(r"$C_\ell$")
        n_seeds = len(items)
        ax.set_title(f"{alg} — {n_seeds} seeds")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle(r"Polar $C_\ell \times C_d$ por seed — consistência aerodinâmica",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_per_seed(
    all_results: Dict[str, list],
    save_path: Path,
) -> None:
    """Curva de melhor histórico de cada seed individual, agrupada por algoritmo."""
    n_alg = len(all_results)
    fig, axes = plt.subplots(n_alg, 1, figsize=(11.5, 3.3 * n_alg), sharex=True)
    if n_alg == 1:
        axes = [axes]

    # Cor distinta por seed, consistente entre paineis (mesma seed = mesma cor)
    all_seeds = sorted({r.seed for results in all_results.values()
                        for r in results})
    n_seeds_total = len(all_seeds)
    if n_seeds_total <= 10:
        palette = [plt.get_cmap("tab10")(k) for k in range(n_seeds_total)]
    elif n_seeds_total <= 20:
        palette = [plt.get_cmap("tab20")(k) for k in range(n_seeds_total)]
    else:
        palette = [plt.get_cmap("turbo")(v)
                   for v in np.linspace(0, 1, n_seeds_total)]
    seed_color = dict(zip(all_seeds, palette))

    for ax, (alg, results) in zip(axes, all_results.items()):
        for r in results:
            if not r.history_best_per_eval:
                continue
            hist = np.asarray(r.history_best_per_eval)
            x_eval = np.arange(1, len(hist) + 1)
            ax.plot(x_eval, hist, color=seed_color[r.seed],
                    linewidth=1.5, alpha=0.9,
                    label=f"seed {r.seed} (final={r.best_f:.1f})",
                    drawstyle="steps-post")

        ax.set_ylabel(r"Melhor $C_\ell / C_d$")
        ax.set_title(f"{alg} — convergência individual por seed",
                     fontsize=11)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=8.5, framealpha=0.95, edgecolor="lightgray")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("Número de avaliações")
    fig.suptitle("Variabilidade intra-algoritmo da convergência",
                 fontsize=13, y=1.00)
    fig.tight_layout(rect=[0, 0, 0.80, 0.97])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_effect_size_analysis(
    all_results: Dict[str, list],
    save_path: Path,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
) -> None:
    """Cliff's delta + IC bootstrap da diferença de medianas (par-a-par).

    Interpretação de |delta|:
      < 0.147   trivial
      < 0.330   pequeno
      < 0.474   médio
      ≥ 0.474   grande
    """
    algs = list(all_results.keys())
    n = len(algs)

    rng = np.random.default_rng(42)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 7),
                              squeeze=False)
    axes = axes[0]

    for ax, (i, j) in zip(axes, pairs):
        a1, a2 = algs[i], algs[j]
        x1 = np.array([r.best_f for r in all_results[a1]])
        x2 = np.array([r.best_f for r in all_results[a2]])

        # Cliff's delta: P(x1>x2) - P(x1<x2)
        n1, n2 = len(x1), len(x2)
        diffs = x1[:, None] - x2[None, :]
        delta = float((np.sum(diffs > 0) - np.sum(diffs < 0)) / (n1 * n2))

        # IC bootstrap da diferença de medianas
        boot_diffs = []
        for _ in range(n_bootstrap):
            s1 = rng.choice(x1, size=n1, replace=True)
            s2 = rng.choice(x2, size=n2, replace=True)
            boot_diffs.append(np.median(s1) - np.median(s2))
        boot_diffs = np.array(boot_diffs)
        lo = np.quantile(boot_diffs, (1 - confidence) / 2)
        hi = np.quantile(boot_diffs, 1 - (1 - confidence) / 2)
        med_diff = float(np.median(x1) - np.median(x2))

        ax.hist(boot_diffs, bins=50, color="lightgray",
                edgecolor="gray", alpha=0.7)
        ax.axvline(med_diff, color="black", linewidth=1.8, linestyle="-",
                   label=f"diferença observada\n= {med_diff:+.2f}")
        ax.axvline(lo, color="#B22222", linewidth=1.4, linestyle="--",
                   label=f"IC {int(confidence*100)}% bootstrap\n[{lo:+.2f}, {hi:+.2f}]")
        ax.axvline(hi, color="#B22222", linewidth=1.4, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)

        abs_d = abs(delta)
        if abs_d < 0.147:
            mag = "trivial"
        elif abs_d < 0.330:
            mag = "pequeno"
        elif abs_d < 0.474:
            mag = "médio"
        else:
            mag = "grande"

        ic_excludes_zero = (lo > 0) or (hi < 0)
        signif_label = "significativo" if ic_excludes_zero else "n.s."

        ax.set_title(f"{a1}  vs  {a2}\n"
                     f"Cliff's δ = {delta:+.3f} ({mag}) — {signif_label}",
                     fontsize=11)
        ax.set_xlabel(r"Diferença de mediana ($C_\ell/C_d$)")
        if ax is axes[0]:
            ax.set_ylabel("Frequência (bootstrap)")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle(f"Tamanho de efeito (Cliff's δ) e IC {int(confidence*100)}% "
                 f"bootstrap das diferenças  —  n_boot = {n_bootstrap}",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
