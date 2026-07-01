"""Configuração centralizada"""

import os
from pathlib import Path

# Caminho do XFoil 
DEFAULT_XFOIL_PATH: str = os.environ.get(
    "DEFAULT_XFOIL_PATH",
    r"C:\caminho\para\xfoil.exe", # Caminho genérico de exemplo
)

# --- Condições aerodinâmicas ---
REYNOLDS: float = 500_000.0
MACH: float = 0.0
NCRIT: float = 9.0
XTR_TOP: float = 1.0
XTR_BOT: float = 1.0

EVAL_MODE: str = "alpha_sweep"
ALPHA_FIXED: float = 5.0
ALPHA_SWEEP_RANGE: tuple = (-2.0, 12.0, 1.0)

XFOIL_ITER: int = 300
XFOIL_TIMEOUT_S: float = 15.0

XFOIL_DEBUG: bool = False

# Modo visível deixa a rodada ~5-10x mais lenta no Windows por causa do
# overhead de inicialização da console GDI. Só ligar pra demo.
XFOIL_VISIBLE: bool = True
XFOIL_VISIBLE_DELAY: float = 0.0

VERBOSE_EVAL: bool = True

# --- Parametrização CST ---
N_CST_UPPER: int = 10
N_CST_LOWER: int = 10
CST_N1: float = 0.5
CST_N2: float = 1.0
N_AIRFOIL_POINTS: int = 200

# Bounds relativos ao baseline restringe a busca a uma vizinhança
# elimina muita geometria inválida
USE_RELATIVE_BOUNDS: bool = True
CST_BOUNDS_DELTA: float = 0.10

CST_UPPER_BOUNDS: tuple = (-0.20, 0.60)
CST_LOWER_BOUNDS: tuple = (-0.60, 0.20)

# Warm start: parte da pop começa perto do baseline pra garantir que tem
# indivíduo viável na geração 1
WARM_START_FRACTION: float = 0.4
WARM_START_NOISE_STD: float = 0.02

# Restrições geométricas
MIN_THICKNESS: float = 0.07
MAX_THICKNESS: float = 0.20
MIN_TE_THICKNESS: float = 0.0
MAX_TE_THICKNESS: float = 0.005

# --- Otimização ---
POP_SIZE: int = 30
N_GENERATIONS: int = 25
N_MAX_EVAL: int = 750
RANDOM_SEED: int = 42
 
N_SEEDS: int = 10
SEEDS: list = [42, 123, 7, 2024, 999, 31, 555, 8, 1234, 77]

ALGORITHMS_TO_RUN: list = ["DE", "GA", "CMAES"]

# "single" maximiza Cl/Cd; "multi" usa NSGA-II (max Cl + min Cd)
OPT_MODE: str = "single"

DE_VARIANT: str = "DE/rand/1/bin"
DE_F: float = 0.7
DE_CR: float = 0.9

GA_CROSSOVER_PROB: float = 0.9
GA_MUTATION_ETA: float = 20

# CMA-ES
CMAES_SIGMA0: float = 0.10
CMAES_RESTARTS: int = 0
CMAES_BIPOP: bool = False

# --- Sanity checks contra fitness patológico ---
# XFoil às vezes prevê Cd absurdamente baixo por causa de bolha laminar
# idealizada. Sem esses filtros a otimização explora essas falhas
# numéricas e gera perfis tipo lâmina que não funcionam em ensaio real.
MAX_REALISTIC_CL_CD: float = 150.0
MIN_REALISTIC_CD: float = 0.0050
SPIKE_DETECTION_RATIO: float = 1.30
USE_ROBUST_FITNESS: bool = True
ROBUST_FITNESS_TOP_K: int = 5

# Espessura mínima em pontos chave do perfil — evita "lâminas"
LOCAL_THICKNESS_CONSTRAINTS: list = [
    (0.25, 0.05),
    (0.50, 0.04),
    (0.75, 0.015),
]

# Restrição de convexidade do extradorso.
ENFORCE_UPPER_CONVEXITY: bool = True
CONVEXITY_TOLERANCE: float = 0.20
CONVEXITY_CHECK_RANGE: tuple = (0.05, 0.85)

# Limite da variação total de y'' do extradorso.
MAX_CURVATURE_ROUGHNESS: float = 8.0

# --- I/O ---
PROJECT_ROOT: Path = Path(__file__).parent.resolve()
OUTPUT_DIR: Path = PROJECT_ROOT / "results"
GEOMETRY_DIR: Path = OUTPUT_DIR / "geometries"
POLAR_DIR: Path = OUTPUT_DIR / "polars"
PLOT_DIR: Path = OUTPUT_DIR / "plots"
LOG_CSV: Path = OUTPUT_DIR / "evaluation_log.csv"

BASELINE_AIRFOIL_DAT: Path = PROJECT_ROOT / "airfoils" / "naca0012.dat"

# pymoo minimiza, e o objetivo é -Cl/Cd; +1e6 garante que indivíduos
# inválidos não influenciem a evolução
PENALTY_VALUE: float = 1.0e6


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, GEOMETRY_DIR, POLAR_DIR, PLOT_DIR):
        d.mkdir(parents=True, exist_ok=True)
