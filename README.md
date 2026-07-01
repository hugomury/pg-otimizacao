# Otimização de aerofólios 2D — PG

Framework em Python pra otimizar aerofólios em 2D usando CST (Class-Shape
Transformation) + XFoil, com DE, GA e CMA-ES rodando via [pymoo](https://pymoo.org).
NSGA-II tá disponível também pro modo multi-objetivo.

## Estrutura

```
airfoil_optimizer/
├── config.py            # parâmetros globais
├── cst.py               # parametrização CST + fit + validação geométrica
├── xfoil_runner.py      # interface com o XFoil via subprocess
├── data_logger.py       # IDs sequenciais, .dat, polares, log CSV
├── problem.py           # AirfoilProblem (pymoo ElementwiseProblem)
├── optimizer.py         # wrapper DE/GA/CMA-ES/NSGA-II
├── plotter.py           # gráficos
├── main.py              # orquestra tudo
├── requirements.txt
├── airfoils/            # NACA 0012 é gerado se não existir
└── results/             # criado em runtime
```

## Instalação

```bash
python -m venv .venv
. .venv/Scripts/activate     # Windows
# ou: source .venv/bin/activate   (Linux/Mac)

pip install -r requirements.txt
```

Antes de rodar, ajusta o caminho do XFoil em `config.py`:

```python
DEFAULT_XFOIL_PATH = r"C:\caminho\para\xfoil.exe"
```

Dá pra sobrescrever via variável de ambiente (`DEFAULT_XFOIL_PATH`) também,
útil quando você tá testando em máquinas diferentes.

## Rodando

```bash
python main.py
```

Na primeira execução o framework gera o `airfoils/naca0012.dat` automaticamente,
roda o XFoil pro baseline e parte pra otimização. Os resultados ficam em
`results/`: geometrias `.dat`, polares `.csv`, log de avaliações e os PNGs em
`results/plots/`.

## Configuração

Praticamente tudo é controlado em `config.py`. Os ajustes que mais costumo mexer:

- **`REYNOLDS`, `MACH`, `ALPHA_FIXED`** — condições de projeto
- **`N_CST_UPPER` / `N_CST_LOWER`** — quantos pesos CST por superfície (10
  cobre bem o espaço útil sem ficar ondulado)
- **`POP_SIZE`, `N_MAX_EVAL`** — controla o custo computacional
- **`ALGORITHMS_TO_RUN`** — quais algoritmos comparar
- **`OPT_MODE = "multi"`** — liga modo multi-objetivo (NSGA-II)
- **`EVAL_MODE = "alpha_sweep"`** — pega o pico de Cl/Cd numa varredura
  em vez de avaliar num ângulo fixo
- **`N_SEEDS`** — usa 1 pra teste rápido (~7 min), 5 pra análise estatística
  (~35 min). Com 1 seed só os gráficos estatísticos são pulados.

## Gráficos gerados

Saem todos numerados em `results/plots/`:

- `01_convergence.png` — Cl/Cd ao longo das avaliações, melhor histórico
- `02_comparative_runtime.png` — tempo, nº avals e Cl/Cd final lado a lado
- `03_geometry.png` — perfis ótimos vs baseline
- `04_polar.png` — polar Cl × Cd completa
- `05_efficiency.png` — Cl × α e Cl/Cd × α (off-design)
- `06_population_boxplot.png` — distribuição da pop por geração
- `07_success_rate.png` — % de avaliações válidas por geração
- `08_diversity.png` — diversidade dos pesos CST (exploração vs explotação)

Se rodar com `N_SEEDS > 1`:

- `09_seeds_boxplot.png` — robustez entre seeds
- `10_convergence_uncertainty.png` — mediana + IQR + min-max
- `11_statistical_comparison.png` — Mann-Whitney p-values
- `12_geometries_per_seed.png` — todas as geometrias ótimas por seed
- `13_polars_per_seed.png` — polares por seed
- `14_convergence_per_seed.png` — uma curva por seed
- `15_effect_size.png` — Cliff's delta + IC bootstrap

E se `OPT_MODE = "multi"`, sai também o `pareto.png`.

## Quando dá problema

**`xfoil.exe not found`** — confere o `DEFAULT_XFOIL_PATH` em `config.py`. Se
ele tá certo mas continua falhando, roda o `xfoil.diagnose()` (já é chamado
automático quando o baseline falha).

**Tudo penalizado / 100% de falhas na geração 1** — geralmente o XFoil não
tá convergindo no Re configurado. Tenta `EVAL_MODE = "alpha_sweep"` se
ainda não tá, ou aumenta o `XFOIL_ITER`. Se persistir, ativa
`XFOIL_DEBUG = True` pra ver o stdout do XFoil.

**Caminho longo no Windows** — o XFoil é Fortran legado e tem buffer de
strings limitado (~64 chars). O framework já contorna isso passando só o
nome do arquivo e usando `cwd`, mas se o caminho do projeto em si for
absurdamente longo, vale colocar em algo mais curto tipo `C:\dev\airfoil`.

**Otimização muito lenta** — reduz `POP_SIZE` ou `N_MAX_EVAL`. Cada avaliação
é uma chamada externa ao XFoil (~50-200 ms cada). E nunca, jamais, deixa o
`XFOIL_VISIBLE = True` pra rodada de produção — fica 5-10x mais lento por
causa do overhead de inicialização da janela do console no Windows.
