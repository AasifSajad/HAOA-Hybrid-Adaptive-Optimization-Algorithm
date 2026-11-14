# HAOA — Hybrid Adaptive Optimization Algorithm

HAOA is a research-oriented, bounded, population-based minimization algorithm. This repository includes the HAOA implementation, standard numerical benchmark functions, a Particle Swarm Optimization (PSO) baseline, experiment runners, and CSV/plot exporters.

## Features

- Hybrid exploration and elite-directed exploitation
- Adaptive velocity, mutation, and stagnation-triggered Lévy exploration
- Scalar and per-dimension bounds
- Reproducible HAOA and PSO runs through isolated random generators
- Sphere, Rastrigin, Rosenbrock, Ackley, and Griewank benchmarks
- Paired multi-seed HAOA/PSO comparison with raw results, aggregate statistics, and convergence plots

## Repository layout

```text
HAOA/
├── algorithm/haoa.py
├── benchmark_functions/
├── comparison_algorithms/pso.py
├── experiments/
│   ├── benchmark_manager.py
│   ├── csv_exporter.py
│   ├── dataset_comparison.py
│   └── full_runner.py
├── results/
└── visualization/
```

## Requirements

Python 3.9 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the complete HAOA benchmark suite

Run from the repository root:

```bash
python -m HAOA.experiments.full_runner
```

The runner evaluates five benchmark functions and writes summary/convergence CSV files and plots under HAOA/results.

## Run a paired HAOA-versus-PSO study

```bash
python -m HAOA.experiments.dataset_comparison
```

By default, the comparison runs 30 paired seeds (0–29) on all five benchmark functions, using the same objective, dimension, bounds, population size, and iteration budget for both algorithms. For each seed, both optimizers receive the same seed value; their random streams remain independent.

Useful options:

```bash
python -m HAOA.experiments.dataset_comparison --runs 50 --dimension 30 --population-size 40 --iterations 200
```

The comparison produces:

- `HAOA/results/benchmark_comparison_runs.csv`: every benchmark/seed result and paired score difference
- `HAOA/results/benchmark_comparison_summary.csv`: means, standard deviations, medians, wins/ties, paired median difference, and a 95% percentile-bootstrap interval
- `HAOA/results/benchmark_comparison.csv`: compatibility summary with the established score columns plus aggregate fields
- One mean best-so-far convergence plot per benchmark

A negative paired difference (HAOA minus PSO) favours HAOA for that seed. The bootstrap interval describes uncertainty in the paired median difference; it is not a substitute for a pre-registered statistical protocol or a formal omnibus test. Do not infer general superiority from one benchmark, one seed, or execution time alone. For publication-quality claims, report the complete experimental configuration and consider additional non-parametric tests and effect sizes.

## Use HAOA directly

```python
import numpy as np
from HAOA.algorithm.haoa import HAOA

def sphere(x):
    return np.sum(x ** 2)

optimizer = HAOA(
    objective_function=sphere,
    dimension=10,
    lower_bound=-5.12,
    upper_bound=5.12,
    population_size=40,
    max_iterations=200,
    random_state=42,
)
result = optimizer.optimize()
print(result["best_score"])
print(result["best_solution"])
```

The objective function must accept a one-dimensional NumPy array and return a finite scalar. Both optimizers minimize the objective and enforce the supplied bounds.

## Tests

```bash
python -m pytest
```

## Research note

Benchmark outcomes depend on seed, search budget, dimensionality, bounds, and implementation details. This repository reports repeated-run distributions and paired comparisons; it does not assume HAOA is universally better than PSO.
