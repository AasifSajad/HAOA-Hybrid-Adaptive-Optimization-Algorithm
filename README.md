# HAOA — Hybrid Adaptive Optimization Algorithm Framework

**HAOA (Hybrid Adaptive Optimization Algorithm)** is a state-of-the-art research framework for solving complex high-dimensional continuous optimization problems on non-convex, multimodal, and deceptive benchmark landscapes.

The framework includes our novel contribution: **Multi-Leader HAOA (ML-HAOA)**, which introduces a dynamic Voronoi Multi-Leader Archive and Predictive Spatial Dispersion Trajectory Re-seeding to prevent premature convergence on rugged landscapes.

---

## 🌟 Key Features

- **Novel ML-HAOA Architecture:** Dynamic Voronoi Multi-Leader Pareto archive preventing single-attractor swarm collapse.
- **Strict NFE Budget Verification:** Evaluated under a fixed budget ($NFE = 15,030$, $D=30$) with non-evaluative state handling.
- **Comprehensive Benchmark Suite:** 5 standard benchmark functions (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank).
- **Rigorous Comparative Baselines:** Benchmarked against Original HAOA, Improved HAOA, Advanced HAOA, Particle Swarm Optimization (PSO), and Differential Evolution (DE).
- **Statistical Rigor:** 30 independent random seeds per benchmark (900 total evaluations) with Friedman rank analysis and Wilcoxon signed-rank tests.

---

## 🔬 Mathematical Model: ML-HAOA

Unlike standard PSO or basic HAOA which collapse toward a single global elite ($x_{gbest}$), **ML-HAOA** maintains a Pareto-optimal set of $K$ diverse local leaders $\mathcal{L} = \{l_1, l_2, \dots, l_K\}$ selected via spatial dispersion:

$$\text{Niche Assignment: } l_i^* = \arg\min_{l \in \mathcal{L}} \|x_i - l\|_2$$

### Velocity Update Rule:

$$v_i^{t+1} = w(t) v_i^t + c_1 r_1 \odot (l_i^* - x_i^t) + c_2 r_2 \odot (x_{gbest} - x_i^t) + c_3 r_3 \odot (x_{r1}^t - x_{r2}^t)$$

where:
- $w(t) = 0.9 - 0.5 \frac{t}{T_{max}}$ is the dynamic inertia weight.
- $l_i^*$ is the assigned Voronoi leader for particle $i$.
- $x_{r1}^t - x_{r2}^t$ is a differential perturbation vector promoting cross-niche exploratory flow.

### Predictive Trajectory Re-seeding:

When stagnation counter $S_i \ge K_{stag}$, the particle performs an Oppositional-Lévy jump:

$$x_i^{new} = \begin{cases} LB + UB - x_i^t & \text{with probability } 0.4 \\ l_i^* + \gamma \cdot (1 - \frac{t}{T_{max}}) \cdot L(\beta) & \text{otherwise} \end{cases}$$

---

## 📊 Comprehensive Benchmark Results (900 Runs, D=30, NFE=15,030)

Mean fitness values over 30 independent seeds per benchmark (Lower is better):

| Benchmark | Function Type | ML-HAOA (Ours) | HAOA_adv | HAOA_impr | HAOA_orig | PSO | DE | Winner |
|---|---|---|---|---|---|---|---|---|
| **Sphere** | Unimodal | 0.8791 | **0.1748** | 0.4840 | 18.311 | 1333.33 | 750.46 | HAOA_adv |
| **Rastrigin** | Multimodal | **57.028** | 80.429 | 59.888 | 150.54 | 122.30 | 250.43 | **ML-HAOA** |
| **Rosenbrock** | Deceptive | 283.91 | **37.354** | 211.25 | 298.32 | 9210.40 | 219383 | HAOA_adv |
| **Ackley** | Multimodal | 4.1438 | **0.6655** | 9.9633 | 2.5332 | 4.5241 | 8.0315 | HAOA_adv |
| **Griewank** | Multimodal | 0.4474 | **0.3529** | 0.6085 | 1.1630 | 12.113 | 7.7542 | HAOA_adv |
| **Overall Rank** | — | **2.80** | **2.18** | 3.23 | 4.17 | 2.83 | 5.78 | **HAOA Variants** |

### 🏆 Key Findings:
1. **ML-HAOA wins Rank 1 on Rastrigin** (57.03 vs 80.43 for HAOA_adv, 122.30 for PSO, 250.43 for DE).
2. **ML-HAOA outperforms Original HAOA on 4 out of 5 benchmarks** (Sphere, Rastrigin, Rosenbrock, Griewank).
3. **ML-HAOA achieves Overall Rank 2.80**, outperforming PSO (2.83), Improved HAOA (3.23), Original HAOA (4.17), and DE (5.78).

---

## 📁 Repository Structure

```text
HAOA-Hybrid-Adaptive-Optimization-Algorithm/
│
├── HAOA/
│   ├── algorithm/
│   │   ├── haoa.py             # Original HAOA baseline
│   │   ├── haoa_improved.py    # Improved HAOA baseline
│   │   ├── haoa_advanced.py    # Advanced single-elite HAOA
│   │   └── haoa_ml.py          # Novel Multi-Leader HAOA (ML-HAOA)
│   │
│   ├── benchmark_functions/
│   │   ├── sphere.py
│   │   ├── rastrigin.py
│   │   ├── rosenbrock.py
│   │   ├── ackley.py
│   │   └── griewank.py
│   │
│   └── comparison_algorithms/
│       ├── pso.py              # Particle Swarm Optimization
│       └── de.py               # Differential Evolution
│
├── tests/
│   └── test_haoa.py            # Pytest suite for NFE compliance & unit tests
│
├── results/
│   ├── ml_experiment/          # 900-run experiment output data
│   │   ├── raw_results.csv
│   │   └── analysis/
│   │       ├── statistical_summary.csv
│   │       ├── friedman_ranks.csv
│   │       └── wilcoxon_tests.csv
│
├── run_ml_experiment.py       # Full 900-run evaluation pipeline
├── analyze_ml_results.py       # Statistical analysis script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/AasifSajad/HAOA-Hybrid-Adaptive-Optimization-Algorithm.git
cd HAOA-Hybrid-Adaptive-Optimization-Algorithm
pip install -r requirements.txt
```

### Running Unit Tests

```bash
pytest tests/test_haoa.py
```

### Executing Full 900-Run Benchmark Pipeline

```bash
python run_ml_experiment.py
```

### Generating Statistical Analysis

```bash
python analyze_ml_results.py
```

---

## 📝 License & Citation

This project is open-source under the MIT License.

**Author:** Aasif Sajad  
**GitHub:** [https://github.com/AasifSajad](https://github.com/AasifSajad)
