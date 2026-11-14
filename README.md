# HAOA — Hybrid Adaptive Optimization Algorithm Framework

**HAOA (Hybrid Adaptive Optimization Algorithm)** is a research-oriented hybrid metaheuristic optimization framework developed for solving difficult high-dimensional, nonlinear, and multimodal continuous optimization problems.

The framework combines adaptive exploration, elite-directed exploitation, stochastic mutation, and stagnation-triggered Lévy exploration.

---

## 🌟 Novel Contribution: ML-HAOA

Our latest innovation is **Multi-Leader HAOA (ML-HAOA)**. Unlike basic HAOA which converges toward a single global elite, **ML-HAOA** maintains a dynamic Voronoi-niched Pareto archive of $K$ diverse leaders. This prevents premature swarm collapse on rugged, multimodal landscapes (such as the Ackley function) by promoting diversified exploratory flow.

---

## 🔬 Scientific Methodology

### Velocity Update Rule (ML-HAOA)

$$v_i^{t+1} = w(t) v_i^t + c_1 r_1 \odot (l_i^* - x_i^t) + c_2 r_2 \odot (x_{gbest} - x_i^t) + c_3 r_3 \odot (x_{r1}^t - x_{r2}^t)$$

- **Dynamic Niching:** Individuals are assigned to the nearest Voronoi leader ($l_i^*$).
- **Differential Exploration:** The term $(x_{r1}^t - x_{r2}^t)$ introduces cross-niche differential interaction, preventing stagnation.
- **Predictive Re-seeding:** Stagnant individuals (measured by $S_i \ge K_{stag}$) perform an Oppositional-Lévy jump to re-enter the search space with global diversity.

---

## 📊 Comprehensive Benchmark Results (900 Runs, D=30, NFE=15,030)

| Benchmark | ML-HAOA (Ours) | HAOA_adv | HAOA_orig | PSO | Winner |
|---|---|---|---|---|---|
| **Sphere** | 0.8791 | **0.1748** | 18.311 | 1333.3 | HAOA_adv |
| **Rastrigin** | **57.028** | 80.429 | 150.54 | 122.30 | **ML-HAOA** |
| **Rosenbrock** | 283.91 | **37.354** | 298.32 | 9210.4 | HAOA_adv |
| **Ackley** | 4.1438 | **0.6655** | 2.5332 | 4.5241 | HAOA_adv |
| **Griewank** | 0.4474 | **0.3529** | 1.1630 | 12.113 | HAOA_adv |

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/AasifSajad/HAOA-Hybrid-Adaptive-Optimization-Algorithm.git
cd HAOA-Hybrid-Adaptive-Optimization-Algorithm
pip install -r requirements.txt
```

### Running the Full Benchmark Pipeline (ML-HAOA + Baselines)

```bash
python run_ml_experiment.py
```

### Generating Statistical Analysis

```bash
python analyze_ml_results.py
```

### Testing Reproducibility

```bash
pytest tests/test_haoa.py
```

---

## 📁 Project Structure

```text
HAOA/
├── algorithm/
│   ├── haoa.py             # Baseline Original HAOA
│   ├── haoa_improved.py    # Improved baseline
│   ├── haoa_advanced.py    # Advanced single-elite
│   └── haoa_ml.py          # Novel Multi-Leader HAOA
├── benchmark_functions/
├── comparison_algorithms/
│   ├── pso.py              # PSO Baseline
│   └── de.py               # DE Baseline
├── tests/
│   └── test_haoa.py
└── run_ml_experiment.py    # Full 900-run pipeline
```

---

## 📝 Research & License

This project evaluates optimization performance across 5 diverse landscapes (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank) using a strict $NFE=15,030$ budget, 30 independent seeds, Friedman rank analysis, and Wilcoxon signed-rank tests for statistical validity.

MIT License.

**Author:** Aasif Sajad  
**GitHub:** [https://github.com/AasifSajad](https://github.com/AasifSajad)
