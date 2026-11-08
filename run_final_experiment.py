import os
import time
import json
import numpy as np
import pandas as pd
import sys

sys.path.append(os.getcwd())

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.algorithm.haoa_advanced import HAOAAdvanced
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution

from HAOA.benchmark_functions.sphere import sphere
from HAOA.benchmark_functions.rastrigin import rastrigin
from HAOA.benchmark_functions.rosenbrock import rosenbrock
from HAOA.benchmark_functions.ackley import ackley
from HAOA.benchmark_functions.griewank import griewank

# --- Configuration ---
BENCHMARKS = {
    "Sphere": {"func": sphere, "lb": -100, "ub": 100},
    "Rastrigin": {"func": rastrigin, "lb": -5.12, "ub": 5.12},
    "Rosenbrock": {"func": rosenbrock, "lb": -30, "ub": 30},
    "Ackley": {"func": ackley, "lb": -32.768, "ub": 32.768},
    "Griewank": {"func": griewank, "lb": -600, "ub": 600}
}
ALGORITHMS = ["HAOA_orig", "HAOA_impr", "HAOA_adv", "PSO", "DE"]
N = 30
D = 30
BUDGET = 15030
SEEDS = list(range(100, 130))  # 30 independent seeds

OUTPUT_DIR = "results/final_experiment"
RAW_RESULTS_FILE = os.path.join(OUTPUT_DIR, "raw_results.csv")
CONVERGENCE_DIR = os.path.join(OUTPUT_DIR, "convergence_histories")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")

# --- Transparent Counter ---
class TransparentCounter:
    def __init__(self, func, budget=15030):
        self.func = func
        self.budget = budget
        self.actual_executions = 0
    def __call__(self, x):
        self.actual_executions += 1
        return self.func(x)

# --- Bounded Wrappers for strict NFE termination ---
class BoundedHAOA(HAOAOriginal):
    def optimize(self):
        for iteration in range(self.max_iterations):
            if self.objective_function.actual_executions >= BUDGET: break
            self.evaluate_population()
            if self.objective_function.actual_executions >= BUDGET: break
            self.update_population(iteration)
            self.convergence_curve.append(self.best_score)
        return {"best_score": self.best_score, "best_solution": self.best_solution, "convergence_curve": self.convergence_curve}

class BoundedHAOAImpr(HAOAImproved):
    def update_population(self, iteration):
        progress, inertia_weight, adaptive_learning, c_e, mutation_strength = self._compute_coefficients(iteration)
        elite, velocity_limit, domain_range = self.best_solution.copy(), 0.2*(self.upper_bound - self.lower_bound), self.upper_bound - self.lower_bound
        levy_scale = self.gamma * domain_range * (1 - progress)

        for i in range(self.population_size):
            if self.objective_function.actual_executions >= BUDGET: break
            r1, r2 = self.rng.random(self.dimension), self.rng.random(self.dimension)
            r1_idx = int(self.rng.integers(0, self.population_size))
            r2_idx = int(self.rng.integers(0, self.population_size))
            while r2_idx == i: r2_idx = int(self.rng.integers(0, self.population_size))
            exploration = adaptive_learning * r2 * (self.population[r1_idx] - self.population[r2_idx])
            exploitation = c_e * r1 * (elite - self.population[i])
            self.velocity[i] = inertia_weight * self.velocity[i] + exploitation + 0.15 * exploration
            self.velocity[i] = np.clip(self.velocity[i], -velocity_limit, velocity_limit)
            new_position = self.population[i] + self.velocity[i] + mutation_strength * self.rng.standard_normal(self.dimension)
            if progress > 0.75: new_position += 0.03 * (elite - new_position)
            if self.stagnation_counters[i] > self.k_stag and i != self.elite_index:
                new_position += levy_scale * self._levy_flight()
            new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
            new_fitness = self.objective_function(new_position)
            if new_fitness < self.fitness_values[i]:
                self.population[i], self.fitness_values[i], self.stagnation_counters[i] = new_position, new_fitness, 0
            else:
                self.velocity[i] = self.alpha * self.velocity[i]
                self.stagnation_counters[i] += 1
        self._update_elite()

    def optimize(self):
        self._update_elite()
        for iteration in range(self.max_iterations):
            if self.objective_function.actual_executions >= BUDGET: break
            self.update_population(iteration)
            self.convergence_curve.append(self.best_score)
        return {"best_score": self.best_score, "best_solution": self.best_solution, "convergence_curve": self.convergence_curve}

class BoundedHAOAAdv(HAOAAdvanced):
    def update_population(self, iteration):
        progress = iteration / self.max_iterations
        inertia_weight, adaptive_learning, c1, c2, mutation_strength = 0.95 - 0.6 * progress, 1.8 * (1 - progress), 1.0 - 0.5 * progress, 0.5 + 1.0 * progress, 0.003 * (1 - progress)
        elite, velocity_limit, domain_range = self.best_solution.copy(), 0.2 * (self.upper_bound - self.lower_bound), self.upper_bound - self.lower_bound

        for i in range(self.population_size):
            if self.objective_function.actual_executions >= BUDGET: break
            r1, r2, r3 = self.rng.random(self.dimension), self.rng.random(self.dimension), self.rng.random(self.dimension)
            local_best = self._get_local_best(i)
            r1_idx, r2_idx = self.rng.choice([idx for idx in range(self.population_size) if idx != i], size=2, replace=False)
            exploration = adaptive_learning * r3 * (self.population[r1_idx] - self.population[r2_idx])
            self.velocity[i] = inertia_weight * self.velocity[i] + c1 * r1 * (local_best - self.population[i]) + c2 * r2 * (elite - self.population[i]) + 0.15 * exploration
            self.velocity[i] = np.clip(self.velocity[i], -velocity_limit, velocity_limit)
            new_position = self.population[i] + self.velocity[i] + mutation_strength * self.rng.standard_normal(self.dimension)
            if self.stagnation_counters[i] > self.k_stag and np.sum(self.velocity[i] ** 2) < 1e-4:
                if self.rng.random() < 0.5: new_position = self.lower_bound + self.upper_bound - self.population[i]
                else: new_position = self.population[i] + 0.01 * domain_range * (1 - progress) * self._levy_flight()
            new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
            new_fitness = self.objective_function(new_position)
            if new_fitness < self.fitness_values[i]:
                self.population[i], self.fitness_values[i], self.stagnation_counters[i] = new_position, new_fitness, 0
            else:
                self.velocity[i] = 0.5 * self.velocity[i]
                self.stagnation_counters[i] += 1
        self._update_elite()
        pairwise_dists = []
        for a in range(self.population_size):
            for b in range(a + 1, self.population_size):
                pairwise_dists.append(np.linalg.norm(self.population[a] - self.population[b]))
        self.diversity_curve.append(np.mean(pairwise_dists) if pairwise_dists else 0.0)

    def optimize(self):
        self._update_elite()
        for iteration in range(self.max_iterations):
            if self.objective_function.actual_executions >= BUDGET: break
            self.update_population(iteration)
            self.convergence_curve.append(self.best_score)
        return {"best_score": self.best_score, "best_solution": self.best_solution, "convergence_curve": self.convergence_curve, "diversity_curve": self.diversity_curve}

class BoundedPSO(PSO):
    def optimize(self):
        w, c1, c2 = 0.7, 1.5, 1.5
        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                if self.objective_function.actual_executions >= BUDGET: break
                fitness = self.objective_function(self.positions[i])
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i], self.personal_best_positions[i] = fitness, self.positions[i].copy()
                if fitness < self.global_best_score:
                    self.global_best_score, self.global_best_position = fitness, self.positions[i].copy()
            if self.objective_function.actual_executions >= BUDGET: break
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dimension), np.random.rand(self.dimension)
                self.velocities[i] = w * self.velocities[i] + c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + c2 * r2 * (self.global_best_position - self.positions[i])
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], self.lower_bound, self.upper_bound)
            self.convergence_curve.append(self.global_best_score)
        return {"best_score": self.global_best_score, "best_solution": self.global_best_position, "convergence_curve": self.convergence_curve}

def get_optimizer(algo_name, config, counter, init_pop, seed):
    if algo_name == "HAOA_orig":
        opt = BoundedHAOA(counter, D, config["lb"], config["ub"], N, 167)
        opt.evaluations_limit = BUDGET
        opt.population = init_pop.copy()
        return opt
    elif algo_name == "HAOA_impr":
        return BoundedHAOAImpr(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)
    elif algo_name == "HAOA_adv":
        return BoundedHAOAAdv(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)
    elif algo_name == "PSO":
        opt = BoundedPSO(counter, D, config["lb"], config["ub"], N, 501)
        opt.evaluations_limit = BUDGET
        opt.positions = init_pop.copy()
        opt.personal_best_positions = init_pop.copy()
        return opt
    elif algo_name == "DE":
        return DifferentialEvolution(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)
    return None

def run_experiment():
    os.makedirs(CONVERGENCE_DIR, exist_ok=True)
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f: progress = json.load(f)

    results = []
    if os.path.exists(RAW_RESULTS_FILE):
        results = pd.read_csv(RAW_RESULTS_FILE).to_dict('records')

    for b_name, config in BENCHMARKS.items():
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            init_pop = rng.uniform(config["lb"], config["ub"], (N, D))

            for algo_name in ALGORITHMS:
                key = f"{algo_name}_{b_name}_{seed}"
                if key in progress: continue

                counter = TransparentCounter(config["func"], budget=BUDGET)
                np.random.seed(seed)
                opt = get_optimizer(algo_name, config, counter, init_pop, seed)

                start_time = time.time()
                res = opt.optimize()
                runtime = time.time() - start_time

                result_entry = {
                    "Algorithm": algo_name, "Benchmark": b_name, "Seed": seed,
                    "ActualNFE": counter.actual_executions, "BestFitness": res["best_score"],
                    "Runtime": runtime, "Termination": "Completed"
                }
                results.append(result_entry)

                conv_file = os.path.join(CONVERGENCE_DIR, f"{key}.csv")
                pd.Series(res["convergence_curve"]).to_csv(conv_file, index=False)
                if "diversity_curve" in res:
                    div_file = os.path.join(CONVERGENCE_DIR, f"{key}_diversity.csv")
                    pd.Series(res["diversity_curve"]).to_csv(div_file, index=False)

                progress[key] = True
                with open(PROGRESS_FILE, 'w') as f: json.dump(progress, f)
                pd.DataFrame(results).to_csv(RAW_RESULTS_FILE, index=False)

                print(f"Finished: {key} | Fitness: {res['best_score']:.5e} | NFE: {counter.actual_executions}")

if __name__ == "__main__":
    run_experiment()
