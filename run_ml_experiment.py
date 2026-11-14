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
from HAOA.algorithm.haoa_ml import MLHAOA
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
ALGORITHMS = ["HAOA_orig", "HAOA_impr", "HAOA_adv", "MLHAOA", "PSO", "DE"]
N = 30
D = 30
BUDGET = 15030
SEEDS = list(range(100, 130))  # 30 independent seeds

OUTPUT_DIR = "results/ml_experiment"
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
class BoundedHAOAOriginal(HAOAOriginal):
    def optimize(self):
        for iteration in range(self.max_iterations):
            if self.objective_function.actual_executions >= BUDGET: break
            self.evaluate_population()
            if self.objective_function.actual_executions >= BUDGET: break
            self.update_population(iteration)
            self.convergence_curve.append(self.best_score)
        return {"best_score": self.best_score, "best_solution": self.best_solution, "convergence_curve": self.convergence_curve}

class BoundedHAOAImproved(HAOAImproved):
    def optimize(self):
        self._update_elite()
        for iteration in range(self.max_iterations):
            if self.objective_function.actual_executions >= BUDGET: break
            self.update_population(iteration)
            self.convergence_curve.append(self.best_score)
        return {"best_score": self.best_score, "best_solution": self.best_solution, "convergence_curve": self.convergence_curve}

class BoundedHAOAAdvanced(HAOAAdvanced):
    def optimize(self):
        self.evaluations_limit = BUDGET
        self._update_elite()
        for iteration in range(self.max_iterations):
            if self.objective_function.actual_executions >= BUDGET: break
            self.update_population(iteration)
            self.convergence_curve.append(self.best_score)
        return {"best_score": self.best_score, "best_solution": self.best_solution, "convergence_curve": self.convergence_curve}

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

class BoundedDE(DifferentialEvolution):
    def optimize(self):
        self._update_best()
        for generation in range(self.max_generations):
            for i in range(self.population_size):
                if self.objective_function.actual_executions >= BUDGET: break
                trial = self._mutate_and_crossover(i)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                trial_fitness = self.objective_function(trial)
                self.evaluations_count += 1
                if trial_fitness < self.fitness_values[i]:
                    self.population[i] = trial
                    self.fitness_values[i] = trial_fitness
            self._update_best()
            self.convergence_curve.append(self.best_score)
        return {"best_score": self.best_score, "best_solution": self.best_solution, "convergence_curve": self.convergence_curve, "evaluations_count": self.evaluations_count}

class BoundedMLHAOA(MLHAOA):
    pass

def get_optimizer(algo_name, config, counter, init_pop, seed):
    if algo_name == "HAOA_orig":
        opt = BoundedHAOAOriginal(counter, D, config["lb"], config["ub"], N, 167)
        opt.evaluations_limit = BUDGET
        opt.population = init_pop.copy()
        return opt
    elif algo_name == "HAOA_impr":
        return BoundedHAOAImproved(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)
    elif algo_name == "HAOA_adv":
        return BoundedHAOAAdvanced(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)
    elif algo_name == "MLHAOA":
        return BoundedMLHAOA(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)
    elif algo_name == "PSO":
        opt = BoundedPSO(counter, D, config["lb"], config["ub"], N, 501)
        opt.evaluations_limit = BUDGET
        opt.positions = init_pop.copy()
        opt.personal_best_positions = init_pop.copy()
        return opt
    elif algo_name == "DE":
        return BoundedDE(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)
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
                result_entry = {"Algorithm": algo_name, "Benchmark": b_name, "Seed": seed, "ActualNFE": counter.actual_executions, "BestFitness": res["best_score"], "Runtime": runtime, "Termination": "Completed"}
                results.append(result_entry)
                conv_file = os.path.join(CONVERGENCE_DIR, f"{key}.csv")
                pd.Series(res["convergence_curve"]).to_csv(conv_file, index=False)
                progress[key] = True
                with open(PROGRESS_FILE, 'w') as f: json.dump(progress, f)
                pd.DataFrame(results).to_csv(RAW_RESULTS_FILE, index=False)
                print(f"Finished: {key} | Fitness: {res['best_score']:.5e} | NFE: {counter.actual_executions}")

if __name__ == "__main__":
    run_experiment()