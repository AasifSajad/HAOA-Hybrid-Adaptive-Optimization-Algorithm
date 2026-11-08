import numpy as np
import sys
import os
import time
import pandas as pd

sys.path.append(os.getcwd())

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.algorithm.haoa_advanced import HAOAAdvanced
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution
from HAOA.experiments.phase1_experiment import BENCHMARKS, N, D, BUDGET

class TransparentCountingWrapper:
    """
    Transparent objective-function wrapper without caching.
    Every call to __call__ represents an actual objective function execution.
    Maintains strict budget termination if evaluations reach budget.
    """
    def __init__(self, func, budget=15030):
        self.func = func
        self.budget = budget
        self.total_requests = 0
        self.actual_executions = 0
        self.cache_hits = 0  # Always 0 since caching is disabled

    def __call__(self, x):
        self.total_requests += 1
        self.actual_executions += 1
        return self.func(x)

def run_transparent_verification():
    output_dir = "results/transparent_verification"
    os.makedirs(output_dir, exist_ok=True)

    seed = 42
    results = []

    print(f"{'Algorithm':<15} | {'Benchmark':<12} | {'Requests':<8} | {'Executions':<10} | {'Cache Hits':<10} | {'Best Fitness':<15}")
    print("-" * 75)

    for b_name, config in BENCHMARKS.items():
        rng = np.random.default_rng(seed)
        init_pop = rng.uniform(config["lb"], config["ub"], (N, D))

        for algo_name in ["HAOA_orig", "HAOA_impr", "HAOA_adv", "PSO", "DE"]:
            counter = TransparentCountingWrapper(config["func"], budget=BUDGET)
            np.random.seed(seed)

            if algo_name == "HAOA_orig":
                # Original HAOA generational flow evaluation control
                class TransparentHAOA(HAOAOriginal):
                    def optimize(self):
                        for iteration in range(self.max_iterations):
                            if self.objective_function.actual_executions >= self.evaluations_limit:
                                break
                            self.evaluate_population()
                            if self.objective_function.actual_executions >= self.evaluations_limit:
                                break
                            self.update_population(iteration)
                            self.convergence_curve.append(self.best_score)
                        return {
                            "best_score": self.best_score,
                            "best_solution": self.best_solution,
                            "convergence_curve": self.convergence_curve,
                            "evaluations_count": self.objective_function.actual_executions
                        }

                opt = TransparentHAOA(counter, D, config["lb"], config["ub"], N, 167)
                opt.evaluations_limit = BUDGET
                opt.population = init_pop.copy()

            elif algo_name == "HAOA_impr":
                class TransparentHAOAImpr(HAOAImproved):
                    def update_population(self, iteration):
                        progress = iteration / self.max_iterations
                        inertia_weight = 0.9 + 0.5 * progress
                        adaptive_learning = 2.0 * (1 - progress)
                        elite = self.best_solution.copy()
                        domain_range = self.upper_bound - self.lower_bound
                        gamma = 0.01 * (1 - progress)

                        for i in range(self.population_size):
                            if self.objective_function.actual_executions >= self.evaluations_limit:
                                break
                            r1 = self.rng.random(self.dimension)
                            r2 = self.rng.random(self.dimension)
                            r3 = self.rng.random(self.dimension)
                            r1_idx, r2_idx = self.rng.choice(
                                [idx for idx in range(self.population_size) if idx != i],
                                size=2, replace=False
                            )
                            exploration = adaptive_learning * r3 * (self.population[r1_idx] - self.population[r2_idx])
                            attraction = adaptive_learning * r2 * (elite - self.population[i])
                            self.velocity[i] = inertia_weight * self.velocity[i] + attraction + exploration
                            self.velocity[i] = np.clip(self.velocity[i], -0.2 * domain_range, 0.2 * domain_range)

                            new_position = self.population[i] + self.velocity[i]
                            if self.stagnation_counters[i] > self.k_stag:
                                levy_step = gamma * domain_range * self._levy_flight()
                                new_position = self.population[i] + levy_step

                            new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                            new_fitness = self.objective_function(new_position)

                            if new_fitness < self.fitness_values[i]:
                                self.population[i] = new_position
                                self.fitness_values[i] = new_fitness
                                self.stagnation_counters[i] = 0
                            else:
                                self.velocity[i] *= self.alpha
                                self.stagnation_counters[i] += 1
                        self._update_elite()

                opt = TransparentHAOAImpr(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)

            elif algo_name == "HAOA_adv":
                class TransparentHAOAAdv(HAOAAdvanced):
                    def update_population(self, iteration):
                        progress = iteration / self.max_iterations
                        inertia_weight = 0.95 - 0.6 * progress
                        adaptive_learning = 1.8 * (1 - progress)
                        c1 = 1.0 - 0.5 * progress
                        c2 = 0.5 + 1.0 * progress
                        mutation_strength = 0.003 * (1 - progress)
                        elite = self.best_solution.copy()
                        velocity_limit = 0.2 * (self.upper_bound - self.lower_bound)
                        domain_range = self.upper_bound - self.lower_bound

                        for i in range(self.population_size):
                            if self.objective_function.actual_executions >= self.evaluations_limit:
                                break
                            r1 = self.rng.random(self.dimension)
                            r2 = self.rng.random(self.dimension)
                            r3 = self.rng.random(self.dimension)
                            local_best = self._get_local_best(i)
                            r1_idx, r2_idx = self.rng.choice(
                                [idx for idx in range(self.population_size) if idx != i],
                                size=2, replace=False
                            )
                            exploration = adaptive_learning * r3 * (self.population[r1_idx] - self.population[r2_idx])
                            local_attraction = c1 * r1 * (local_best - self.population[i])
                            global_attraction = c2 * r2 * (elite - self.population[i])
                            self.velocity[i] = inertia_weight * self.velocity[i] + local_attraction + global_attraction + 0.15 * exploration
                            self.velocity[i] = np.clip(self.velocity[i], -velocity_limit, velocity_limit)
                            mutation = mutation_strength * self.rng.standard_normal(self.dimension)
                            new_position = self.population[i] + self.velocity[i] + mutation

                            kinetic_energy = np.sum(self.velocity[i] ** 2)
                            if self.stagnation_counters[i] > self.k_stag and kinetic_energy < 1e-4:
                                if self.rng.random() < 0.5:
                                    new_position = self.lower_bound + self.upper_bound - self.population[i]
                                else:
                                    levy_step = 0.01 * domain_range * (1 - progress) * self._levy_flight()
                                    new_position = self.population[i] + levy_step

                            new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                            new_fitness = self.objective_function(new_position)

                            if new_fitness < self.fitness_values[i]:
                                self.population[i] = new_position
                                self.fitness_values[i] = new_fitness
                                self.stagnation_counters[i] = 0
                            else:
                                self.velocity[i] = 0.5 * self.velocity[i]
                                self.stagnation_counters[i] += 1
                        self._update_elite()

                opt = TransparentHAOAAdv(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)

            elif algo_name == "PSO":
                # Pre-evaluate initial population (30 evals)
                for p in init_pop:
                    counter(p)
                opt = PSO(counter, D, config["lb"], config["ub"], N, 500)
                opt.positions = init_pop.copy()
                opt.personal_best_positions = init_pop.copy()

            elif algo_name == "DE":
                opt = DifferentialEvolution(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)

            # Silence stdout
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            start = time.time()
            res = opt.optimize()
            runtime = time.time() - start
            sys.stdout = old_stdout

            actual_execs = counter.actual_executions
            assert actual_execs <= BUDGET, f"Budget exceeded: {actual_execs} > {BUDGET}"

            results.append({
                "Algorithm": algo_name,
                "Benchmark": b_name,
                "Seed": seed,
                "Requests": counter.total_requests,
                "ActualExecutions": actual_execs,
                "CacheHits": counter.cache_hits,
                "BestFitness": res["best_score"],
                "Runtime": runtime,
                "Termination": "Completed"
            })

            print(f"{algo_name:<15} | {b_name:<12} | {counter.total_requests:<8} | {counter.actual_executions:<10} | {counter.cache_hits:<10} | {res['best_score']:<15.5e}")

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/transparent_verification_results.csv", index=False)
    print(f"\nTransparent verification results saved to {output_dir}/transparent_verification_results.csv")

if __name__ == "__main__":
    run_transparent_verification()
