import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution
from HAOA.experiments.phase1_experiment import BENCHMARKS, N, D

class MetricsCounter:
    def __init__(self, func):
        self.func = func
        self.total_requests = 0
        self.actual_calls = 0
        self.cache_hits = 0
        self.seen_hashes = set()

    def __call__(self, x):
        self.total_requests += 1
        h = hash(x.tobytes())
        if h in self.seen_hashes:
            self.cache_hits += 1
        else:
            self.seen_hashes.add(h)
            self.actual_calls += 1
        return self.func(x)

def verify_audit():
    seed = 42

    print(f"{'Benchmark':<12} | {'Algorithm':<12} | {'Requests':<8} | {'Calls':<8} | {'Hits':<8}")
    print("-" * 65)

    for b_name, config in BENCHMARKS.items():
        init_pop = np.random.default_rng(seed).uniform(config["lb"], config["ub"], (N, D))

        for algo_name in ["HAOA_orig", "HAOA_impr", "PSO", "DE"]:
            counter = MetricsCounter(config["func"])

            if algo_name == "HAOA_orig":
                opt = HAOAOriginal(counter, D, config["lb"], config["ub"], N, 167)
                opt.population = init_pop.copy()
                opt.optimize()

            elif algo_name == "HAOA_impr":
                opt = HAOAImproved(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
                opt.optimize()

            elif algo_name == "PSO":
                # Harness wrapper pre-evals init population (30)
                for p in init_pop: counter(p)
                opt = PSO(counter, D, config["lb"], config["ub"], N, 500)
                opt.positions = init_pop.copy()
                opt.personal_best_positions = init_pop.copy()
                opt.optimize()

            elif algo_name == "DE":
                opt = DifferentialEvolution(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
                opt.optimize()

            print(f"{b_name:<12} | {algo_name:<12} | {counter.total_requests:<8} | {counter.actual_calls:<8} | {counter.cache_hits:<8}")

def verify_reproducibility():
    print("\n--- Reproducibility Test (Seed 123) ---")
    b = "Sphere"
    config = BENCHMARKS[b]
    init_pop = np.random.default_rng(123).uniform(config["lb"], config["ub"], (N, D))

    def run():
        opt = HAOAImproved(config["func"], D, config["lb"], config["ub"], N, 10, initial_population=init_pop, random_seed=123)
        return opt.optimize()["best_score"]

    s1 = run()
    s2 = run()
    print(f"Result 1: {s1:.6e}")
    print(f"Result 2: {s2:.6e}")
    assert s1 == s2
    print("Reproducibility verified.")

if __name__ == "__main__":
    verify_audit()
    verify_reproducibility()
