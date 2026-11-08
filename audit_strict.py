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
    def __init__(self, func, enable_cache=False):
        self.func = func
        self.total_requests = 0
        self.actual_calls = 0
        self.cache_hits = 0
        self.seen_hashes = set()
        self.enable_cache = enable_cache

    def __call__(self, x):
        self.total_requests += 1
        h = hash(x.tobytes())
        if self.enable_cache and h in self.seen_hashes:
            self.cache_hits += 1
        else:
            self.seen_hashes.add(h)
            self.actual_calls += 1
        return self.func(x)

def run_verify(enable_cache):
    results = []
    for b_name, config in BENCHMARKS.items():
        seed = 42
        rng = np.random.default_rng(seed)
        init_pop = rng.uniform(config["lb"], config["ub"], (N, D))

        for algo_name in ["HAOA_orig", "HAOA_impr", "PSO", "DE"]:
            counter = MetricsCounter(config["func"], enable_cache=enable_cache)

            init_evals = 0
            if algo_name == "HAOA_orig":
                opt = HAOAOriginal(counter, D, config["lb"], config["ub"], N, 167)
                opt.population = init_pop.copy()
                opt.optimize()
            elif algo_name == "HAOA_impr":
                opt = HAOAImproved(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
                opt.optimize()
            elif algo_name == "PSO":
                # Harness wrapper pre-evals init population
                for p in init_pop:
                    counter(p)
                    init_evals += 1
                opt = PSO(counter, D, config["lb"], config["ub"], N, 500)
                opt.positions = init_pop.copy()
                opt.personal_best_positions = init_pop.copy()
                opt.optimize()
            elif algo_name == "DE":
                opt = DifferentialEvolution(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
                opt.optimize()

            results.append({
                "Benchmark": b_name, "Algo": algo_name, "Reqs": counter.total_requests,
                "Calls": counter.actual_calls, "Hits": counter.cache_hits, "Init": init_evals
            })
    return results

print("--- UNCACHED VERIFICATION (Cache disabled) ---")
uncached = run_verify(False)
for r in uncached: print(r)

print("\n--- CACHED VERIFICATION (Cache enabled) ---")
cached = run_verify(True)
for r in cached: print(r)
