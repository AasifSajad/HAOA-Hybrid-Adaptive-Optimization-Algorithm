import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution
from HAOA.experiments.phase1_experiment import BENCHMARKS, N, D

class CountingWrapper:
    def __init__(self, func):
        self.func = func
        self.total_calls = 0
        self.unique_calls = 0
        self.cache_hits = 0
        self.seen_hashes = set()

    def __call__(self, x):
        self.total_calls += 1
        h = hash(x.tobytes())
        if h in self.seen_hashes:
            self.cache_hits += 1
        else:
            self.seen_hashes.add(h)
            self.unique_calls += 1
        return self.func(x)

def get_initial_population(seed, benchmark_name):
    config = BENCHMARKS[benchmark_name]
    rng = np.random.default_rng(seed)
    return rng.uniform(config["lb"], config["ub"], (N, D))

def verify_full_suite():
    seed = 42
    target = 15030

    print(f"{'Benchmark':<12} | {'Algorithm':<12} | {'Total Calls':<12} | {'Unique Calls':<12} | {'Cache Hits':<12}")
    print("-" * 75)

    for b_name, config in BENCHMARKS.items():
        init_pop = get_initial_population(seed, b_name)

        for algo_name in ["HAOA_orig", "HAOA_impr", "PSO", "DE"]:
            counter = CountingWrapper(config["func"])

            if algo_name == "HAOA_orig":
                # T = 167 iterations. Per iter: 3N = 90.
                # Total calls = 167 * 90 = 15,030.
                opt = HAOAOriginal(counter, D, config["lb"], config["ub"], N, 167)
                opt.population = init_pop.copy()
                opt.optimize()

            elif algo_name == "HAOA_impr":
                # Init = N (30). T = 500. Per iter = 30.
                # Total calls = 30 + 500*30 = 15,030.
                opt = HAOAImproved(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=target)
                opt.optimize()

            elif algo_name == "PSO":
                # PSO Init = N (30 wrapper). T = 500 iterations (15,000).
                # Total calls = 30 + 15,000 = 15,030.
                for p in init_pop: counter(p)
                opt = PSO(counter, D, config["lb"], config["ub"], N, 500)
                opt.positions = init_pop.copy()
                opt.personal_best_positions = init_pop.copy()
                opt.optimize()

            elif algo_name == "DE":
                # Init = N (30). T = 500. Per iter = 30.
                # Total calls = 30 + 500*30 = 15,030.
                opt = DifferentialEvolution(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=target)
                opt.optimize()

            print(f"{b_name:<12} | {algo_name:<12} | {counter.total_calls:<12} | {counter.unique_calls:<12} | {counter.cache_hits:<12}")
            assert counter.total_calls == target, f"{algo_name} failed on {b_name}: total={counter.total_calls}"

def test_reproducibility():
    b_name = "Sphere"
    seed = 123
    init_pop1 = get_initial_population(seed, b_name)
    init_pop2 = get_initial_population(seed, b_name)
    assert np.array_equal(init_pop1, init_pop2)

    opt1 = HAOAImproved(BENCHMARKS[b_name]["func"], D, -100, 100, N, 10, initial_population=init_pop1, random_seed=seed)
    opt2 = HAOAImproved(BENCHMARKS[b_name]["func"], D, -100, 100, N, 10, initial_population=init_pop2, random_seed=seed)

    res1 = opt1.optimize()
    res2 = opt2.optimize()
    assert res1["best_score"] == res2["best_score"]
    print(f"\nReproducibility verified: Same seed results match exactly (Score: {res1['best_score']:.5e})")

if __name__ == "__main__":
    verify_full_suite()
    test_reproducibility()
