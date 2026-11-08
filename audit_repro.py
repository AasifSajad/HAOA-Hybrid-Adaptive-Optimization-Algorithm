import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution
from HAOA.experiments.phase1_experiment import BENCHMARKS, N, D

def get_init_pop(seed, b_name):
    config = BENCHMARKS[b_name]
    return np.random.default_rng(seed).uniform(config["lb"], config["ub"], (N, D))

def trace_seeds():
    print(f"{'Seed':<5} | {'Algo':<12} | {'Init Pop Hash':<20} | {'Best Fitness':<15} | {'NFE'}")
    print("-" * 75)
    for seed in [42, 43, 44, 45, 46]:
        init_pop = get_init_pop(seed, "Sphere")
        h = hash(init_pop.tobytes())

        for algo in ["HAOA_orig", "HAOA_impr", "PSO", "DE"]:
            class C:
                def __init__(self, f): self.f=f; self.c=0
                def __call__(self, x): self.c+=1; return self.f(x)

            counter1 = C(BENCHMARKS["Sphere"]["func"])
            counter2 = C(BENCHMARKS["Sphere"]["func"])

            if algo == "HAOA_orig":
                np.random.seed(seed)
                opt1 = HAOAOriginal(counter1, D, -100, 100, N, 167)
                opt1.population = init_pop.copy()
                res1 = opt1.optimize()

                np.random.seed(seed)
                opt2 = HAOAOriginal(counter2, D, -100, 100, N, 167)
                opt2.population = init_pop.copy()
                res2 = opt2.optimize()
            elif algo == "HAOA_impr":
                opt1 = HAOAImproved(counter1, D, -100, 100, N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
                res1 = opt1.optimize()
                opt2 = HAOAImproved(counter2, D, -100, 100, N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
                res2 = opt2.optimize()
            elif algo == "PSO":
                np.random.seed(seed)
                for p in init_pop: counter1(p)
                opt1 = PSO(counter1, D, -100, 100, N, 500)
                opt1.positions = init_pop.copy()
                opt1.personal_best_positions = init_pop.copy()
                res1 = opt1.optimize()

                np.random.seed(seed)
                for p in init_pop: counter2(p)
                opt2 = PSO(counter2, D, -100, 100, N, 500)
                opt2.positions = init_pop.copy()
                opt2.personal_best_positions = init_pop.copy()
                res2 = opt2.optimize()
            elif algo == "DE":
                opt1 = DifferentialEvolution(counter1, D, -100, 100, N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
                res1 = opt1.optimize()
                opt2 = DifferentialEvolution(counter2, D, -100, 100, N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
                res2 = opt2.optimize()

            print(f"{seed:<5} | {algo:<12} | {h:<20} | {res1['best_score']:<15.5e} | {counter1.c}")
            assert res1['best_score'] == res2['best_score']

trace_seeds()
print("\nAll repeat runs matched successfully.")
