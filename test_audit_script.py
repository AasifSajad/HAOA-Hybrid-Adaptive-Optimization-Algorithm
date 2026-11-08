import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from HAOA.experiments.phase1_experiment import BENCHMARKS, N, D
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.comparison_algorithms.de import DifferentialEvolution
from run_transparent_verification import TransparentCountingWrapper

def test_initial_population_and_counter():
    seed = 42
    b_name = "Sphere"
    config = BENCHMARKS[b_name]

    # 1. Test initial population equality
    rng1 = np.random.default_rng(seed)
    pop1 = rng1.uniform(config["lb"], config["ub"], (N, D))

    rng2 = np.random.default_rng(seed)
    pop2 = rng2.uniform(config["lb"], config["ub"], (N, D))

    np.testing.assert_array_equal(pop1, pop2)
    print("Initial population array equality test: PASSED")

    # 2. Test counter and budget
    counter = TransparentCountingWrapper(config["func"], budget=15030)
    opt = HAOAImproved(
        objective_function=counter,
        dimension=D,
        lower_bound=config["lb"],
        upper_bound=config["ub"],
        population_size=N,
        max_iterations=500,
        initial_population=pop1,
        random_seed=seed,
        evaluations_limit=15030
    )
    res = opt.optimize()
    print(f"HAOAImproved actual executions: {counter.actual_executions}")
    assert counter.actual_executions == 15030
    print("Counter budget test: PASSED")

if __name__ == "__main__":
    test_initial_population_and_counter()
