import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from HAOA.experiments.phase1_experiment import ExperimentHarness, BENCHMARKS, BUDGET

class CountingWrapper:
    def __init__(self, func):
        self.func = func
        self.count = 0
    def __call__(self, x):
        self.count += 1
        return self.func(x)

def test_full_budget_nfe_counts():
    harness = ExperimentHarness()
    benchmark_name = "Sphere"
    seed = 42

    print("=== Testing Full-Run NFE Budgets (Target = 15,030) ===")

    for algo_name in ["HAOA_orig", "HAOA_impr", "PSO", "DE"]:
        # Wrap the function
        original_func = BENCHMARKS[benchmark_name]["func"]
        counter = CountingWrapper(original_func)

        # Temporarily swap the benchmark function
        BENCHMARKS[benchmark_name]["func"] = counter

        res = harness.run_single(algo_name, benchmark_name, seed)

        # Restore
        BENCHMARKS[benchmark_name]["func"] = original_func

        print(f"[{algo_name}] Target NFE: {BUDGET} | Actual Objective Calls: {counter.count}")
        assert counter.count == BUDGET, f"Mismatch for {algo_name}: {counter.count} != {BUDGET}"

if __name__ == "__main__":
    test_full_budget_nfe_counts()
