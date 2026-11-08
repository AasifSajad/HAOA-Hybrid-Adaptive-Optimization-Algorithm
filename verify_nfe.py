import numpy as np
import sys
import os

# Ensure we can import from the repo
sys.path.append(os.getcwd())

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution

class CountingWrapper:
    def __init__(self, func):
        self.func = func
        self.count = 0
    def __call__(self, x):
        self.count += 1
        return self.func(x)

def dummy_obj(x):
    return np.sum(x**2)

def verify_nfe():
    N = 30
    D = 30

    print("--- NFE Verification (1 iteration test) ---")

    # 1. Original HAOA
    # Math: 3 * N * T. For 1 iteration, expect 3 * 30 = 90.
    counter = CountingWrapper(dummy_obj)
    opt = HAOAOriginal(counter, D, -10, 10, N, 1)
    opt.optimize()
    print(f"Original HAOA (1 iter): Actual={counter.count}, Expected=90")

    # 2. Improved HAOA
    # Math: N (init) + N * T. For 1 iteration, expect 30 + 30 = 60.
    counter = CountingWrapper(dummy_obj)
    opt = HAOAImproved(counter, D, -10, 10, N, 1, random_seed=42)
    opt.optimize()
    print(f"Improved HAOA (1 iter): Actual={counter.count}, Expected=60")

    # 3. PSO
    # Math: N * T. For 1 iteration, expect 30.
    counter = CountingWrapper(dummy_obj)
    opt = PSO(counter, D, -10, 10, N, 1)
    opt.optimize()
    print(f"PSO (1 iter): Actual={counter.count}, Expected=30")

    # 4. DE
    # Math: N (init) + N * T. For 1 iteration, expect 30 + 30 = 60.
    counter = CountingWrapper(dummy_obj)
    opt = DifferentialEvolution(counter, D, -10, 10, N, 1, random_seed=42)
    opt.optimize()
    print(f"DE (1 iter): Actual={counter.count}, Expected=60")

    print("\n--- Full Run Projections (to 15,030) ---")
    # HAOA_orig: 167 * 90 = 15,030
    # HAOA_impr: 30 + 500 * 30 = 15,030
    # DE: 30 + 500 * 30 = 15,030
    # PSO: 501 * 30 = 15,030 (using T=501)

    print("Projected HAOA_orig (T=167): 167 * 90 = 15,030")
    print("Projected HAOA_impr (T=500): 30 + 500 * 30 = 15,030")
    print("Projected DE (T=500): 30 + 500 * 30 = 15,030")
    print("Projected PSO (T=501): 501 * 30 = 15,030")

if __name__ == "__main__":
    verify_nfe()
