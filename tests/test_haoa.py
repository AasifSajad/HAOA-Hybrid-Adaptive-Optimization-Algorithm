import numpy as np
import pytest

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.algorithm.haoa_ml import MLHAOA
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution
from HAOA.benchmark_functions.sphere import sphere
from HAOA.benchmark_functions.rastrigin import rastrigin
from HAOA.benchmark_functions.griewank import griewank
from HAOA.benchmark_functions.ackley import ackley
from HAOA.benchmark_functions.rosenbrock import rosenbrock


def counting_sphere(x, counter_list):
    counter_list[0] += 1
    return sphere(x)


class TestHAOAUnit:

    def test_nfe_budget_ml_haoa(self):
        eval_counter = [0]

        def obj(x):
            return counting_sphere(x, eval_counter)

        dim = 10
        pop_size = 30
        max_iters = 500

        opt = MLHAOA(
            objective_function=obj,
            dimension=dim,
            lower_bound=-100,
            upper_bound=100,
            population_size=pop_size,
            max_iterations=max_iters,
            random_seed=42,
            evaluations_limit=15030
        )
        res = opt.optimize()

        expected_nfe = pop_size * (max_iters + 1)
        assert eval_counter[0] == expected_nfe
        assert res["evaluations_count"] == expected_nfe
        assert expected_nfe == 15030

    def test_nfe_budget_improved_haoa(self):
        eval_counter = [0]

        def obj(x):
            return counting_sphere(x, eval_counter)

        dim = 10
        pop_size = 30
        max_iters = 500

        opt = HAOAImproved(
            objective_function=obj,
            dimension=dim,
            lower_bound=-100,
            upper_bound=100,
            population_size=pop_size,
            max_iterations=max_iters,
            random_seed=42,
            evaluations_limit=15030
        )
        res = opt.optimize()

        expected_nfe = pop_size * (max_iters + 1)
        assert eval_counter[0] == expected_nfe
        assert res["evaluations_count"] == expected_nfe
        assert expected_nfe == 15030

    def test_nfe_budget_original_haoa(self):
        eval_counter = [0]

        def obj(x):
            return counting_sphere(x, eval_counter)

        dim = 10
        pop_size = 30
        max_iters = 167

        opt = HAOAOriginal(
            objective_function=obj,
            dimension=dim,
            lower_bound=-100,
            upper_bound=100,
            population_size=pop_size,
            max_iterations=max_iters
        )
        res = opt.optimize()

        expected_nfe = 3 * pop_size * max_iters
        assert eval_counter[0] == expected_nfe
        assert expected_nfe == 15030

    def test_nfe_budget_de(self):
        eval_counter = [0]

        def obj(x):
            return counting_sphere(x, eval_counter)

        dim = 10
        pop_size = 30
        max_gens = 500

        opt = DifferentialEvolution(
            objective_function=obj,
            dimension=dim,
            lower_bound=-100,
            upper_bound=100,
            population_size=pop_size,
            max_generations=max_gens,
            random_seed=42,
            evaluations_limit=15030
        )
        res = opt.optimize()

        expected_nfe = pop_size * (max_gens + 1)
        assert eval_counter[0] == expected_nfe
        assert res["evaluations_count"] == expected_nfe
        assert expected_nfe == 15030

    def test_nfe_budget_pso_with_wrapper(self):
        eval_counter = [0]

        def obj(x):
            return counting_sphere(x, eval_counter)

        dim = 10
        pop_size = 30
        max_iters = 500

        rng = np.random.default_rng(42)
        X0 = rng.uniform(-100, 100, (pop_size, dim))

        for ind in X0:
            obj(ind)
        assert eval_counter[0] == 30

        pso = PSO(
            objective_function=obj,
            dimension=dim,
            lower_bound=-100,
            upper_bound=100,
            population_size=pop_size,
            max_iterations=max_iters
        )
        pso.optimize()

        total_nfe = eval_counter[0]
        assert total_nfe == 15030

    def test_initial_population_pairing(self):
        dim = 5
        pop_size = 10
        rng = np.random.default_rng(123)
        init_pop = rng.uniform(-10, 10, (pop_size, dim))

        haoa_impr = HAOAImproved(
            objective_function=sphere,
            dimension=dim,
            lower_bound=-10,
            upper_bound=10,
            population_size=pop_size,
            max_iterations=10,
            initial_population=init_pop,
            random_seed=42
        )

        de = DifferentialEvolution(
            objective_function=sphere,
            dimension=dim,
            lower_bound=-10,
            upper_bound=10,
            population_size=pop_size,
            max_generations=10,
            initial_population=init_pop,
            random_seed=42
        )

        np.testing.assert_array_equal(haoa_impr.population, init_pop)
        np.testing.assert_array_equal(de.population, init_pop)

    def test_monotonicity_of_best_score(self):
        opt = HAOAImproved(
            objective_function=rastrigin,
            dimension=5,
            lower_bound=-5.12,
            upper_bound=5.12,
            population_size=20,
            max_iterations=100,
            random_seed=99
        )
        res = opt.optimize()
        curve = res["convergence_curve"]

        for i in range(1, len(curve)):
            assert curve[i] <= curve[i - 1] + 1e-12

    def test_velocity_damping_on_rejection(self):
        opt = HAOAImproved(
            objective_function=sphere,
            dimension=2,
            lower_bound=-10,
            upper_bound=10,
            population_size=5,
            max_iterations=5,
            random_seed=42,
            alpha=0.5
        )
        opt._update_elite()
        opt.velocity[0] = np.array([2.0, 2.0])
        old_vel = opt.velocity[0].copy()

        opt.fitness_values[0] = -1e9
        opt.update_population(0)

        assert opt.velocity[0][0] < old_vel[0]

    def test_levy_scaling_domain_invariance(self):
        opt_small = HAOAImproved(
            objective_function=sphere,
            dimension=5,
            lower_bound=-5.12,
            upper_bound=5.12,
            population_size=10,
            max_iterations=1,
            random_seed=1,
            gamma=0.01
        )

        opt_large = HAOAImproved(
            objective_function=sphere,
            dimension=5,
            lower_bound=-600,
            upper_bound=600,
            population_size=10,
            max_iterations=1,
            random_seed=1,
            gamma=0.01
        )

        scale_small = opt_small.gamma * (opt_small.upper_bound - opt_small.lower_bound)
        scale_large = opt_large.gamma * (opt_large.upper_bound - opt_large.lower_bound)

        assert pytest.approx(scale_small) == 0.01 * 10.24
        assert pytest.approx(scale_large) == 0.01 * 1200.0

    def test_elite_protection(self):
        opt = HAOAImproved(
            objective_function=sphere,
            dimension=3,
            lower_bound=-10,
            upper_bound=10,
            population_size=10,
            max_iterations=50,
            random_seed=77
        )
        res = opt.optimize()

        best_score = res["best_score"]
        best_sol = res["best_solution"]

        assert sphere(best_sol) == pytest.approx(best_score)


def test_optimizer_returns_complete_finite_result():
    def test_obj(x):
        return float(np.sum(np.asarray(x) ** 2))
    opt = HAOAOriginal(
        objective_function=test_obj,
        dimension=4,
        lower_bound=-3.0,
        upper_bound=3.0,
        population_size=12,
        max_iterations=15
    )
    result = opt.optimize()
    assert np.isfinite(result["best_score"])
    assert result["best_solution"].shape == (4,)
    assert len(result["convergence_curve"]) == 15
    assert np.all(np.isfinite(result["convergence_curve"]))
    assert np.all(np.diff(result["convergence_curve"]) <= 1e-12)
