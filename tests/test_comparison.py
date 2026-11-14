import numpy as np
import pytest

from HAOA.algorithm.haoa import HAOA
from HAOA.comparison_algorithms.pso import PSO
from HAOA.experiments.dataset_comparison import SimplePSO, summarize_paired_scores


def sphere(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


@pytest.mark.parametrize("optimizer_class", [PSO, SimplePSO])
def test_pso_returns_finite_bounded_monotone_result(optimizer_class):
    result = optimizer_class(sphere, 5, -4.0, 4.0, 10, 12, random_state=17).optimize()
    assert np.isfinite(result["best_score"]) and result["best_score"] >= 0
    assert len(result["convergence_curve"]) == 12
    assert np.all(np.isfinite(result["convergence_curve"]))
    assert np.all(np.diff(result["convergence_curve"]) <= 1e-12)
    assert result["best_solution"].shape == (5,)
    assert np.all(result["best_solution"] >= -4.0)
    assert np.all(result["best_solution"] <= 4.0)


@pytest.mark.parametrize("optimizer_class", [PSO, SimplePSO])
def test_pso_same_seed_reproduces_solution_and_curve(optimizer_class):
    args = (sphere, 4, -3.0, 3.0, 9, 11)
    first = optimizer_class(*args, random_state=2026).optimize()
    second = optimizer_class(*args, random_state=2026).optimize()
    assert first["best_score"] == pytest.approx(second["best_score"])
    np.testing.assert_array_equal(first["best_solution"], second["best_solution"])
    np.testing.assert_array_equal(first["convergence_curve"], second["convergence_curve"])


def test_pso_does_not_mutate_global_numpy_rng():
    np.random.seed(123)
    before = np.random.get_state()
    PSO(sphere, 3, -2.0, 2.0, 8, 5, random_state=9).optimize()
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("optimizer_class", [HAOA, PSO, SimplePSO])
def test_optimizers_respect_vector_bounds(optimizer_class):
    args = dict(objective_function=sphere, dimension=3,
                lower_bound=np.array([-1.0, -2.0, -3.0]),
                upper_bound=np.array([1.0, 2.0, 3.0]),
                population_size=8, max_iterations=8)
    if optimizer_class is HAOA:
        args["random_state"] = 5
    result = optimizer_class(**args).optimize()
    assert np.isfinite(result["best_score"])
    assert len(result["convergence_curve"]) == 8
    assert np.all(result["best_solution"] >= args["lower_bound"])
    assert np.all(result["best_solution"] <= args["upper_bound"])


def test_paired_summary_reports_wins_and_confidence_interval():
    result = summarize_paired_scores([1.0, 3.0, 2.0, 4.0], [2.0, 2.0, 2.0, 4.0])
    assert result["runs"] == 4
    assert result["haoa_wins"] == 2
    assert result["pso_wins"] == 1
    assert result["ties"] == 1
    assert result["median_paired_difference"] == pytest.approx(0.0)
    assert result["paired_median_ci_2_5"] <= result["paired_median_ci_97_5"]


@pytest.mark.parametrize("left,right", [([], []), ([1.0], [1.0, 2.0]), ([np.nan], [1.0])])
def test_paired_summary_rejects_invalid_samples(left, right):
    with pytest.raises(ValueError):
        summarize_paired_scores(left, right)


def test_pso_rejects_invalid_bounds_and_non_finite_objective():
    with pytest.raises(ValueError, match="lower bound"):
        PSO(sphere, 2, 2.0, 1.0, 5, 4)
    with pytest.raises(ValueError, match="non-finite"):
        PSO(lambda _: float("nan"), 2, -1.0, 1.0, 5, 4, random_state=1).optimize()
