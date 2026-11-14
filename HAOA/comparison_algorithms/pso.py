"""Reproducible bounded particle swarm optimization baseline."""
import numpy as np


class PSO:
    """Global-best PSO with an isolated NumPy random generator."""

    def __init__(self, objective_function, dimension, lower_bound, upper_bound,
                 population_size, max_iterations, *, random_state=None,
                 verbose=False):
        if not callable(objective_function):
            raise TypeError("objective_function must be callable")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(population_size, int) or isinstance(population_size, bool) or population_size < 2:
            raise ValueError("population_size must be an integer >= 2")
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.objective_function = objective_function
        self.dimension = dimension
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.verbose = bool(verbose)
        self.rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        self.lower_bound = np.broadcast_to(np.asarray(lower_bound, dtype=float), (dimension,)).copy()
        self.upper_bound = np.broadcast_to(np.asarray(upper_bound, dtype=float), (dimension,)).copy()
        if not np.all(np.isfinite(self.lower_bound)) or not np.all(np.isfinite(self.upper_bound)):
            raise ValueError("bounds must be finite")
        if np.any(self.lower_bound >= self.upper_bound):
            raise ValueError("each lower bound must be below its upper bound")
        self.positions = self.rng.uniform(self.lower_bound, self.upper_bound, (population_size, dimension))
        self.velocities = self.rng.uniform(-1.0, 1.0, (population_size, dimension))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.convergence_curve = []

    def _evaluate(self, position):
        value = np.asarray(self.objective_function(position.copy()))
        if value.ndim != 0:
            raise ValueError("objective_function must return a scalar")
        score = float(value)
        if not np.isfinite(score):
            raise ValueError("objective_function returned a non-finite value")
        return score

    def optimize(self):
        w, c1, c2 = 0.7, 1.5, 1.5
        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                fitness = self._evaluate(self.positions[i])
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i].copy()
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i].copy()
            for i in range(self.population_size):
                r1 = self.rng.random(self.dimension)
                r2 = self.rng.random(self.dimension)
                cognitive = c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social = c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = w * self.velocities[i] + cognitive + social
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i],
                                            self.lower_bound, self.upper_bound)
            self.convergence_curve.append(self.global_best_score)
            if self.verbose:
                print(f"PSO Iteration {iteration + 1}/{self.max_iterations} | Best Score: {self.global_best_score:.10e}")
        for position in self.positions:
            fitness = self._evaluate(position)
            if fitness < self.global_best_score:
                self.global_best_score = fitness
                self.global_best_position = position.copy()
        if self.convergence_curve:
            self.convergence_curve[-1] = min(self.convergence_curve[-1], self.global_best_score)
        return {"best_score": float(self.global_best_score),
                "best_solution": self.global_best_position.copy(),
                "convergence_curve": list(self.convergence_curve)}
