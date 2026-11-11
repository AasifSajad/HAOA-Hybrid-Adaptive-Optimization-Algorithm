import numpy as np
import math


class HAOAAdvanced:

    def __init__(
        self,
        objective_function,
        dimension,
        lower_bound,
        upper_bound,
        population_size,
        max_iterations,
        initial_population=None,
        random_seed=None,
        evaluations_limit=None,
        k_stag=10,
        beta=1.5
    ):

        self.objective_function = objective_function
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.max_iterations = max_iterations

        self.k_stag = k_stag
        self.beta = beta

        self.rng = np.random.default_rng(random_seed)

        if initial_population is not None:
            self.population = np.array(initial_population, dtype=float).copy()
        else:
            self.population = self.rng.uniform(
                self.lower_bound,
                self.upper_bound,
                (self.population_size, self.dimension)
            )

        self.velocity = np.zeros(
            (self.population_size, self.dimension)
        )

        self.fitness_values = np.empty(self.population_size)
        for i in range(self.population_size):
            self.fitness_values[i] = self.objective_function(self.population[i])
        self.evaluations_count = self.population_size

        self.best_solution = None
        self.best_score = float("inf")
        self.convergence_curve = []
        self.stagnation_counters = np.zeros(self.population_size, dtype=int)
        self.diversity_curve = []

        self.evaluations_limit = evaluations_limit

    def _levy_flight(self):
        beta = self.beta
        sigma = (
            (
                math.gamma(1 + beta)
                * np.sin(np.pi * beta / 2)
            )
            /
            (
                math.gamma((1 + beta) / 2)
                * beta
                * 2 ** ((beta - 1) / 2)
            )
        ) ** (1 / beta)

        u = self.rng.standard_normal(self.dimension) * sigma
        v = self.rng.standard_normal(self.dimension)

        step = u / (np.abs(v) ** (1 / beta) + 1e-10)
        return step

    def _update_elite(self):
        best_idx = int(np.argmin(self.fitness_values))
        if self.fitness_values[best_idx] < self.best_score:
            self.best_score = self.fitness_values[best_idx]
            self.best_solution = self.population[best_idx].copy()

    def _get_local_best(self, i):
        # Ring neighborhood radius = 2
        n_pop = self.population_size
        indices = [(i + offset) % n_pop for offset in [-2, -1, 0, 1, 2]]
        local_fits = self.fitness_values[indices]
        best_local_idx = indices[int(np.argmin(local_fits))]
        return self.population[best_local_idx]

    def update_population(self, iteration):
        progress = iteration / self.max_iterations
        inertia_weight = 0.95 - 0.6 * progress
        adaptive_learning = 1.8 * (1 - progress)
        c1 = 1.2 * (1 - progress)
        c2 = 2.2 - 0.8 * progress
        mutation_strength = 0.003 * (1 - progress)

        elite = self.best_solution.copy()
        velocity_limit = 0.2 * (self.upper_bound - self.lower_bound)
        domain_range = self.upper_bound - self.lower_bound

        for i in range(self.population_size):
            if self.evaluations_limit is not None and self.evaluations_count >= self.evaluations_limit:
                break

            r1 = self.rng.random(self.dimension)
            r2 = self.rng.random(self.dimension)
            r3 = self.rng.random(self.dimension)

            local_best = self._get_local_best(i)

            r1_idx, r2_idx = self.rng.choice(
                [idx for idx in range(self.population_size) if idx != i],
                size=2, replace=False
            )
            exploration_de = adaptive_learning * r3 * (
                self.population[r1_idx] - self.population[r2_idx]
            )
            exploration_orig = adaptive_learning * self.rng.random(self.dimension) * (
                self.rng.uniform(self.lower_bound, self.upper_bound, self.dimension) - self.population[i]
            )

            # Multi-exemplar guidance
            local_attraction = c1 * r1 * (local_best - self.population[i])
            global_attraction = c2 * r2 * (elite - self.population[i])

            self.velocity[i] = (
                inertia_weight * self.velocity[i]
                + local_attraction
                + global_attraction
                + 0.10 * exploration_de
                + 0.05 * exploration_orig
            )

            self.velocity[i] = np.clip(
                self.velocity[i],
                -velocity_limit,
                velocity_limit
            )

            mutation = mutation_strength * self.rng.standard_normal(self.dimension)
            new_position = self.population[i] + self.velocity[i] + mutation

            # Late-stage local refinement around elite
            if progress > 0.75:
                new_position += 0.04 * (elite - new_position)

            # Energy-based stagnation detection & OBL / Lévy escape
            kinetic_energy = np.sum(self.velocity[i] ** 2)
            if self.stagnation_counters[i] > self.k_stag and kinetic_energy < 1e-3:
                if self.rng.random() < 0.5:
                    new_position = self.lower_bound + self.upper_bound - self.population[i]
                else:
                    levy_step = 0.02 * domain_range * (1 - progress) * self._levy_flight()
                    new_position += levy_step

            new_position = np.clip(
                new_position,
                self.lower_bound,
                self.upper_bound
            )

            new_fitness = self.objective_function(new_position)
            self.evaluations_count += 1

            if new_fitness < self.fitness_values[i]:
                self.population[i] = new_position
                self.fitness_values[i] = new_fitness
                self.stagnation_counters[i] = 0
            else:
                self.velocity[i] = 0.5 * self.velocity[i]
                self.stagnation_counters[i] += 1

        self._update_elite()

        # Record population diversity (mean pairwise Euclidean distance)
        pairwise_dists = []
        for a in range(self.population_size):
            for b in range(a + 1, self.population_size):
                pairwise_dists.append(np.linalg.norm(self.population[a] - self.population[b]))
        self.diversity_curve.append(np.mean(pairwise_dists) if pairwise_dists else 0.0)

    def optimize(self):
        self._update_elite()

        for iteration in range(self.max_iterations):
            if self.evaluations_limit is not None and self.evaluations_count >= self.evaluations_limit:
                break
            self.update_population(iteration)
            self.convergence_curve.append(self.best_score)

        return {
            "best_score": self.best_score,
            "best_solution": self.best_solution,
            "convergence_curve": self.convergence_curve,
            "diversity_curve": self.diversity_curve,
            "evaluations_count": self.evaluations_count
        }