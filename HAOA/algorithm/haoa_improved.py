import numpy as np
import math


class HAOAImproved:

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
        alpha=0.5,
        k_stag=15,
        gamma=0.01,
        c_min=0.5,
        c_max=2.0,
        beta=1.5
    ):

        self.objective_function = objective_function
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.max_iterations = max_iterations

        self.alpha = alpha
        self.k_stag = k_stag
        self.gamma = gamma
        self.c_min = c_min
        self.c_max = c_max
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

        self.elite_index = None

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

    def _compute_coefficients(self, iteration):
        progress = iteration / self.max_iterations
        inertia_weight = 0.95 - 0.6 * progress
        adaptive_learning = 1.8 * (1 - progress)
        c_e = self.c_min + (self.c_max - self.c_min) * progress
        mutation_strength = 0.003 * (1 - progress)
        return progress, inertia_weight, adaptive_learning, c_e, mutation_strength

    def _update_elite(self):
        best_idx = int(np.argmin(self.fitness_values))
        if self.fitness_values[best_idx] < self.best_score:
            self.best_score = self.fitness_values[best_idx]
            self.best_solution = self.population[best_idx].copy()
            self.elite_index = best_idx

    def update_population(self, iteration):
        progress, inertia_weight, adaptive_learning, c_e, mutation_strength = (
            self._compute_coefficients(iteration)
        )

        elite = self.best_solution.copy()
        velocity_limit = 0.2 * (self.upper_bound - self.lower_bound)
        domain_range = self.upper_bound - self.lower_bound
        levy_scale = self.gamma * domain_range * (1 - progress)

        for i in range(self.population_size):
            if self.evaluations_limit is not None and self.evaluations_count >= self.evaluations_limit:
                break

            r1 = self.rng.random(self.dimension)
            r2 = self.rng.random(self.dimension)

            r1_idx = int(self.rng.integers(0, self.population_size))
            r2_idx = int(self.rng.integers(0, self.population_size))
            while r2_idx == i:
                r2_idx = int(self.rng.integers(0, self.population_size))

            exploration = adaptive_learning * r2 * (
                self.population[r1_idx] - self.population[r2_idx]
            )

            exploitation = c_e * r1 * (elite - self.population[i])

            self.velocity[i] = (
                inertia_weight * self.velocity[i]
                + exploitation
                + 0.15 * exploration
            )

            self.velocity[i] = np.clip(
                self.velocity[i],
                -velocity_limit,
                velocity_limit
            )

            mutation = mutation_strength * self.rng.standard_normal(self.dimension)

            new_position = (
                self.population[i]
                + self.velocity[i]
                + mutation
            )

            if progress > 0.75:
                local_refinement = 0.03 * (elite - new_position)
                new_position += local_refinement

            if self.stagnation_counters[i] > self.k_stag and i != self.elite_index:
                levy_step = levy_scale * self._levy_flight()
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
                self.velocity[i] = self.alpha * self.velocity[i]
                self.stagnation_counters[i] += 1

        self._update_elite()

    def optimize(self):
        print("\nRunning Improved HAOA...\n")

        self._update_elite()

        for iteration in range(self.max_iterations):
            self.update_population(iteration)
            self.convergence_curve.append(self.best_score)

            if (iteration + 1) % 50 == 0 or iteration == 0:
                print(
                    f"Iteration "
                    f"{iteration + 1}/"
                    f"{self.max_iterations}"
                    f" | Best Score: "
                    f"{self.best_score:.10e}"
                )

        return {
            "best_score": self.best_score,
            "best_solution": self.best_solution,
            "convergence_curve": self.convergence_curve,
            "evaluations_count": self.evaluations_count
        }