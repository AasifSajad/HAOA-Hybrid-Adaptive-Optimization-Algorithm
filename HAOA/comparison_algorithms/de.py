import numpy as np


class DifferentialEvolution:

    def __init__(
        self,
        objective_function,
        dimension,
        lower_bound,
        upper_bound,
        population_size,
        max_generations,
        initial_population=None,
        random_seed=None,
        evaluations_limit=None,
        F=0.8,
        CR=0.9
    ):

        self.objective_function = objective_function
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR

        self.rng = np.random.default_rng(random_seed)

        if initial_population is not None:
            self.population = np.array(initial_population, dtype=float).copy()
        else:
            self.population = self.rng.uniform(
                self.lower_bound,
                self.upper_bound,
                (self.population_size, self.dimension)
            )

        self.fitness_values = np.empty(self.population_size)
        for i in range(self.population_size):
            self.fitness_values[i] = self.objective_function(self.population[i])
        self.evaluations_count = self.population_size

        self.best_solution = None
        self.best_score = float("inf")
        self.convergence_curve = []

        self.evaluations_limit = evaluations_limit

    def _mutate_and_crossover(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)

        mutant = (
            self.population[r1]
            + self.F * (self.population[r2] - self.population[r3])
        )

        cross_mask = self.rng.random(self.dimension) < self.CR
        if not np.any(cross_mask):
            cross_mask[self.rng.integers(0, self.dimension)] = True

        trial = np.where(cross_mask, mutant, self.population[target_idx])
        return trial

    def optimize(self):
        print("\nRunning Differential Evolution...\n")

        self._update_best()

        for generation in range(self.max_generations):
            for i in range(self.population_size):
                if self.evaluations_limit is not None and self.evaluations_count >= self.evaluations_limit:
                    break

                trial = self._mutate_and_crossover(i)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                trial_fitness = self.objective_function(trial)
                self.evaluations_count += 1

                if trial_fitness < self.fitness_values[i]:
                    self.population[i] = trial
                    self.fitness_values[i] = trial_fitness

            self._update_best()
            self.convergence_curve.append(self.best_score)

            if (generation + 1) % 50 == 0 or generation == 0:
                print(
                    f"Generation "
                    f"{generation + 1}/"
                    f"{self.max_generations}"
                    f" | Best Score: "
                    f"{self.best_score:.10e}"
                )

        return {
            "best_score": self.best_score,
            "best_solution": self.best_solution,
            "convergence_curve": self.convergence_curve,
            "evaluations_count": self.evaluations_count
        }

    def _update_best(self):
        best_idx = int(np.argmin(self.fitness_values))
        if self.fitness_values[best_idx] < self.best_score:
            self.best_score = self.fitness_values[best_idx]
            self.best_solution = self.population[best_idx].copy()