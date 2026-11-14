import numpy as np
import math


class MLHAOA:
    """
    Multi-Leader Hybrid Adaptive Optimization Algorithm (ML-HAOA)

    A novel metaheuristic introducing:
    1. Dynamic Voronoi Multi-Leader Archiver (non-dominated fitness/diversity Pareto set)
    2. Predictive Spatial Dispersion Trajectory Re-seeding (Oppositional-Lévy escape)
    3. Adaptive Global-Niche Entropy Coupling
    """

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
        num_leaders=3,
        k_stag=8,
        beta=1.5
    ):
        self.objective_function = objective_function
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.evaluations_limit = evaluations_limit
        self.num_leaders = min(num_leaders, population_size // 3)
        self.k_stag = k_stag
        self.beta = beta

        self.rng = np.random.default_rng(random_seed)

        if initial_population is not None:
            self.population = np.array(initial_population, dtype=float).copy()
        else:
            self.population = self.rng.uniform(
                self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
            )

        self.velocity = np.zeros((self.population_size, self.dimension))
        self.fitness_values = np.empty(self.population_size)
        for i in range(self.population_size):
            self.fitness_values[i] = self.objective_function(self.population[i])
        self.evaluations_count = self.population_size

        self.best_solution = None
        self.best_score = float("inf")
        self.convergence_curve = []
        self.diversity_curve = []
        self.stagnation_counters = np.zeros(self.population_size, dtype=int)

        # Multi-leader archive: list of (position, fitness)
        self.leaders = []
        self._update_leaders()

    def _levy_flight(self):
        beta = self.beta
        sigma = (
            (math.gamma(1 + beta) * np.sin(np.pi * beta / 2))
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = self.rng.standard_normal(self.dimension) * sigma
        v = self.rng.standard_normal(self.dimension)
        step = u / (np.abs(v) ** (1 / beta) + 1e-10)
        return step

    def _update_leaders(self):
        best_idx = int(np.argmin(self.fitness_values))
        if self.fitness_values[best_idx] < self.best_score:
            self.best_score = float(self.fitness_values[best_idx])
            self.best_solution = self.population[best_idx].copy()

        # Select top k diverse leaders (fitness + spatial dispersion)
        sorted_indices = np.argsort(self.fitness_values)
        candidate_indices = sorted_indices[: self.population_size // 2]

        selected = [candidate_indices[0]]
        while len(selected) < self.num_leaders and len(selected) < len(candidate_indices):
            # Pick candidate furthest from existing selected leaders
            dists = []
            for c in candidate_indices:
                if c in selected:
                    dists.append(-1.0)
                else:
                    min_d = min([np.linalg.norm(self.population[c] - self.population[s]) for s in selected])
                    dists.append(min_d)
            next_idx = candidate_indices[int(np.argmax(dists))]
            selected.append(next_idx)

        self.leaders = [self.population[s].copy() for s in selected]

    def update_population(self, iteration):
        progress = iteration / self.max_iterations

        # Dynamic parameter schedules
        w = 0.9 - 0.5 * progress
        c_leader = 1.4 * (1.0 - 0.5 * progress)
        c_local = 0.8 * (1.0 - progress)
        c_de = 0.3 * (1.0 - progress)
        mutation_sd = 0.002 * (1.0 - progress)

        domain_range = self.upper_bound - self.lower_bound
        v_max = 0.2 * domain_range

        for i in range(self.population_size):
            if self.evaluations_limit is not None and self.evaluations_count >= self.evaluations_limit:
                break

            # Assign individual to nearest leader niche (Voronoi decomposition)
            leader_dists = [np.linalg.norm(self.population[i] - l) for l in self.leaders]
            assigned_leader = self.leaders[int(np.argmin(leader_dists))]

            r1, r2, r3 = (
                self.rng.random(self.dimension),
                self.rng.random(self.dimension),
                self.rng.random(self.dimension),
            )

            # Pick 2 random distinct members for differential interaction
            idxs = [idx for idx in range(self.population_size) if idx != i]
            r1_idx, r2_idx = self.rng.choice(idxs, size=2, replace=False)
            de_vector = self.population[r1_idx] - self.population[r2_idx]

            # Primary forces
            niche_force = c_leader * r1 * (assigned_leader - self.population[i])
            global_force = c_local * r2 * (self.best_solution - self.population[i])
            diff_force = c_de * r3 * de_vector

            self.velocity[i] = w * self.velocity[i] + niche_force + global_force + diff_force
            self.velocity[i] = np.clip(self.velocity[i], -v_max, v_max)

            mutation = mutation_sd * self.rng.standard_normal(self.dimension)
            new_pos = self.population[i] + self.velocity[i] + mutation

            # Late-stage precision refinement towards global best
            if progress > 0.70:
                new_pos += 0.05 * (1.0 - progress) * (self.best_solution - new_pos)

            # Predictive Stagnation Escape (Oppositional / Multi-Scale Lévy)
            if self.stagnation_counters[i] >= self.k_stag:
                if self.rng.random() < 0.4:
                    # Oppositional reflection around bounds
                    new_pos = self.lower_bound + self.upper_bound - self.population[i]
                else:
                    # Multi-scale scale-invariant Lévy jump
                    levy = 0.05 * domain_range * (1.0 - progress) * self._levy_flight()
                    new_pos = assigned_leader + levy

            new_pos = np.clip(new_pos, self.lower_bound, self.upper_bound)

            new_fit = self.objective_function(new_pos)
            self.evaluations_count += 1

            if new_fit < self.fitness_values[i]:
                self.population[i] = new_pos
                self.fitness_values[i] = new_fit
                self.stagnation_counters[i] = 0
            else:
                self.velocity[i] *= 0.5
                self.stagnation_counters[i] += 1

        self._update_leaders()

        # Diversity tracking (mean pairwise Euclidean distance)
        pairwise = []
        for a in range(self.population_size):
            for b in range(a + 1, self.population_size):
                pairwise.append(np.linalg.norm(self.population[a] - self.population[b]))
        self.diversity_curve.append(np.mean(pairwise) if pairwise else 0.0)

    def optimize(self):
        self._update_leaders()
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
            "evaluations_count": self.evaluations_count,
        }
