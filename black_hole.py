import numpy as np
from numpy.typing import NDArray

def black_hole_algorithm(
    objective_function,
    bounds,
    n_stars=150,
    max_iter=2000,
    event_horizon_factor=0.003,
    tolerance=1e-10,
    patience=300,
    seed=None,
    scaling: NDArray | None = None
):
    rng = np.random.default_rng(seed)

    bounds = np.array(bounds, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    dim = len(bounds)

    function_calls = 0

    if scaling is not None:
        lower *= scaling
        upper *= scaling
        temp = lower.copy()
        lower = np.minimum(lower, upper)
        upper = np.maximum(temp, upper)
        original_objective = objective_function
        objective_function = lambda x: original_objective(x / scaling)
    else:
        scaling = np.ones_like(dim)


    # ------------------------------------------------------------
    # Helper to count objective function evaluations
    # ------------------------------------------------------------

    def evaluate(x):
        nonlocal function_calls
        function_calls += 1
        return objective_function(x)

    # ------------------------------------------------------------
    # Initial random population of stars
    # ------------------------------------------------------------



    stars = rng.uniform(lower, upper, size=(n_stars, dim))
    fitness = np.array([evaluate(star) for star in stars])

    history = []
    function_calls_history = []
    positions_history = []
    best_positions_history = []

    best_index = np.argmin(fitness)
    global_best = stars[best_index].copy()
    global_best_fitness = fitness[best_index]

    design_space_size = np.linalg.norm(upper - lower)

    no_improvement_counter = 0
    previous_best = global_best_fitness


    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------

    for iteration in range(max_iter):

        positions_history.append(stars.copy()/scaling)

        best_index = np.argmin(fitness)

        if fitness[best_index] < global_best_fitness:
            global_best = stars[best_index].copy()
            global_best_fitness = fitness[best_index]

        black_hole = global_best.copy()

        history.append(global_best_fitness)
        function_calls_history.append(function_calls)
        best_positions_history.append(global_best.copy() / scaling)

        event_horizon_radius = (
            event_horizon_factor
            * design_space_size
            * (1 - iteration / max_iter)
        )

        # --------------------------------------------------------
        # Move stars toward the black hole
        # --------------------------------------------------------

        for i in range(n_stars):

            r = rng.random()
            candidate = stars[i] + r * (black_hole - stars[i])
            candidate_repaired = np.clip(candidate, lower, upper)

            violation = np.sum((candidate - candidate_repaired) ** 2)
            candidate_fitness = evaluate(candidate_repaired) + 1e6 * violation

            if candidate_fitness < fitness[i]:
                stars[i] = candidate
                fitness[i] = candidate_fitness

                if candidate_fitness < global_best_fitness:
                    global_best = candidate.copy()
                    global_best_fitness = candidate_fitness
                    black_hole = global_best.copy()

            distance = np.linalg.norm(stars[i] - global_best)

            if distance < event_horizon_radius and fitness[i] > global_best_fitness:
                stars[i] = rng.uniform(lower, upper, size=dim)
                fitness[i] = evaluate(stars[i])

        # --------------------------------------------------------
        # Local search around the black hole
        # --------------------------------------------------------

        sigma = (upper - lower) * 0.2 * (1 - iteration / max_iter) + 1e-8

        local_candidate = global_best + rng.normal(0, sigma)
        local_candidate = np.clip(local_candidate, lower, upper)

        local_candidate_fitness = evaluate(local_candidate)

        if local_candidate_fitness < global_best_fitness:
            global_best = local_candidate.copy()
            global_best_fitness = local_candidate_fitness

        # Store updated best value after movement/local search
        history[-1] = global_best_fitness
        function_calls_history[-1] = function_calls
        best_positions_history[-1] = global_best.copy() / scaling

        # --------------------------------------------------------
        # Early stopping
        # --------------------------------------------------------

        if abs(previous_best - global_best_fitness) < tolerance:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0

        previous_best = global_best_fitness

        if no_improvement_counter >= patience:
            print(f"Black Hole stopped early at iteration {iteration + 1}")
            break

    return (
        global_best / scaling,
        global_best_fitness,
        history,
        function_calls_history,
        positions_history,
        best_positions_history
    )