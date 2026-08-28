from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import math
from typing import Callable


@dataclass(slots=True)
class CmaesState:
    mean: NDArray
    sigma: float
    C: NDArray
    B: NDArray
    D: NDArray
    C_1_2_inv: NDArray
    p_sigma: NDArray
    p_c: NDArray
    generation: int


@dataclass(slots=True)
class CmaesStrategyParameters:
    population_size: int
    mu: int
    mu_eff: float
    weights: NDArray
    c_m: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c_1: float
    c_mu: float


@dataclass(slots=True)
class CmaesConstants:
    dimension: int
    chi_n: float


def default_strategy_parameters(population_size: int, dimension: int) -> CmaesStrategyParameters:
    mu = population_size // 2
    weights = [(math.log((population_size + 1) / 2) - math.log(i)) for i in range(1, population_size + 1)]
    weights = np.array(weights)
    mu_eff = np.sum(weights[:mu]) ** 2 / np.sum(weights[:mu] ** 2)

    c_m = 1

    # Step Size Constrol
    c_sigma = (mu_eff + 2) / (dimension + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dimension + 1)) - 1) + c_sigma

    # Covariance Matrix Adaptation
    alpha_cov = 2
    c_c = (4 + mu_eff / dimension) / (dimension + 4 + 2 * mu_eff / dimension)
    c_1 = alpha_cov / ((dimension + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, alpha_cov * (1 / 4 + mu_eff + 1 / mu_eff - 2) / ((dimension + 2) ** 2 + alpha_cov * mu_eff / 2))

    mu_eff_neg = np.sum(weights[mu:]) ** 2 / np.sum(weights[mu:] ** 2)
    alpha_mu = 1 + c_1 / c_mu
    alpha_mu_eff = 1 + 2 * mu_eff_neg / (mu_eff + 2)
    alpha_pos_def = (1 - c_1 - c_mu) / (dimension * c_mu)

    sum_p = np.sum(weights[:mu])
    sum_n = abs(np.sum(weights[mu:]))

    def actual_weights(weight):
        if weight >= 0:
            return weight / sum_p
        else:
            return weight * min(alpha_mu, alpha_mu_eff, alpha_pos_def) / sum_n

    calculate_weights = np.vectorize(actual_weights)
    weights = calculate_weights(weights)

    return CmaesStrategyParameters(
        population_size=population_size,
        mu=mu,
        mu_eff=mu_eff,
        weights=weights,
        c_m=c_m,
        c_sigma=c_sigma,
        d_sigma=d_sigma,
        c_c=c_c,
        c_1=c_1,
        c_mu=c_mu
    )


def cma_es(objective_function: callable,
           generation_callback: Callable | None,
           bounds: list[tuple[float | int | None, float | int | None]],
           initial_mean: NDArray,
           initial_step_size: float,
           population_size: int | None,
           max_iterations: int = 1000,
           seed: int | None = None,
           penalty_factor_bounds=1e6,
           scaling: NDArray | None = None,
           return_mean: bool = False):
    history = {
        "best_fitness": [],
        "median_fitness": [],
        "sigma": [],
        "condition": [],
        "mean": [],
        "best_fitness_solution": []
    }

    dimension = len(initial_mean)
    chi_n = np.sqrt(dimension) * (1 - 1 / (4 * dimension) + 1 / (21 * dimension ** 2))

    if not population_size:
        population_size = math.ceil(4 + 3 * math.log(dimension))
    cmaes_parameters = default_strategy_parameters(population_size, dimension)
    cmaes_state = CmaesState(
        mean=initial_mean,
        sigma=initial_step_size,
        C=np.eye(dimension),
        B=np.eye(dimension),
        D=np.ones(dimension),
        C_1_2_inv=np.eye(dimension),
        p_sigma=np.zeros_like(initial_mean),
        p_c=np.zeros_like(initial_mean),
        generation=0
    )
    random_generator = np.random.default_rng(seed)

    lower_bounds = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
    upper_bounds = np.array([b[1] if b[1] is not None else np.inf for b in bounds])

    if scaling is not None:
        lower_bounds *= scaling
        upper_bounds *= scaling
        temp = lower_bounds.copy()
        lower_bounds = np.minimum(lower_bounds, upper_bounds)
        upper_bounds = np.maximum(temp, upper_bounds)
        cmaes_state.mean = cmaes_state.mean * scaling
        original_objective = objective_function
        objective_function = lambda x: original_objective(x / scaling)
    else:
        scaling = np.ones_like(cmaes_state.mean)

    while True:
        # print(cmaes_state.generation)
        if generation_callback is not None:
            generation_callback()
        z = random_generator.standard_normal([population_size, dimension])
        y = (z * cmaes_state.D) @ cmaes_state.B.T
        x = cmaes_state.mean + y * cmaes_state.sigma

        x_repaired = np.clip(x, lower_bounds, upper_bounds)
        violation = np.sum((x - x_repaired) ** 2, axis=1)

        fitness = np.apply_along_axis(objective_function, 1, x_repaired) + penalty_factor_bounds * violation
        sort_indices = np.argsort(fitness)
        x = x[sort_indices]
        y = y[sort_indices]

        fitness = fitness[sort_indices]

        y_w = np.sum(cmaes_parameters.weights[:cmaes_parameters.mu, None] * y[:cmaes_parameters.mu], axis=0)
        cmaes_state.mean = cmaes_state.mean + cmaes_parameters.c_m * cmaes_state.sigma * y_w

        cmaes_state.p_sigma = (
                (1 - cmaes_parameters.c_sigma) * cmaes_state.p_sigma
                + np.sqrt(cmaes_parameters.c_sigma * (2 - cmaes_parameters.c_sigma) * cmaes_parameters.mu_eff)
                * (cmaes_state.C_1_2_inv @ y_w)
        )
        norm_p_sigma = np.linalg.norm(cmaes_state.p_sigma, 2)
        cmaes_state.sigma = cmaes_state.sigma * np.exp(
            (cmaes_parameters.c_sigma / cmaes_parameters.d_sigma)
            * (norm_p_sigma / chi_n - 1)
        )
        h_sigma = int(
            norm_p_sigma / (np.sqrt(
                1 - (1 - cmaes_parameters.c_sigma)
                ** (2 * (cmaes_state.generation + 1))
            ) * chi_n)
            < (1.4 + 2 / (dimension + 1))
        )
        # rank one
        cmaes_state.p_c = (
                (1 - cmaes_parameters.c_c)
                * cmaes_state.p_c
                + h_sigma
                * np.sqrt(
            cmaes_parameters.c_c
            * (2 - cmaes_parameters.c_c)
            * cmaes_parameters.mu_eff
        ) * y_w)

        # rank mu
        normalized_weights = _modify_weights(cmaes_parameters.weights, cmaes_state.C_1_2_inv, dimension, y)
        rank_mu = np.zeros_like(cmaes_state.C)

        for i in range(population_size):
            rank_mu += (
                    normalized_weights[i]
                    * np.outer(y[i], y[i])
            )

        delta_h_sigma = 1 if (1 - h_sigma) * (2 - cmaes_parameters.c_c) * cmaes_parameters.c_c < 1 else 0
        cmaes_state.C = (
                (1 - cmaes_parameters.c_1 + cmaes_parameters.c_1 * delta_h_sigma
                 - cmaes_parameters.c_mu * np.sum(cmaes_parameters.weights))
                * cmaes_state.C
                + cmaes_parameters.c_1
                * np.outer(cmaes_state.p_c, cmaes_state.p_c)
                + cmaes_parameters.c_mu
                * rank_mu
        )

        # Update State
        cmaes_state.generation += 1

        eigvals, B = np.linalg.eigh(cmaes_state.C)
        eigvals = np.maximum(eigvals, 1e-30)
        cmaes_state.B = B
        cmaes_state.D = np.sqrt(eigvals)
        cmaes_state.C_1_2_inv = cmaes_state.B @ np.diag(1 / cmaes_state.D) @ cmaes_state.B.T

        history["best_fitness"].append(fitness[0])
        history["best_fitness_solution"].append(x[0, :] / scaling)
        history["median_fitness"].append(np.median(fitness))
        history["sigma"].append(cmaes_state.sigma)

        condition = np.max(cmaes_state.D) / np.min(cmaes_state.D)
        history["condition"].append(condition)
        history["mean"].append(cmaes_state.mean / scaling)
        if termination_criteria(cmaes_state, history, max_iterations, initial_step_size, population_size):
            break
    if return_mean:
        best_solution = cmaes_state.mean / scaling
    else:
        best_solution = min(zip(history["best_fitness"], history["best_fitness_solution"]))[1]
    return best_solution, history


def termination_criteria(state, history, max_iterations, initial_sigma, lambda_):
    n = len(state.mean)

    # Maximum iterations
    if state.generation >= max_iterations:
        return True

    # ConditionCov: condition number exceeds 1e14
    if np.max(state.D) / np.min(state.D) > 1e14:
        return True

    # TolXUp: sigma * max(diag(D)) increased > 1e4 times initial sigma
    if state.sigma * np.max(state.D) > 1e4 * initial_sigma:
        return True

    # TolX: standard deviation too small
    tolx = 1e-12 * initial_sigma
    if np.all(state.sigma * state.D < tolx):
        return True

    # EqualFunValues: range of best fitness values is zero
    if len(history["best_fitness"]) > 0:
        window = 10 + int(30 * n / lambda_)
        if len(history["best_fitness"]) >= window:
            recent_best = history["best_fitness"][-window:]
            # Use relative tolerance for numerical stability
            if np.max(recent_best) - np.min(recent_best) < 1e-12 * max(1, abs(np.median(recent_best))):
                return True

    # Stagnation: check if fitness is not improving
    if len(history["best_fitness"]) > 200:
        # Window size: 20% of history, min 120+30n/λ, max 20000
        window = max(120 + int(30 * n / lambda_), int(0.2 * len(history["best_fitness"])))
        window = min(window, 20000, len(history["best_fitness"]))

        if window >= 30:  # Need enough points for meaningful comparison
            segment = history["best_fitness"][-window:]
            split_point = window // 3

            first_third = segment[:split_point]
            last_third = segment[-split_point:]

            # Check if median hasn't improved
            if np.median(last_third) >= np.median(first_third):
                return True

    return False


def _modify_weights(weights, C_1_2_inv, dimension, y):
    new_weights = [weight if weight > 0 else weight * dimension /
                                             np.linalg.norm(C_1_2_inv @ y[i, :], 2) ** 2
                   for i, weight in enumerate(weights)]
    return np.array(new_weights)


def main():
    print(cma_es(lambda x: x[0], [(0, 1)], [0, 1], 1, 20))


if __name__ == '__main__':
    main()
