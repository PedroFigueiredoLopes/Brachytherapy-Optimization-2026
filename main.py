from brachytherapy_optimization import BrachytherapyOptimization, visualize_state
from distance_field_precomputation import SignedDistanceField, load_sdf
from black_hole import black_hole_algorithm
import numpy as np
import cma_es
import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle
import io
import math
from PIL import Image


# ── Top-level configuration ────────────────────────────────────────────────────

ALGORITHM        = "cma_es"   # "cma_es" or "black_hole"
N_STARTS         = 8
SOURCE_COUNT     = 6
MAX_ITERATIONS   = 10000
PRESCRIBED_DOSE  = 5
N_SAMPLES        = 2 ** 14
EXCLUSION_RADIUS = 1e-3
RESULTS_PATH     = "results.pkl"

# CMA-ES specific
CMA_POPULATION_SIZE = None   # None → use CMA-ES default: ceil(4 + 3*ln(dim))

# Black Hole specific
BH_N_STARS              = 50
BH_EVENT_HORIZON_FACTOR = 0.003
BH_TOLERANCE            = 1e-10
BH_PATIENCE             = 300


# ── SDF helpers ────────────────────────────────────────────────────────────────

def build_sdfs():
    whole_prostate = SignedDistanceField(*load_sdf("prostate_sdf.npz"))
    urethra = SignedDistanceField(*load_sdf("urethra_sdf.npz"))
    assert whole_prostate.metadata == urethra.metadata
    prostate = SignedDistanceField(
        np.maximum(whole_prostate.sdf, -urethra.sdf),
        whole_prostate.metadata,
    )
    return prostate, urethra, whole_prostate


def build_problem(prostate, urethra, whole_prostate, n_samples=N_SAMPLES):
    return BrachytherapyOptimization(
        prostate_sdf=prostate,
        urethra_sdf=urethra,
        whole_prostate_sdf=whole_prostate,
        n_samples=n_samples,
        exclusion_radius=EXCLUSION_RADIUS,
        prescribed_dose=PRESCRIBED_DOSE,
    )


def make_bounds_and_scaling(problem):
    bounds = (
            [(problem.bounds[0][0], problem.bounds[0][1]),
             (problem.bounds[1][0], problem.bounds[1][1]),
             (0, 900)]
            * SOURCE_COUNT
    )
    scaling = np.array([[1, 1, 1 / 500] for _ in range(SOURCE_COUNT)]).flatten()
    return bounds, scaling


def random_init(rng, bounds):
    """Circular ring with randomised radius, angle offset, and z."""
    r = rng.uniform(10e-3, 20e-3)
    phi0 = rng.uniform(0, 2 * np.pi)
    z0 = rng.uniform(15, 25)
    theta = np.linspace(phi0, phi0 + 2 * np.pi, SOURCE_COUNT, endpoint=False)
    x_init = np.column_stack((
        r * np.cos(theta),
        r * np.sin(theta),
        z0 * np.ones(SOURCE_COUNT),
    ))
    return x_init.flatten()


def clip_and_reshape(result, bounds):
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    return np.clip(result, lower, upper).reshape(-1, 3)


# ── Workers ────────────────────────────────────────────────────────────────────

def run_cma_es(args):
    run_id, seed = args
    prostate, urethra, whole_prostate = build_sdfs()
    problem = build_problem(prostate, urethra, whole_prostate)
    rng = np.random.default_rng(seed)
    bounds, scaling = make_bounds_and_scaling(problem)

    x_init = random_init(rng, bounds)
    step = rng.uniform(0.003, 0.008)

    try:
        result, history = cma_es.cma_es(
            objective_function=problem.evaluate_penalized,
            generation_callback=problem.update_integration_points,
            bounds=bounds,
            scaling=scaling,
            initial_mean=x_init,
            initial_step_size=step,
            population_size=CMA_POPULATION_SIZE,
            max_iterations=MAX_ITERATIONS,
            seed=int(seed),
            penalty_factor_bounds=1e6,
            return_mean=True,
        )
        result = clip_and_reshape(result, bounds)
        penalized = problem.evaluate_penalized(result)
        obj, _ = problem.evaluate(result)

        dim = SOURCE_COUNT * 3
        pop_size = CMA_POPULATION_SIZE if CMA_POPULATION_SIZE is not None else math.ceil(4 + 3 * math.log(dim))
        n_gen = len(history["best_fitness"])
        fcalls = [g * pop_size for g in range(1, n_gen + 1)]

        print(f"[CMA-ES  start {run_id:02d}]  penalized = {penalized:.6f}  objective = {obj:.6f}")
        return result, penalized, obj, history["best_fitness"], fcalls, history["mean"]

    except Exception as exc:
        print(f"[CMA-ES  start {run_id:02d}]  FAILED — {exc}")
        return None, np.inf, np.inf, [], [], []


def run_black_hole(args):
    run_id, seed = args
    prostate, urethra, whole_prostate = build_sdfs()
    problem = build_problem(prostate, urethra, whole_prostate)
    rng = np.random.default_rng(seed)
    bounds = (
            [(problem.bounds[0][0], problem.bounds[0][1]),
             (problem.bounds[1][0], problem.bounds[1][1]),
             (0, 30)]
            * SOURCE_COUNT
    )
    x_init = random_init(rng, bounds)  # used only to perturb; BH initialises its own population
    scaling = np.array([[1, 1, 1 / 500] for _ in range(SOURCE_COUNT)]).flatten()
    try:
        best, best_val, fit_hist, fcall_hist, _, _ = black_hole_algorithm(
            objective_function=problem.evaluate_penalized,
            bounds=bounds,
            n_stars=BH_N_STARS,
            max_iter=MAX_ITERATIONS,
            event_horizon_factor=BH_EVENT_HORIZON_FACTOR,
            tolerance=BH_TOLERANCE,
            patience=BH_PATIENCE,
            seed=int(seed),
            scaling = scaling
        )
        result = clip_and_reshape(best, bounds)
        penalized = problem.evaluate_penalized(result)
        obj, _ = problem.evaluate(result)

        print(f"[BH      start {run_id:02d}]  penalized = {penalized:.6f}  objective = {obj:.6f}")
        return result, penalized, obj, fit_hist, fcall_hist, []

    except Exception as exc:
        print(f"[BH      start {run_id:02d}]  FAILED — {exc}")
        return None, np.inf, np.inf, [], [], []


WORKERS = {
    "cma_es": run_cma_es,
    "black_hole": run_black_hole,
}


# ── Convergence plots ──────────────────────────────────────────────────────────

def plot_convergence_all_runs(
        runs,  # list of (label, fit_hist, fcall_hist)
        title="CMA-ES convergence — all runs",
        x_axis="generations",  # "generations" or "function_calls"
):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")

    for i, (label, fit_hist, fcall_hist) in enumerate(runs):
        y = np.array(fit_hist, dtype=float)
        x = np.array(fcall_hist if x_axis == "function_calls"
                     else range(1, len(y) + 1), dtype=float)
        ax.plot(x, y, linewidth=1.5, color=cmap(i % 10), label=label, alpha=0.85)

    ax.set_xscale("log")
    if all(np.all(np.array(r[1], dtype=float) > 0) for r in runs):
        ax.set_yscale("log")

    ax.set_xlabel("Function evaluations" if x_axis == "function_calls" else "Generation")
    ax.set_ylabel("Best objective value")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both")
    fig.tight_layout()
    plt.show()


# ── Animation ─────────────────────────────────────────────────────────────────

def create_mean_trajectory_animation(
        problem, history,
        output_path="mean_trajectory.gif",
        fps=60, step=1, figsize=(10, 8), dpi=200,
):
    means = history["mean"]
    fitnesses = history["best_fitness"]
    if not means:
        print("No mean trajectory available (Black Hole algorithm does not record means).")
        return

    n_gen = len(means)
    indices = list(range(0, n_gen, step))
    n_sources = len(means[0]) // 3
    x_min, x_max = problem.bounds[0]
    y_min, y_max = problem.bounds[1]

    frames = []
    for i, gen_idx in enumerate(indices):
        fig, ax = plt.subplots(figsize=figsize)
        state = np.array(means[gen_idx]).reshape(n_sources, 3)
        visualize_state(problem, state, ax=ax, dose_vmin=0, dose_vmax=50,
                        dwell_vmin=10, dwell_vmax=20)
        ax.set_xlim(x_min * 100, x_max * 100)
        ax.set_ylim(y_min * 100, y_max * 100)
        ax.set_title(
            f"Generation {gen_idx + 1}/{n_gen}  —  "
            f"best fitness: {fitnesses[gen_idx]:.4f}"
        )
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
        if i % 20 == 0:
            print(f"  Rendered {i}/{len(indices)} frames")

    frames[0].save(
        output_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0,
    )
    print(f"Saved '{output_path}'  ({len(indices)} frames, {fps} fps)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if ALGORITHM not in WORKERS:
        raise ValueError(f"Unknown ALGORITHM '{ALGORITHM}'. Choose from: {list(WORKERS)}")

    print(f"\nAlgorithm : {ALGORITHM}")
    print(f"Starts    : {N_STARTS}")
    print(f"Max iters : {MAX_ITERATIONS}\n")

    master_rng = np.random.default_rng(42)
    seeds = master_rng.integers(0, 2 ** 31 - 1, size=N_STARTS)
    args = [(i, int(s)) for i, s in enumerate(seeds)]

    worker = WORKERS[ALGORITHM]
    with mp.Pool(processes=N_STARTS) as pool:
        outcomes = pool.map(worker, args)

    # outcomes: list of (result, penalized, obj, fit_hist, fcall_hist, mean_hist)
    valid = [(res, pen, obj, fh, fc, mh) for res, pen, obj, fh, fc, mh in outcomes if res is not None]
    if not valid:
        raise RuntimeError("All multistart runs failed.")

    valid.sort(key=lambda x: x[1])  # sort by penalized

    print("\n=== Multistart leaderboard (sorted by penalized objective) ===")
    for rank, (_, pen, obj, _, _, _) in enumerate(valid):
        tag = "  ← best" if rank == 0 else ""
        print(f"  #{rank + 1}  penalized = {pen:.6f}  objective = {obj:.6f}{tag}")

    best_result, best_pen, best_obj, best_fit_hist, best_fcall_hist, best_mean_hist = valid[0]
    print(f"\nBest source positions:\n{best_result}")

    # Build a history dict compatible with animation helper
    best_history = {"best_fitness": best_fit_hist, "mean": best_mean_hist}

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({
            "algorithm": ALGORITHM,
            "result": best_result,
            "history": best_history,
            "fit_hist": best_fit_hist,
            "fcall_hist": best_fcall_hist,
            "all_outcomes": [(r, pen, obj, fh, fc, mh) for r, pen, obj, fh, fc, mh in valid],
            "params": {
                "n_starts": N_STARTS,
                "source_count": SOURCE_COUNT,
                "max_iterations": MAX_ITERATIONS,
                "prescribed_dose": PRESCRIBED_DOSE,
                "n_samples": N_SAMPLES,
                "exclusion_radius": EXCLUSION_RADIUS,
                "cma_population_size": CMA_POPULATION_SIZE,
                "bh_n_stars": BH_N_STARS,
                "bh_event_horizon_factor": BH_EVENT_HORIZON_FACTOR,
                "bh_tolerance": BH_TOLERANCE,
                "bh_patience": BH_PATIENCE,
            },
        }, f)
    print(f"Results saved to '{RESULTS_PATH}'")

    # ── Convergence plot for all runs ──────────────────────────────────────────
    run_labels = [
        (f"run #{r + 1}  pen={pen:.4f}  obj={obj:.4f}", fh, fc)
        for r, (_, pen, obj, fh, fc, _) in enumerate(valid)
    ]

    plot_convergence_all_runs(run_labels,
                              title=f"{ALGORITHM} — penalized objective, all runs",
                              x_axis="generations")

    # Best run only by function calls
    _, best_pen, _, best_fh, best_fc, _ = valid[0]
    plot_convergence_all_runs(
        [(f"best run  pen={best_pen:.4f}", best_fh, best_fc)],
        title=f"{ALGORITHM} — penalized objective, best run (function calls)",
        x_axis="function_calls",
    )
    # ── Final state visualisation ──────────────────────────────────────────────
    prostate, urethra, whole_prostate = build_sdfs()
    problem = build_problem(prostate, urethra, whole_prostate)

    fig = visualize_state(problem, best_result)
    plt.show()

    create_mean_trajectory_animation(problem, best_history)


# ── Load & inspect saved results ──────────────────────────────────────────────

def load_results(path=RESULTS_PATH):
    with open(path, "rb") as f:
        data = pickle.load(f)

    result = data["result"]
    fit_hist = data["fit_hist"]
    fcall_hist = data["fcall_hist"]
    all_outcomes = data["all_outcomes"]

    print(f"Algorithm : {data.get('algorithm', 'unknown')}")
    print(f"Runs saved: {len(all_outcomes)}")

    params = data.get("params", {})
    if params:
        print("Parameters:")
        for k, v in params.items():
            print(f"  {k} = {v}")

    best_pen = min(pen for _, pen, _, _, _, _ in all_outcomes)
    best_obj = next(obj for _, pen, obj, _, _, _ in all_outcomes if pen == best_pen)
    print(f"Best penalized: {best_pen:.6f}  objective: {best_obj:.6f}")
    print(f"Best positions:\n{result}")

    prostate, urethra, whole_prostate = build_sdfs()
    problem = build_problem(prostate, urethra, whole_prostate)

    # All runs by iteration
    run_labels_iter = [
        (f"run #{r + 1}  pen={pen:.4f}", fh, fc)
        for r, (_, pen, obj, fh, fc, _) in enumerate(all_outcomes)
    ]
    plot_convergence_all_runs(run_labels_iter,
                              title=f"Convergence — penalized objective, all runs",
                              x_axis="generations")

    # Best run by function calls
    _, best_pen, _, best_fh, best_fc, _ = all_outcomes[0]
    plot_convergence_all_runs(
        [(f"best run  pen={best_pen:.4f}", best_fh, best_fc)],
        title=f"Convergence — penalized objective, best run (function calls)",
        x_axis="function_calls",
    )

    fig = visualize_state(problem, result)
    plt.show()

    return data


# ── Legacy loaders ─────────────────────────────────────────────────────────────
# These handle pickle files produced by older versions of this script.
# Do not use for new runs.

def load_results_legacy_v1(path=RESULTS_PATH):
    """
    LEGACY — for pickles with keys: 'result', 'history', 'all_outcomes'
    where all_outcomes is a list of (result, value) tuples (no fit_hist/fcall_hist)
    and value is the raw unpenalized objective.
    Re-evaluates every result with the penalized objective and re-ranks.
    """
    import warnings
    warnings.warn(
        "load_results_legacy_v1: loading an old-format pickle (v1). "
        "Re-ranking by penalized objective — original ranking may have been by unpenalized value.",
        UserWarning, stacklevel=2,
    )

    with open(path, "rb") as f:
        data = pickle.load(f)

    all_outcomes = data["all_outcomes"]  # list of (result, value)
    history = data["history"]  # dict with 'best_fitness', possibly 'mean'

    prostate, urethra, whole_prostate = build_sdfs()
    problem = build_problem(prostate, urethra, whole_prostate)

    print("Re-evaluating all runs with penalized objective (legacy v1)...")
    reranked = []
    for result, _old_value in all_outcomes:
        penalized = problem.evaluate_penalized(result)
        obj, constraints = problem.evaluate(result)
        reranked.append((result, penalized, obj, constraints))

    reranked.sort(key=lambda x: x[1])

    print(f"\n=== Re-ranked leaderboard — legacy v1 (penalized) ===")
    for rank, (_, pen, obj, constraints) in enumerate(reranked):
        tag = "  ← best" if rank == 0 else ""
        print(f"  #{rank + 1}  penalized={pen:.6f}  objective={obj:.6f}  constraints={constraints}{tag}")

    best_result, best_pen, best_obj, best_constraints = reranked[0]
    print(f"\nBest source positions:\n{best_result}")

    # Convergence of the single stored history (no per-run histories in v1)
    fit_hist = history.get("best_fitness", [])
    if fit_hist:
        dim = SOURCE_COUNT * 3
        pop_size = CMA_POPULATION_SIZE or math.floor(4 + 3 * math.log(dim))
        fcalls = [g * pop_size for g in range(1, len(fit_hist) + 1)]
        plot_convergence_all_runs(
            [("best run", fit_hist, fcalls)],
            title="Convergence — legacy v1 (best run only)",
            x_axis="function_calls",
        )

    fig = visualize_state(problem, best_result)
    plt.show()

    return best_result, reranked


def load_results_reranked(path=RESULTS_PATH):
    """
    LEGACY — for pickles produced by the intermediate version where all_outcomes
    is a list of (result, value, fit_hist, fcall_hist) and value is the
    unpenalized objective (not penalized). Re-ranks by penalized objective.
    """
    import warnings
    warnings.warn(
        "load_results_reranked: loading a legacy pickle where sorting was by "
        "unpenalized objective. Re-ranking by penalized objective now. "
        "New runs are already sorted correctly — use load_results() instead.",
        UserWarning, stacklevel=2,
    )

    with open(path, "rb") as f:
        data = pickle.load(f)

    all_outcomes = data["all_outcomes"]  # list of (result, value, fit_hist, fcall_hist)

    prostate, urethra, whole_prostate = build_sdfs()
    problem = build_problem(prostate, urethra, whole_prostate)

    print("Re-evaluating all runs with penalized objective (legacy reranked)...")
    reranked = []
    for result, _old_value, fh, fc in all_outcomes:
        penalized = problem.evaluate_penalized(result)
        obj, constraints = problem.evaluate(result)
        reranked.append((result, penalized, obj, constraints, fh, fc))

    reranked.sort(key=lambda x: x[1])

    print(f"\n=== Re-ranked leaderboard (penalized) ===")
    for rank, (_, pen, obj, constraints, _, _) in enumerate(reranked):
        tag = "  ← best" if rank == 0 else ""
        print(f"  #{rank + 1}  penalized={pen:.6f}  objective={obj:.6f}  constraints={constraints}{tag}")

    best_result, best_pen, best_obj, best_constraints, best_fh, best_fc = reranked[0]
    print(f"\nBest source positions:\n{best_result}")

    run_labels = [
        (f"run #{r + 1}  pen={pen:.4f}  obj={obj:.4f}", fh, fc)
        for r, (_, pen, obj, _, fh, fc) in enumerate(reranked)
    ]
    plot_convergence_all_runs(run_labels, title="Convergence — all runs (re-ranked)", x_axis="function_calls")

    fig = visualize_state(problem, best_result)
    plt.show()

    return best_result, reranked

if __name__ == "__main__":
    mp.freeze_support()
    main()

    # To inspect saved results without re-running:
    # load_results()