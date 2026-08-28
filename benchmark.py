"""
benchmark.py — Brachytherapy Integration Benchmark Suite
=========================================================

Compares numerical integration methods for TG-43-style dose integrals
inside optimisation loops.  Methods are registered via a decorator and
run through a common harness that records errors, variance, confidence
intervals, convergence slopes, and wall-clock runtimes.

By default, a single high-N Sobol reference is used as ground truth.
Pass ``--self-reference`` to compare each method against its own
high-N run (useful for diagnosing systematic bias).

Usage
-----
    # Run with defaults (all methods, 20 repeats):
    python benchmark.py

    # Fewer repeats, save results to disk:
    python benchmark.py --runs 10 --output results/run_01

    # Only specific methods:
    python benchmark.py --methods mc sobol grid

    # Skip the plots (CI / headless):
    python benchmark.py --no-plot

    # Quick smoke-test:
    python benchmark.py --runs 2 --min-power 4 --max-power 10

Output
------
    <output_dir>/
        results.csv          — per-(method, N) summary
        results.json         — full structured results
        convergence.png      — error vs N (log-log)
        variance.png         — std dev vs N (log-log)
        runtime.png          — wall-time vs N (log-log)
        efficiency.png       — error vs runtime (Pareto front)
        report.txt           — human-readable summary table
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import qmc

import tg_43  # project-local TG-43 dose calculator


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """All tunable parameters in one place."""

    # Domain
    size: float = 0.04                  # [m]  side length of the square domain
    exclusion_radius: float = 0.001     # [m]  minimum distance from a source

    # Source geometry
    source_count: int = 6
    source_orbit_radius: float = 0.23   # fraction of size

    # Importance-sampling parameters
    is_global_fraction: float = 0.30    # fraction of samples drawn globally
    is_sigma: float = 0.003             # std dev of per-source Gaussians

    # Reference solution – a single high-N Sobol run (shared ref by default)
    reference_samples: int = 2**26      # ~67M points for the reference
    self_reference: bool = False        # if True, each method uses its own ref

    # Benchmark harness
    benchmark_runs: int = 20            # independent repeats per (method, N)
    confidence_level: float = 0.95      # for CI bands on plots
    min_power: int = 4                  # 2^min_power = smallest N
    max_power: int = 19                 # 2^max_power = largest N

    # Output
    output_dir: Optional[Path] = None
    plot: bool = True
    seed: int = 42


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SingleRun:
    """Result of one call to an integration method."""
    estimate: float
    runtime: float          # seconds
    samples_used: int       # points inside the ROI


@dataclass
class MethodResult:
    """Aggregated results for one method over many runs at one N."""
    method_name: str
    N: int                          # nominal sample size requested
    samples_used: float             # mean samples that fell inside ROI
    mean_estimate: float
    mean_abs_error: float
    std_estimate: float
    ci_low: float                   # CI on the mean estimate
    ci_high: float
    mean_runtime: float             # seconds
    std_runtime: float
    convergence_slope: float = 0.0  # filled in later by the runner
    efficiency: float = 0.0         # mean_abs_error * mean_runtime (lower=better)


# ──────────────────────────────────────────────────────────────────────────────
# Method registry
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    """Decorator: register an integration function under a short name."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def available_methods() -> List[str]:
    return list(_REGISTRY.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def generate_sources(cfg: BenchmarkConfig) -> np.ndarray:
    """Return (source_count, 2) array of source positions [m]."""
    theta = np.linspace(0, 2 * math.pi, cfg.source_count, endpoint=False)
    r = cfg.source_orbit_radius * cfg.size
    cx = cy = cfg.size / 2
    return np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))


def roi_mask(x: np.ndarray, y: np.ndarray, cfg: BenchmarkConfig) -> np.ndarray:
    """Boolean mask: True where (x, y) lies inside the ellipsoidal ROI."""
    h = cfg.size / 2
    return (
        np.abs((x - h) / 3.5) ** 2.4
        + np.abs((y - h) / 3.5) ** 2.3
    ) < 3.5e-6


def estimate_roi_area(cfg: BenchmarkConfig, n: int = 200_000) -> float:
    """
    Monte Carlo estimate of the **effective** ROI area [m²].

    The effective ROI is the ellipsoidal region MINUS the circular
    exclusion zones around each source.  Points inside the ROI but
    within *exclusion_radius* of any source are NOT counted, because
    the dose integral excludes them to avoid the 1/r singularity.
    """
    rng = np.random.default_rng()
    x = rng.uniform(0, cfg.size, n)
    y = rng.uniform(0, cfg.size, n)

    in_roi = roi_mask(x, y, cfg)

    valid = np.ones(n, dtype=bool)
    sources = generate_sources(cfg)
    for sx, sy in sources:
        r = np.hypot(x - sx, y - sy)
        valid &= (r > cfg.exclusion_radius)

    effective_roi = in_roi & valid
    return float(np.mean(effective_roi) * cfg.size ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# Dose evaluation
# ──────────────────────────────────────────────────────────────────────────────

def compute_total_dose(
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    dose_fn: Callable,
    exclusion_radius: float,
) -> np.ndarray:
    """
    Sum TG-43 dose contributions from every source.

    Points closer than *exclusion_radius* to any source are excluded
    (set to NaN) to avoid the 1/r singularity.

    Accepts both 1-D arrays (MC/Sobol/IS methods) and 2-D meshgrid arrays
    (grid method).
    """
    dose = np.zeros_like(x, dtype=float)
    valid = np.ones_like(x, dtype=bool)

    for sx, sy in sources:
        r = np.hypot(x - sx, y - sy)
        local_valid = r > exclusion_radius
        valid &= local_valid

        tmp = np.zeros_like(r)
        tmp[local_valid] = dose_fn(r[local_valid])
        dose += tmp

    dose[~valid] = np.nan
    return dose


# ──────────────────────────────────────────────────────────────────────────────
# Importance-sampling helpers
# ──────────────────────────────────────────────────────────────────────────────

def _importance_pdf(
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    cfg: BenchmarkConfig,
) -> np.ndarray:
    """Mixture PDF: uniform component + Gaussian peaks around each source."""
    alpha = cfg.is_global_fraction
    sigma = cfg.is_sigma
    uniform_part = alpha / cfg.size ** 2

    gauss_part = np.zeros_like(x, dtype=float)
    for sx, sy in sources:
        r2 = (x - sx) ** 2 + (y - sy) ** 2
        gauss_part += np.exp(-r2 / (2 * sigma ** 2))
    gauss_part *= (1 - alpha) / (len(sources) * 2 * math.pi * sigma ** 2)

    return uniform_part + gauss_part


def _sample_importance(
    n: int,
    sources: np.ndarray,
    cfg: BenchmarkConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw *n* i.i.d. samples from the importance mixture distribution."""
    alpha = cfg.is_global_fraction
    sigma = cfg.is_sigma
    S = len(sources)
    size = cfg.size

    comp = rng.choice(S + 1, size=n, p=[alpha] + [(1 - alpha) / S] * S)

    x = np.empty(n)
    y = np.empty(n)

    mask_unif = (comp == 0)
    n_unif = int(mask_unif.sum())
    x[mask_unif] = rng.uniform(0, size, n_unif)
    y[mask_unif] = rng.uniform(0, size, n_unif)

    for s_idx in range(S):
        mask_s = (comp == s_idx + 1)
        n_s = int(mask_s.sum())
        if n_s == 0:
            continue
        sx, sy = sources[s_idx]
        x[mask_s] = rng.normal(sx, sigma, n_s)
        y[mask_s] = rng.normal(sy, sigma, n_s)

    x = np.clip(x, 0.0, size)
    y = np.clip(y, 0.0, size)
    return x, y


def _sample_importance_qmc(
    n: int,
    sources: np.ndarray,
    cfg: BenchmarkConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw *n* low-discrepancy samples from the importance mixture."""
    alpha = cfg.is_global_fraction
    sigma = cfg.is_sigma
    S = len(sources)
    size = cfg.size

    sampler = qmc.Sobol(d=3, scramble=True)
    pts = sampler.random(n)
    u1, u2, u3 = pts[:, 0], pts[:, 1], pts[:, 2]

    x = np.empty(n)
    y = np.empty(n)

    mask_global = u3 < alpha
    x[mask_global] = u1[mask_global] * size
    y[mask_global] = u2[mask_global] * size

    mask_local = ~mask_global
    u3_local = (u3[mask_local] - alpha) / (1.0 - alpha)
    src_idx = np.floor(u3_local * S).astype(int)
    sx_arr = sources[src_idx, 0]
    sy_arr = sources[src_idx, 1]

    u1_loc = u1[mask_local]
    u2_loc = u2[mask_local]
    r = sigma * np.sqrt(-2.0 * np.log(np.maximum(u1_loc, 1e-16)))
    th = 2.0 * math.pi * u2_loc

    x[mask_local] = sx_arr + r * np.cos(th)
    y[mask_local] = sy_arr + r * np.sin(th)

    x = np.clip(x, 0.0, size)
    y = np.clip(y, 0.0, size)

    return x, y


# ──────────────────────────────────────────────────────────────────────────────
# Integration methods
# ──────────────────────────────────────────────────────────────────────────────

# Every registered function has the signature:
#   fn(n_samples, sources, dose_fn, roi_area, cfg) -> SingleRun
# (roi_area is kept for backward compatibility but ignored by MC/Sobol)


@register("mc")
def integrate_mc(
    n_samples: int,
    sources: np.ndarray,
    dose_fn: Callable,
    roi_area: float,
    cfg: BenchmarkConfig,
) -> SingleRun:
    """
    Hit-or-miss Monte Carlo over the full domain.
    Integral = L² / N × Σ f(x_i) for x_i in ROI & valid.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng()

    x = rng.uniform(0, cfg.size, n_samples)
    y = rng.uniform(0, cfg.size, n_samples)

    mask = roi_mask(x, y, cfg)
    x_roi, y_roi = x[mask], y[mask]

    doses = compute_total_dose(x_roi, y_roi, sources, dose_fn, cfg.exclusion_radius)
    estimate = (cfg.size ** 2 / n_samples) * np.nansum(doses)

    return SingleRun(estimate, time.perf_counter() - t0, len(x_roi))


@register("sobol")
def integrate_sobol(
    n_samples: int,
    sources: np.ndarray,
    dose_fn: Callable,
    roi_area: float,
    cfg: BenchmarkConfig,
) -> SingleRun:
    """Hit-or-miss Sobol QMC over the full domain."""
    t0 = time.perf_counter()

    sampler = qmc.Sobol(d=2, scramble=True)
    pts = sampler.random(n_samples)

    x = pts[:, 0] * cfg.size
    y = pts[:, 1] * cfg.size

    mask = roi_mask(x, y, cfg)
    x_roi, y_roi = x[mask], y[mask]

    doses = compute_total_dose(x_roi, y_roi, sources, dose_fn, cfg.exclusion_radius)
    estimate = (cfg.size ** 2 / n_samples) * np.nansum(doses)

    return SingleRun(estimate, time.perf_counter() - t0, len(x_roi))


@register("is_mc")
def integrate_importance_mc(
    n_samples: int,
    sources: np.ndarray,
    dose_fn: Callable,
    roi_area: float,
    cfg: BenchmarkConfig,
) -> SingleRun:
    """Importance-sampling MC with a uniform + per-source Gaussian mixture."""
    t0 = time.perf_counter()
    rng = np.random.default_rng()

    x, y = _sample_importance(n_samples, sources, cfg, rng)

    mask = roi_mask(x, y, cfg)
    x_roi, y_roi = x[mask], y[mask]

    doses = compute_total_dose(x_roi, y_roi, sources, dose_fn, cfg.exclusion_radius)
    q = _importance_pdf(x_roi, y_roi, sources, cfg)

    estimate = np.nansum(doses / q) / n_samples

    return SingleRun(estimate, time.perf_counter() - t0, len(x_roi))


@register("is_sobol")
def integrate_importance_sobol(
    n_samples: int,
    sources: np.ndarray,
    dose_fn: Callable,
    roi_area: float,
    cfg: BenchmarkConfig,
) -> SingleRun:
    """Importance-sampled Sobol QMC (vectorised Box-Muller)."""
    t0 = time.perf_counter()

    x, y = _sample_importance_qmc(n_samples, sources, cfg)

    mask = roi_mask(x, y, cfg)
    x_roi, y_roi = x[mask], y[mask]

    doses = compute_total_dose(x_roi, y_roi, sources, dose_fn, cfg.exclusion_radius)
    q = _importance_pdf(x_roi, y_roi, sources, cfg)

    estimate = np.nansum(doses / q) / n_samples

    return SingleRun(estimate, time.perf_counter() - t0, len(x_roi))


@register("grid")
def integrate_grid(
    n_samples: int,
    sources: np.ndarray,
    dose_fn: Callable,
    roi_area: float,
    cfg: BenchmarkConfig,
) -> SingleRun:
    """Midpoint-rule Cartesian grid."""
    t0 = time.perf_counter()
    grid_res = max(2, int(math.sqrt(n_samples)))

    dx = cfg.size / grid_res
    xi = np.linspace(dx / 2, cfg.size - dx / 2, grid_res)
    yi = np.linspace(dx / 2, cfg.size - dx / 2, grid_res)
    X, Y = np.meshgrid(xi, yi)

    mask = roi_mask(X, Y, cfg)
    doses = compute_total_dose(X, Y, sources, dose_fn, cfg.exclusion_radius)

    estimate = np.nansum(doses[mask]) * dx ** 2

    return SingleRun(estimate, time.perf_counter() - t0, int(mask.sum()))


# ──────────────────────────────────────────────────────────────────────────────
# Reference solutions
# ──────────────────────────────────────────────────────────────────────────────

def compute_shared_reference(cfg: BenchmarkConfig, sources, dose_fn, roi_area) -> float:
    """High-N Sobol estimate used as ground truth for all methods."""
    log.info("Computing shared reference (Sobol, N=%d) …", cfg.reference_samples)
    t0 = time.perf_counter()
    result = integrate_sobol(cfg.reference_samples, sources, dose_fn, roi_area, cfg)
    log.info("Time it took to calculate reference: %.3f", time.perf_counter()-t0)
    log.info("Shared reference integral: %.6f Gy·m²", result.estimate)
    return result.estimate


def compute_per_method_references(
    method_names: List[str], sources, dose_fn, roi_area, cfg: BenchmarkConfig,
) -> Dict[str, float]:
    """Compute a separate high-N reference for each requested method."""
    refs = {}
    for name in method_names:
        log.info("Computing reference for '%s' …", name)
        fn = _REGISTRY[name]
        result = fn(cfg.reference_samples, sources, dose_fn, roi_area, cfg)
        refs[name] = result.estimate
        log.info("  Reference for %s: %.6f Gy·m²", name, result.estimate)
    return refs


# ──────────────────────────────────────────────────────────────────────────────
# Convergence slope estimation
# ──────────────────────────────────────────────────────────────────────────────

def fit_convergence_slope(N_vals: List[int], errors: List[float]) -> float:
    """Fit log(error) = slope * log(N) + const via OLS."""
    log_N = np.log10(np.array(N_vals, dtype=float))
    log_e = np.log10(np.clip(errors, 1e-20, None))
    finite = np.isfinite(log_e)
    if finite.sum() < 2:
        return float("nan")
    slope, *_ = np.polyfit(log_N[finite], log_e[finite], 1)
    return float(slope)


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Orchestrates the full benchmark loop.

    By default uses a **single shared reference** (high-N Sobol).
    If *cfg.self_reference* is True, each method is compared to its own
    high-N run.
    """

    def __init__(
        self,
        cfg: BenchmarkConfig,
        methods: Optional[List[str]] = None,
    ):
        self.cfg = cfg
        self.method_names = methods or list(_REGISTRY.keys())

        unknown = set(self.method_names) - set(_REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown methods: {unknown}.  Available: {available_methods()}"
            )

        np.random.seed(cfg.seed)

        self.dose_fn: Callable = tg_43.load_dose_calculator()
        self.sources: np.ndarray = generate_sources(cfg)
        # roi_area is kept for logging only; integrators no longer depend on it
        self.roi_area: float = estimate_roi_area(cfg, 50_000_000)

        self.shared_reference: Optional[float] = None
        self.per_method_references: Dict[str, float] = {}
        self.results: Dict[str, List[MethodResult]] = {}

    # ------------------------------------------------------------------ run --

    def run(self) -> Dict[str, List[MethodResult]]:
        log.info("Sources:\n%s", self.sources)
        log.info("Effective ROI area (excl. source holes): %.4e m²", self.roi_area)
        log.info("Domain area: %.4e m²", self.cfg.size ** 2)

        # Decide which reference(s) to use
        if self.cfg.self_reference:
            self.per_method_references = compute_per_method_references(
                self.method_names, self.sources, self.dose_fn, self.roi_area, self.cfg
            )
        else:
            self.shared_reference = compute_shared_reference(
                self.cfg, self.sources, self.dose_fn, self.roi_area
            )

        sample_sizes = [
            2 ** p for p in range(self.cfg.min_power, self.cfg.max_power + 1)
        ]

        for name in self.method_names:
            self.results[name] = self._benchmark_one(name, sample_sizes)

        # Post-process: convergence slopes & efficiency
        for name, res_list in self.results.items():
            slope = fit_convergence_slope(
                [r.N for r in res_list],
                [r.mean_abs_error for r in res_list],
            )
            for r in res_list:
                r.convergence_slope = slope
                r.efficiency = r.mean_abs_error * r.mean_runtime

        return self.results

    # -------------------------------------------------------- _benchmark_one --

    def _benchmark_one(
        self,
        name: str,
        sample_sizes: List[int],
    ) -> List[MethodResult]:
        fn = _REGISTRY[name]
        # Get appropriate reference for error computation
        if self.cfg.self_reference:
            ref = self.per_method_references[name]
        else:
            ref = self.shared_reference

        log.info(
            "Benchmarking: %s  (reference = %.6e)", name, ref
        )

        results: List[MethodResult] = []
        for N in sample_sizes:
            estimates, runtimes, n_used = [], [], []

            for _ in range(self.cfg.benchmark_runs):
                run = fn(
                    N, self.sources, self.dose_fn, self.roi_area, self.cfg,
                )
                estimates.append(run.estimate)
                runtimes.append(run.runtime)
                n_used.append(run.samples_used)

            estimates = np.array(estimates)
            runtimes = np.array(runtimes)

            errors = np.abs(estimates - ref)

            t_crit = stats.t.ppf(
                (1 + self.cfg.confidence_level) / 2,
                df=len(estimates) - 1,
            )
            sem = np.std(estimates, ddof=1) / math.sqrt(len(estimates))

            mr = MethodResult(
                method_name=name,
                N=N,
                samples_used=float(np.mean(n_used)),
                mean_estimate=float(np.mean(estimates)),
                mean_abs_error=float(np.mean(errors)),
                std_estimate=float(np.std(estimates, ddof=1)),
                ci_low=float(np.mean(estimates) - t_crit * sem),
                ci_high=float(np.mean(estimates) + t_crit * sem),
                mean_runtime=float(np.mean(runtimes)),
                std_runtime=float(np.std(runtimes, ddof=1)),
            )
            results.append(mr)

            log.info(
                "  N=%7d | err=%10.4e | std=%10.4e | t=%7.4fs",
                N, mr.mean_abs_error, mr.std_estimate, mr.mean_runtime,
            )

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

_METHOD_LABELS = {
    "mc":       "Monte Carlo",
    "sobol":    "Sobol QMC",
    "is_mc":    "IS Monte Carlo",
    "is_sobol": "IS Sobol",
    "grid":     "Grid",
}

_COLORS = {
    "mc":       "#E07B54",
    "sobol":    "#5B9BD5",
    "is_mc":    "#70AD47",
    "is_sobol": "#9B59B6",
    "grid":     "#F4C430",
}


def save_csv(results: Dict[str, List[MethodResult]], path: Path) -> None:
    fields = list(MethodResult.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for res_list in results.values():
            for r in res_list:
                w.writerow(asdict(r))
    log.info("CSV saved → %s", path)


def save_json(
    results: Dict[str, List[MethodResult]],
    shared_ref: Optional[float],
    per_refs: Dict[str, float],
    path: Path,
) -> None:
    payload = {
        "shared_reference": shared_ref,
        "per_method_references": per_refs if per_refs else None,
        "methods": {
            name: [asdict(r) for r in res_list]
            for name, res_list in results.items()
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("JSON saved → %s", path)


def print_report(
    results: Dict[str, List[MethodResult]],
    shared_ref: Optional[float],
    per_refs: Dict[str, float],
    output_path: Optional[Path] = None,
) -> None:
    lines = ["=" * 78, "BRACHYTHERAPY INTEGRATION BENCHMARK — FINAL REPORT"]
    if shared_ref is not None:
        lines.append(f"Shared reference (Sobol): {shared_ref:.6e} Gy·m²")
    else:
        lines.append("Per-method self-references used:")
        for name, ref in per_refs.items():
            label = _METHOD_LABELS.get(name, name)
            lines.append(f"  {label:<20} {ref:.6e} Gy·m²")
    lines.append("=" * 78)

    lines.append(
        f"\n{'Method':<16} {'Samples':>10} {'Error':>12} {'Std Dev':>12}"
        f" {'CI 95% low':>12} {'CI 95% hi':>12} {'Runtime':>10} {'Slope':>8}"
    )
    lines.append("-" * 94)

    for name, res_list in results.items():
        r = res_list[-1]
        label = _METHOD_LABELS.get(name, name)
        lines.append(
            f"{label:<16} {r.N:>10,} {r.mean_abs_error:>12.4e} "
            f"{r.std_estimate:>12.4e} {r.ci_low:>12.4e} {r.ci_high:>12.4e} "
            f"{r.mean_runtime:>9.4f}s {r.convergence_slope:>7.2f}"
        )

    lines.append("\nConvergence slope reference:  MC ≈ -0.50   QMC ≈ -1.00")
    lines.append("=" * 78)

    text = "\n".join(lines)
    print(text)

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(text + "\n")
        log.info("Report saved → %s", output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _get_or_import_plt():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
    })
    return plt


def plot_convergence(
    results: Dict[str, List[MethodResult]],
    output_path: Optional[Path] = None,
) -> None:
    plt = _get_or_import_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, res_list in results.items():
        N = [r.N for r in res_list]
        err = [r.mean_abs_error for r in res_list]
        std = [r.std_estimate for r in res_list]
        color = _COLORS.get(name, None)
        label = _METHOD_LABELS.get(name, name)
        slope = res_list[-1].convergence_slope
        ax.loglog(
            N, err, "o-", color=color,
            label=f"{label}  (slope {slope:+.2f})", lw=1.8, ms=5,
        )

    N_ref = np.array([100, 50_000])
    ax.loglog(
        N_ref, 3e-2 * N_ref ** -0.50, "k--", lw=1, alpha=0.4, label="∝ N⁻¹/²",
    )
    ax.loglog(
        N_ref, 3e-1 * N_ref ** -1.00, "k:", lw=1, alpha=0.4, label="∝ N⁻¹",
    )

    ax.set_xlabel("Nominal sample size  N")
    ax.set_ylabel("Mean absolute error  (cGy·m²·s/h)")
    ax.set_title("Convergence comparison")
    ax.legend(loc="lower left")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        log.info("Plot saved → %s", output_path)
    plt.show()


def plot_variance(
    results: Dict[str, List[MethodResult]],
    output_path: Optional[Path] = None,
) -> None:
    plt = _get_or_import_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, res_list in results.items():
        if name == "grid":
            continue
        N = [r.N for r in res_list]
        std = [r.std_estimate for r in res_list]
        color = _COLORS.get(name, None)
        ax.loglog(
            N, std, "s-", color=color,
            label=_METHOD_LABELS.get(name, name), lw=1.8, ms=5,
        )

    ax.set_xlabel("Nominal sample size  N")
    ax.set_ylabel("Std deviation across runs  (cGy·m²·s/h)")
    ax.set_title("Estimator variance")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        log.info("Plot saved → %s", output_path)
    plt.show()


def plot_runtime(
    results: Dict[str, List[MethodResult]],
    output_path: Optional[Path] = None,
) -> None:
    plt = _get_or_import_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, res_list in results.items():
        N = [r.N for r in res_list]
        rt = [r.mean_runtime for r in res_list]
        rt_std = [r.std_runtime for r in res_list]
        color = _COLORS.get(name, None)
        ax.loglog(
            N, rt, "^-", color=color,
            label=_METHOD_LABELS.get(name, name), lw=1.8, ms=5,
        )

    ax.set_xlabel("Nominal sample size  N")
    ax.set_ylabel("Mean wall-clock runtime  (s)")
    ax.set_title("Runtime scaling")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        log.info("Plot saved → %s", output_path)
    plt.show()


def plot_efficiency_frontier(
    results: Dict[str, List[MethodResult]],
    output_path: Optional[Path] = None,
) -> None:
    plt = _get_or_import_plt()
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res_list in results.items():
        rt = np.array([r.mean_runtime for r in res_list])
        err = np.array([r.mean_abs_error for r in res_list])
        N_arr = [r.N for r in res_list]
        color = _COLORS.get(name, None)
        label = _METHOD_LABELS.get(name, name)

        ax.loglog(
            rt, err, "o-", color=color, label=label, lw=1.6, ms=6, zorder=3,
        )

        for i in range(0, len(N_arr), max(1, len(N_arr) // 4)):
            ax.annotate(
                f"N={N_arr[i]:,}",
                (rt[i], err[i]),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Mean runtime per call  (s)")
    ax.set_ylabel("Mean absolute error  (Gy·m²)")
    ax.set_title("Efficiency frontier  ←  lower-left is better")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        log.info("Plot saved → %s", output_path)
    plt.show()


def plot_all(
    results: Dict[str, List[MethodResult]],
    output_dir: Optional[Path] = None,
) -> None:
    def _out(stem: str) -> Optional[Path]:
        return (output_dir / stem) if output_dir else None

    plot_convergence(results, _out("convergence.png"))
    plot_variance(results, _out("variance.png"))
    plot_runtime(results, _out("runtime.png"))
    plot_efficiency_frontier(results, _out("efficiency.png"))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Brachytherapy integration benchmark suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--methods", nargs="+", default=None,
        choices=list(_REGISTRY.keys()),
        metavar="METHOD",
        help=f"Methods to benchmark. Available: {available_methods()}",
    )
    p.add_argument("--runs", type=int, default=20, help="Repeats per (method, N)")
    p.add_argument("--min-power", type=int, default=4, help="Smallest N = 2^min_power")
    p.add_argument("--max-power", type=int, default=19, help="Largest  N = 2^max_power")
    p.add_argument("--ref-power", type=int, default=26, help="Reference N = 2^ref_power")
    p.add_argument(
        "--output", type=str, default=None,
        help="Directory for CSV/JSON/PNG output (created if absent)",
    )
    p.add_argument("--no-plot", action="store_true", help="Disable matplotlib output")
    p.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    p.add_argument(
        "--self-reference", action="store_true",
        help="Use per-method high-N reference instead of a single shared one",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir: Optional[Path] = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info("Output directory: %s", output_dir)

    cfg = BenchmarkConfig(
        benchmark_runs=args.runs,
        min_power=args.min_power,
        max_power=args.max_power,
        reference_samples=2 ** args.ref_power,
        self_reference=args.self_reference,
        output_dir=output_dir,
        plot=not args.no_plot,
        seed=args.seed,
    )

    runner = BenchmarkRunner(cfg, methods=args.methods)
    results = runner.run()

    # Save artefacts
    if output_dir:
        save_csv(results, output_dir / "results.csv")
        save_json(
            results,
            runner.shared_reference,
            runner.per_method_references,
            output_dir / "results.json",
        )

    print_report(
        results,
        runner.shared_reference,
        runner.per_method_references,
        output_dir / "report.txt" if output_dir else None,
    )

    if cfg.plot:
        plot_all(results, output_dir)


if __name__ == "__main__":
    main()