import tg_43
from pathlib import Path
import numpy as np
from distance_field_precomputation import load_sdf, SignedDistanceField, plot_sdf, evaluate_bspline_contour
import matplotlib.pyplot as plt
from scipy.stats import qmc
from typing import Callable
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches

def visualize_state(optimization, state, ax=None, dose_vmin=None, dose_vmax=None, dwell_vmin=None, dwell_vmax=None):
    """
    Visualize the current state of brachytherapy optimization.

    Args:
        optimization: BrachytherapyOptimization instance
        source_positions: (n_sources, 2) array of source positions in meters
        dwell_times: (n_sources,) array of dwell times (optional)
        ax: matplotlib axis (creates new figure if None)
        dose_vmin: Minimum value for dose colorbar (optional)
        dose_vmax: Maximum value for dose colorbar (optional)
        dwell_vmin: Minimum value for dwell time colorbar (optional)
        dwell_vmax: Maximum value for dwell time colorbar (optional)
    """
    source_positions = state[:,0:2]
    dwell_times = state[:,2]
    # Create grid for visualization (higher resolution than Monte Carlo points)
    grid_res = 300
    x = np.linspace(optimization.bounds[0][0], optimization.bounds[0][1], grid_res)
    y = np.linspace(optimization.bounds[1][0], optimization.bounds[1][1], grid_res)
    X, Y = np.meshgrid(x, y)

    # Calculate dose on grid (simplified for visualization only)
    doses = np.zeros_like(X)
    valid_mask = np.ones_like(X, dtype=bool)

    for i, (sx, sy) in enumerate(source_positions):
        source_dist = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2)
        valid = source_dist > optimization.exclusion_radius
        valid_mask = valid_mask & valid
        doses[valid] += optimization.dose_calculator(source_dist[valid]) * dwell_times[i] / 3600 /100

    doses[~valid_mask] = np.nan

    # Mask to prostate and contours
    prostate_mask = optimization.whole_prostate_sdf(X, Y) <= 0
    doses[~prostate_mask] = np.nan

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Plot dose distribution - vmin/vmax work when non-None
    im = ax.pcolormesh(X * 100, Y * 100, doses, cmap='plasma',
                       shading='auto', vmin=dose_vmin, vmax=dose_vmax)

    # Plot contours
    # Prostate contour (outer boundary)
    contour_levels = np.linspace(-3e-3, 3e-3, 20)
    ax.contour(X * 100, Y * 100, optimization.whole_prostate_sdf(X, Y),
               levels=[0], colors='blue', linewidths=2, label='Prostate boundary')
    # Urethra contour (inner avoidance)
    ax.contour(X * 100, Y * 100, optimization.urethra_sdf(X, Y),
               levels=[0], colors='red', linewidths=2, linestyles='--', label='Urethra')
    #
    prescribed = optimization.prescribed_dose / 3600 / 100
    levels = [prescribed]
    ax.contour(X * 100, Y * 100, doses, levels=levels,
               colors='white', linewidths=1, alpha=0.7, linestyles = '--')
    # Plot sources - vmin/vmax work when non-None
    source_x_cm = source_positions[:, 0] * 100
    source_y_cm = source_positions[:, 1] * 100
    scatter = ax.scatter(source_x_cm, source_y_cm, c=dwell_times,
                         s=100, cmap='cool', edgecolors='white',
                         linewidths=2, zorder=5, vmin=dwell_vmin, vmax=dwell_vmax)

    # Add exclusion circles around sources (1mm radius)
    for sx, sy in zip(source_x_cm, source_y_cm):
        exclusion_circle = Circle((sx, sy), optimization.exclusion_radius * 100,
                                  color='white', fill=False, linestyle=':', linewidth=1)
        ax.add_patch(exclusion_circle)

    # Formatting
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title('Dose Distribution')
    ax.set_aspect('equal', adjustable='box')

    # Colorbars - these will automatically respect the vmin/vmax set above
    cbar1 = fig.colorbar(im, ax=ax, label='Dose (Gy)')
    cbar2 = fig.colorbar(scatter, ax=ax, label='Dwell Time (s)')

    # Create legend for contours
    blue_patch = mpatches.Patch(color='blue', label='Prostate')
    red_patch = mpatches.Patch(color='red', linestyle='--', label='Urethra')
    ax.legend(handles=[blue_patch, red_patch], loc='upper right')

    plt.tight_layout()
    return fig, ax

class BrachytherapyOptimization:
    __slots__ = ['dose_calculator', 'prostate_sdf', 'urethra_sdf', 'whole_prostate_sdf',
                 'bounds', 'x_scale', 'y_scale', 'n_samples', 'sampler', 'box_area', 'exclusion_radius_squared', 'exclusion_radius',
                 'integration_points', 'prostate_mask', 'urethra_mask', 'prescribed_dose']

    def __init__(self, prostate_sdf: SignedDistanceField, urethra_sdf: SignedDistanceField,
                 whole_prostate_sdf: SignedDistanceField, n_samples: int, exclusion_radius: float, prescribed_dose: float):
        self.dose_calculator: callable = tg_43.load_dose_calculator()
        self.prostate_sdf = prostate_sdf
        self.urethra_sdf = urethra_sdf
        self.whole_prostate_sdf = whole_prostate_sdf

        self.bounds = [(self.prostate_sdf.xmin, self.prostate_sdf.xmax),
                       (self.prostate_sdf.ymin, self.prostate_sdf.ymax)]
        self.x_scale = (self.bounds[0][1] - self.bounds[0][0])
        self.y_scale = (self.bounds[1][1] - self.bounds[1][0])

        self.n_samples = n_samples
        self.sampler = qmc.Sobol(d=2, scramble=True)
        self.update_integration_points()

        self.box_area = self.y_scale * self.x_scale
        self.exclusion_radius = exclusion_radius
        self.exclusion_radius_squared = exclusion_radius ** 2

        self.prescribed_dose = prescribed_dose * 3600 * 100 # Convert to cGy * s/h

    def update_integration_points(self) -> None:
        points = self.sampler.random(self.n_samples)
        x = (points[:, 0]) * self.x_scale + self.bounds[0][0]
        y = (points[:, 1]) * self.y_scale + self.bounds[1][0]
        mask = self.whole_prostate_sdf(x, y) <= 0
        self.integration_points = np.column_stack((x[mask], y[mask]))
        self.prostate_mask = self.prostate_sdf(self.integration_points[:, 0], self.integration_points[:, 1]) <= 0
        self.urethra_mask = self.urethra_sdf(self.integration_points[:, 0], self.integration_points[:, 1]) <= 0

    def evaluate(self, x) -> float:
        x = x.reshape([x.size//3,3])
        source_positions = x[:, 0:2]
        dwell_times = x[:, 2]

        valid_mask = self.get_valid_mask(source_positions)
        points = self.integration_points[valid_mask]
        dose = np.zeros(points.shape[0])
        for source in x:
            radii = np.hypot(points[:, 0] - source[0], points[:, 1] - source[1])
            dose += self.dose_calculator(radii) * source[2]
        objective_value = self.box_area * np.sum(dose) / self.n_samples

        # Constraint 1
        prescribed_dose_mask = dose[self.prostate_mask[valid_mask]] > self.prescribed_dose
        constraint1 = 0.9 - np.count_nonzero(prescribed_dose_mask)/np.count_nonzero(self.prostate_mask[valid_mask])

        # Constraint 2
        over_dosage_mask = dose[self.urethra_mask[valid_mask]] > (self.prescribed_dose*1.15)
        constraint2 = np.count_nonzero(over_dosage_mask)/np.count_nonzero(self.urethra_mask[valid_mask]) - 0.10

        # Source Constraints
        proximity_constraint = []
        valid_region_constraint = []
        for i, position in enumerate(source_positions):
            dist = np.sqrt(np.sum((source_positions - position)**2,axis=1))
            dist[i] = np.inf
            proximity_constraint.append(5e-3 - np.min(dist[:]))
            valid_region_constraint.append(2e-3 + self.prostate_sdf(position[0], position[1]))

        constraints = np.array([constraint1, constraint2, *proximity_constraint, *valid_region_constraint])

        return objective_value, constraints

    def evaluate_penalized(self, x):
        penalization_factor = 1e6
        objective_value, constraints = self.evaluate(x)
        x = x.reshape([x.size//3,3])
        constraints_conditioning = np.array([100, 100] + [1e5]*x.shape[0] + [1e5]*x.shape[0])
        return objective_value + np.sum(constraints_conditioning * np.clip(constraints, 0, None)**2)*penalization_factor

    def get_valid_mask(self, source_positions):
        valid_mask = np.ones([self.integration_points.shape[0]], dtype=bool)
        for position in source_positions:
            valid_mask &= np.sum((self.integration_points - position) ** 2, axis=1) >= self.exclusion_radius_squared
        return valid_mask


def main():
    whole_prostate = SignedDistanceField(*load_sdf("prostate_sdf.npz"))
    urethra = SignedDistanceField(*load_sdf("urethra_sdf.npz"))
    assert (whole_prostate.metadata == urethra.metadata)
    prostate = SignedDistanceField(np.maximum(whole_prostate.sdf, -urethra.sdf), whole_prostate.metadata)
    from distance_field_precomputation import plot_queries
    plot_queries(prostate)
    plot_queries(urethra)
    plot_queries(whole_prostate)
    source_count = 6
    theta = np.linspace(0, 2 * np.pi, source_count, endpoint=False)
    r = 15e-3
    x = np.column_stack((r * np.cos(theta), r * np.sin(theta), 2*np.ones(source_count)))

    problem = BrachytherapyOptimization(prostate_sdf=prostate, urethra_sdf=urethra, whole_prostate_sdf=whole_prostate,
                                        n_samples=2**20, exclusion_radius=1e-3, prescribed_dose= 5)
    import time
    t0 = time.perf_counter()
    a =problem.evaluate(x.flatten())
    print(time.perf_counter()-t0)
    print(a)
    fig, ax = visualize_state(problem, x)
    plt.show()

if __name__ == '__main__':
    main()
