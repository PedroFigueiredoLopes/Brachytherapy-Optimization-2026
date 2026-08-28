import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from matplotlib.path import Path
from scaling_bsplines import evaluate_bspline_contour

def load_bspline_npz(filename):
    """
    Load B-spline parameters from an npz file.

    Returns:
        tck tuple (t, [c_x, c_y], k) and u array
    """
    data = np.load(filename, allow_pickle=True)

    t = data['t']
    c = [data['c_x'], data['c_y']]
    k = int(data['k'])  # Convert from numpy type to int

    tck = (t, c, k)

    structure_name = data.get('structure_name', 'unknown')
    print(f"Loaded B-spline parameters for: {structure_name}")

    return tck, structure_name


def compute_signed_distance_field(tck, grid_shape=(2048, 2048), bounds=None):
    """
    Precompute a signed distance field from a closed B-spline.

    Args:
        tck: B-spline parameters from splprep
        grid_shape: (height, width) of distance map (higher = more accurate)
        bounds: (xmin, xmax, ymin, ymax) or None for auto

    Returns:
        sdf: 2D array of signed distances (negative inside, positive outside)
        metadata: dict with bounds, grid_shape, etc.
        grid: (xx, yy) meshgrid for plotting
    """
    print("Sampling contour...")
    contour_pts = evaluate_bspline_contour(tck, num_points=5000)

    # Determine bounds
    if bounds is None:
        xmin, xmax = contour_pts[:, 0].min(), contour_pts[:, 0].max()
        ymin, ymax = contour_pts[:, 1].min(), contour_pts[:, 1].max()
        # Add 10% padding
        pad_x = (xmax - xmin) * 0.1
        pad_y = (ymax - ymin) * 0.1
        xmin, xmax = xmin - pad_x, xmax + pad_x
        ymin, ymax = ymin - pad_y, ymax + pad_y
        print(f"Bounds: x=[{xmin:.2f}, {xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}]")
    else:
        xmin = bounds[0][0]
        xmax = bounds[0][1]
        ymin = bounds[1][0]
        ymax = bounds[1][1]
    # Create grid
    x = np.linspace(xmin, xmax, grid_shape[1])
    y = np.linspace(ymin, ymax, grid_shape[0])
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Compute distances to contour
    print(f"Computing distances for {grid_shape[0] * grid_shape[1]:,} points...")
    tree = cKDTree(contour_pts)
    distances, _ = tree.query(points)
    distance_map = distances.reshape(grid_shape)

    # Determine inside/outside using winding number
    print("Determining inside/outside...")
    path = Path(contour_pts)
    inside = path.contains_points(points).reshape(grid_shape)

    # Make signed: negative inside, positive outside
    sdf = distance_map.copy()
    sdf[inside] *= -1

    metadata = {
        'xmin': xmin, 'xmax': xmax,
        'ymin': ymin, 'ymax': ymax,
        'grid_shape': grid_shape,
        'dx': (xmax - xmin) / (grid_shape[1] - 1),
        'dy': (ymax - ymin) / (grid_shape[0] - 1)
    }

    return sdf, metadata, (xx, yy)


class SignedDistanceField:
    """Fast O(1) lookup of precomputed signed distance field."""

    def __init__(self, sdf, metadata):
        self.sdf = sdf
        self.metadata = metadata

        # Unpack metadata for fast access
        self.xmin = float(metadata['xmin'])
        self.xmax = float(metadata['xmax'])
        self.ymin = float(metadata['ymin'])
        self.ymax = float(metadata['ymax'])
        self.h, self.w = self.sdf.shape
        self.dx = metadata['dx']
        self.dy = metadata['dy']

    def __call__(self, x, y):
        """
        Query signed distance at (x, y).

        Supports:
        - Scalars: distance = sdf(5.0, 3.0)
        - Arrays: distances = sdf(x_array, y_array)
        """
        # Convert to grid coordinates
        i = (x - self.xmin) / self.dx
        j = (y - self.ymin) / self.dy

        # Bilinear interpolation
        i0 = np.floor(i).astype(int)
        i1 = i0 + 1
        j0 = np.floor(j).astype(int)
        j1 = j0 + 1

        # Clip to bounds
        i0 = np.clip(i0, 0, self.w - 1)
        i1 = np.clip(i1, 0, self.w - 1)
        j0 = np.clip(j0, 0, self.h - 1)
        j1 = np.clip(j1, 0, self.h - 1)

        # Bilinear weights
        wi = i - i0
        wj = j - j0

        # Sample
        v00 = self.sdf[j0, i0]
        v01 = self.sdf[j1, i0]
        v10 = self.sdf[j0, i1]
        v11 = self.sdf[j1, i1]

        # Interpolate
        v0 = v00 * (1 - wj) + v01 * wj
        v1 = v10 * (1 - wj) + v11 * wj
        return v0 * (1 - wi) + v1 * wi

    def __getitem__(self, idx):
        """Direct array indexing (for debugging)."""
        return self.sdf[idx]


    def direct(self, x, y):
        """
        Direct grid sampling (no interpolation) - fastest but less accurate.
        Points are snapped to nearest grid cell.

        Returns:
            Signed distance at nearest grid point
        """
        # Convert to grid indices
        i = np.round((x - self.xmin) / self.dx).astype(int)
        j = np.round((y - self.ymin) / self.dy).astype(int)

        # Clip to bounds
        i = np.clip(i, 0, self.w - 1)
        j = np.clip(j, 0, self.h - 1)

        return self.sdf[j, i]

def plot_sdf(sdf, metadata, contour_tck=None, title="Signed Distance Field"):
    """Plot the signed distance field with contour overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Full SDF
    im1 = axes[0].imshow(sdf, extent=[metadata['xmin'], metadata['xmax'],
                                      metadata['ymin'], metadata['ymax']],
                         origin='lower', cmap='RdBu_r')
    axes[0].set_title(f"{title}\nFull Field")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    plt.colorbar(im1, ax=axes[0], label="Signed Distance [mm]")

    # Zoomed view around zero (contour)
    # Find where SDF is near zero
    zero_crossing = np.abs(sdf) < np.percentile(np.abs(sdf), 10)
    zoom_bounds = None
    if np.any(zero_crossing):
        ys, xs = np.where(zero_crossing)
        ymin_z = max(0, ys.min() - 50)
        ymax_z = min(sdf.shape[0], ys.max() + 50)
        xmin_z = max(0, xs.min() - 50)
        xmax_z = min(sdf.shape[1], xs.max() + 50)

        zoom_sdf = sdf[ymin_z:ymax_z, xmin_z:xmax_z]
        xmin_g = metadata['xmin'] + xmin_z * metadata['dx']
        xmax_g = metadata['xmin'] + xmax_z * metadata['dx']
        ymin_g = metadata['ymin'] + ymin_z * metadata['dy']
        ymax_g = metadata['ymin'] + ymax_z * metadata['dy']

        im2 = axes[1].imshow(zoom_sdf, extent=[xmin_g, xmax_g, ymin_g, ymax_g],
                             origin='lower', cmap='RdBu_r')
        axes[1].set_title("Zoomed at Contour")
        axes[1].set_xlabel("x [mm]")
        axes[1].set_ylabel("y [mm]")
        plt.colorbar(im2, ax=axes[1], label="Signed Distance [mm]")
    else:
        axes[1].set_title("No contour region found")

    # Contour line only
    axes[2].contour(sdf, levels=[0], extent=[metadata['xmin'], metadata['xmax'],
                                             metadata['ymin'], metadata['ymax']],
                    origin='lower', colors='black', linewidths=2)

    # Overlay original B-spline if provided
    if contour_tck is not None:
        contour_pts = evaluate_bspline_contour(contour_tck, num_points=500)
        axes[2].plot(contour_pts[:, 0], contour_pts[:, 1], 'r--', linewidth=1,
                     label='B-spline contour', alpha=0.7)

    axes[2].set_title("Zero Crossing (Contour)")
    axes[2].set_xlabel("x [mm]")
    axes[2].set_ylabel("y [mm]")
    axes[2].set_aspect('equal')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_distance_profile(sdf_field, contour_tck, num_points=1000):
    """Plot distances along a line through the contour for validation."""
    # Create a line through the centroid
    contour_pts = evaluate_bspline_contour(contour_tck, num_points=500)
    centroid = contour_pts.mean(axis=0)

    # Line from left to right through centroid
    xmin = sdf_field.xmin
    xmax = sdf_field.xmax
    y_line = centroid[1]

    x_vals = np.linspace(xmin, xmax, num_points)
    y_vals = np.full_like(x_vals, y_line)
    distances = sdf_field(x_vals, y_vals)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_vals, distances, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=centroid[0], color='r', linestyle='--', alpha=0.5, label='Centroid')
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Signed Distance [mm]")
    ax.set_title(f"Distance Profile at y = {y_line:.2f} mm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_queries(sdf_field, num_queries=5000):
    """Test random queries and plot them colored by distance."""
    # Generate random points within bounds
    x_rand = np.random.uniform(sdf_field.xmin, sdf_field.xmax, num_queries)
    y_rand = np.random.uniform(sdf_field.ymin, sdf_field.ymax, num_queries)
    distances = sdf_field(x_rand, y_rand) * 1000

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(x_rand, y_rand, c=distances, cmap='RdBu_r',
                         s=10, alpha=0.6, vmin=-10, vmax=10)
    ax.set_xlim(sdf_field.xmin, sdf_field.xmax)
    ax.set_ylim(sdf_field.ymin, sdf_field.ymax)
    ax.set_aspect('equal')
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Random Queries ({num_queries:,} points)\nColor = Signed Distance")
    plt.colorbar(scatter, label="Distance [mm]")
    ax.grid(True, alpha=0.3)
    return fig


def save_sdf(sdf, metadata, filename):
    """
    Save signed distance field and metadata to a compressed .npz file.

    Args:
        sdf: 2D array of signed distances
        metadata: dict with bounds, grid_shape, dx, dy, etc.
        filename: output filename (should end with .npz)
    """
    # Convert metadata to arrays for saving (npz doesn't like nested dicts)
    save_dict = {
        'sdf': sdf.astype(np.float32),  # Save as float32 to save space
        'xmin': metadata['xmin'],
        'xmax': metadata['xmax'],
        'ymin': metadata['ymin'],
        'ymax': metadata['ymax'],
        'dx': metadata['dx'],
        'dy': metadata['dy'],
        'grid_shape_h': metadata['grid_shape'][0],
        'grid_shape_w': metadata['grid_shape'][1],
    }

    np.savez_compressed(filename, **save_dict)
    print(f"Saved SDF to: {filename}")
    print(f"  Size: {sdf.nbytes / 1024 / 1024:.2f} MB (uncompressed)")


def load_sdf(filename):
    """
    Load signed distance field and metadata from a .npz file.

    Returns:
        sdf: 2D array of signed distances
        metadata: dict with bounds, grid_shape, dx, dy, etc.
    """
    data = np.load(filename)

    sdf = data['sdf']

    metadata = {
        'xmin': float(data['xmin']),
        'xmax': float(data['xmax']),
        'ymin': float(data['ymin']),
        'ymax': float(data['ymax']),
        'dx': float(data['dx']),
        'dy': float(data['dy']),
        'grid_shape': (int(data['grid_shape_h']), int(data['grid_shape_w'])),
    }

    # print(f"Loaded SDF from: {filename}")
    # print(f"  Shape: {sdf.shape}, dtype: {sdf.dtype}")

    return sdf, metadata

def main():
    # Load B-spline
    print("Loading B-spline...")
    prostate_tck, _ = load_bspline_npz("prostate_bspline_transformed.npz")

    contour_pts = evaluate_bspline_contour(prostate_tck, num_points=5000)
    xmin, xmax = contour_pts[:, 0].min(), contour_pts[:, 0].max()
    ymin, ymax = contour_pts[:, 1].min(), contour_pts[:, 1].max()
    print(f"Bounds: x=[{xmin:.3f}, {xmax:.3f}], y=[{ymin:.3f}, {ymax:.3f}]")

    bspline_tck, _ = load_bspline_npz("prostate_bspline_transformed.npz") # or load_bspline_npz("urethra_bspline_transformed.npz")
    file_name_out = "prostate_sdf.npz" # or "urethra_sdf.npz" 
    # Precompute SDF (faster for testing with lower resolution)
    print("\n--- Computing Signed Distance Field ---")
    grid_size = 1024
    sdf, metadata, grid = compute_signed_distance_field(
        bspline_tck,
        grid_shape=(grid_size, grid_size),
        bounds = [[xmin, xmax],[ymin,ymax]]
    )

    print(f"\nSDF shape: {sdf.shape}")
    print(f"Memory usage: {sdf.nbytes / 1024 ** 2:.2f} MB")

    save_sdf(sdf, metadata, file_name_out)
    print(f"SDF saved to: {file_name_out}")
    # Create query object
    sdf_field = SignedDistanceField(sdf, metadata)

    # Test a few queries
    print("\n--- Testing Queries ---")
    test_points = [(0, 0), (10, 5), (-10, -5), (20, 0)]
    for x, y in test_points:
        dist = sdf_field(x, y)
        print(f"  sdf({x:6.2f}, {y:6.2f}) = {dist:8.4f} mm")

    # Plot SDF
    print("\n--- Generating Plots ---")
    fig1 = plot_sdf(sdf, metadata, bspline_tck, title="Prostate SDF")

    fig2 = plot_distance_profile(sdf_field, bspline_tck)

    fig3 = plot_queries(sdf_field, num_queries=5000)

    # Performance test
    print("\n--- Performance Test ---")
    import time

    num_test_queries = 100000
    x_test = np.random.uniform(metadata['xmin'], metadata['xmax'], num_test_queries)
    y_test = np.random.uniform(metadata['ymin'], metadata['ymax'], num_test_queries)

    start = time.time()
    distances = sdf_field(x_test, y_test)
    elapsed = time.time() - start

    print(f"Queried {num_test_queries:,} points in {elapsed:.3f} seconds")
    print(f"  Rate: {num_test_queries / elapsed:.0f} queries/second")

    start = time.time()
    distances = sdf_field.direct(x_test, y_test)
    elapsed = time.time() - start

    print(f"Queried {num_test_queries:,} points in {elapsed:.3f} seconds")
    print(f"  Rate: {num_test_queries / elapsed:.0f} queries/second")

    # Show all plots
    plt.show()

    print("\n--- Done ---")

if __name__ == "__main__":
    main()