import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


def get_contour_from_clicks(image_path, structure_name="structure"):
    """Click points around a contour, then press Enter when done."""
    fig, ax = plt.subplots()

    img = plt.imread(image_path)
    ax.imshow(img)
    ax.set_title(f"Click points around the {structure_name} contour, press Enter when done")

    points = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    return np.array(points)


def fit_bspline(points, num_points=200, smoothing=0):
    """
    Fit a closed B-spline through the points.

    Args:
        points: Nx2 array of clicked points
        num_points: Number of points in the smooth output curve
        smoothing: Smoothing factor (0 = interpolate exactly)
    """
    if len(points) < 4:
        raise ValueError("You need at least 4 points for a cubic closed B-spline.")

    points_closed = np.vstack([points, points[0]])

    tck, u = splprep(
        [points_closed[:, 0], points_closed[:, 1]],
        s=smoothing,
        per=True,
        k=3
    )

    u_new = np.linspace(0, 1, num_points)
    x_smooth, y_smooth = splev(u_new, tck)

    smooth_contour = np.column_stack([x_smooth, y_smooth])
    return smooth_contour, tck, u


def save_bspline_npz(tck, u, filename, structure_name=""):
    """
    Save B-spline parameters (tck and u) to an npz file.

    Args:
        tck: tuple of (t, c, k) from splprep
        u: parameter values from splprep
        filename: output filename (.npz extension)
        structure_name: optional name for the structure (for metadata)
    """
    t, c, k = tck

    # c is a list of arrays [c_x, c_y] for 2D curves
    # Save them as separate arrays
    np.savez(
        filename,
        t=t,  # knot vector
        c_x=c[0],  # x coefficients (control points)
        c_y=c[1],  # y coefficients (control points)
        k=k,  # degree
        u=u,  # parameter values from fitting
        structure_name=structure_name
    )
    print(f"Saved B-spline parameters to: {filename}")


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
    u = data['u']

    tck = (t, c, k)

    structure_name = data.get('structure_name', 'unknown')
    print(f"Loaded B-spline parameters for: {structure_name}")

    return tck, u


def evaluate_bspline_contour(tck, num_points=200):
    """
    Evaluate a B-spline contour at evenly spaced parameter values.

    Args:
        tck: tuple of (t, c, k) from splprep
        num_points: number of points to generate

    Returns:
        Nx2 array of contour points
    """
    u_new = np.linspace(0, 1, num_points)
    x_smooth, y_smooth = splev(u_new, tck)
    return np.column_stack([x_smooth, y_smooth])


def main():
    image_path = "prostate.png"

    # ---------- PROSTATE ----------
    print("Step 1: Trace the prostate contour.")
    prostate_points = get_contour_from_clicks(image_path, structure_name="prostate")

    if prostate_points.size == 0:
        raise RuntimeError("No prostate points were clicked.")

    prostate_contour, prostate_tck, prostate_u = fit_bspline(
        prostate_points,
        num_points=200,
        smoothing=0
    )

    # Save as npz (B-spline parameters only)
    save_bspline_npz(prostate_tck, prostate_u, "prostate_bspline.npz", structure_name="prostate")

    # ---------- URETHRA ----------
    print("Step 2: Trace the urethra contour.")
    urethra_points = get_contour_from_clicks(image_path, structure_name="urethra")

    if urethra_points.size == 0:
        raise RuntimeError("No urethra points were clicked.")

    urethra_contour, urethra_tck, urethra_u = fit_bspline(
        urethra_points,
        num_points=200,
        smoothing=0
    )

    # Save as npz (B-spline parameters only)
    save_bspline_npz(urethra_tck, urethra_u, "urethra_bspline.npz", structure_name="urethra")

    # ---------- DEMO: LOAD AND RECONSTRUCT ----------
    print("\n--- Demonstrating loading from npz ---")

    # Load prostate B-spline and reconstruct the contour
    loaded_prostate_tck, loaded_prostate_u = load_bspline_npz("prostate_bspline.npz")
    reconstructed_prostate = evaluate_bspline_contour(loaded_prostate_tck, num_points=200)

    # Load urethra B-spline and reconstruct the contour
    loaded_urethra_tck, loaded_urethra_u = load_bspline_npz("urethra_bspline.npz")
    reconstructed_urethra = evaluate_bspline_contour(loaded_urethra_tck, num_points=200)

    # Verify reconstruction matches (should be very close)
    print(f"Prostate reconstruction error: {np.max(np.abs(prostate_contour - reconstructed_prostate)):.6f}")
    print(f"Urethra reconstruction error: {np.max(np.abs(urethra_contour - reconstructed_urethra)):.6f}")

    # ---------- FINAL PLOT ----------
    fig, ax = plt.subplots()
    img = plt.imread(image_path)
    ax.imshow(img)

    # prostate (using reconstructed to show loading works)
    ax.plot(
        prostate_points[:, 0], prostate_points[:, 1],
        'ro', markersize=5, label='Prostate control points'
    )
    ax.plot(
        reconstructed_prostate[:, 0], reconstructed_prostate[:, 1],
        'r-', linewidth=2, label='Prostate B-spline (from npz)'
    )

    # urethra (using reconstructed to show loading works)
    ax.plot(
        urethra_points[:, 0], urethra_points[:, 1],
        'go', markersize=5, label='Urethra control points'
    )
    ax.plot(
        reconstructed_urethra[:, 0], reconstructed_urethra[:, 1],
        'g-', linewidth=2, label='Urethra B-spline (from npz)'
    )

    ax.set_aspect('equal')
    ax.legend()
    plt.title("Prostate and urethra contours (loaded from .npz)")
    plt.show()

    print(f"\nProstate contour shape: {reconstructed_prostate.shape}")
    print(f"Urethra contour shape: {reconstructed_urethra.shape}")
    print("\nFiles saved:")
    print("  - prostate_bspline.npz")
    print("  - urethra_bspline.npz")


if __name__ == '__main__':
    main()

    # Shows the bspline
    """
    tck, u = load_bspline_npz("prostate_bspline.npz")
    t, c, k = tck

    from scipy.interpolate import BSpline

    u_new = np.linspace(0, 1, 200)
    spline_x = BSpline(t, c[0], k)
    spline_y = BSpline(t, c[1], k)

    x_smooth = spline_x(u_new)
    y_smooth = spline_y(u_new)
    fig, ax = plt.subplots()
    ax.plot(x_smooth, y_smooth)
    plt.show()
    """
