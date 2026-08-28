import numpy as np
import matplotlib.pyplot as plt
from bsplines_from_png import load_bspline_npz
from scipy.interpolate import BSpline


def evaluate_bspline_contour(tck, num_points=200):
    """
    Evaluate a B-spline contour at evenly spaced parameter values.

    Args:
        tck: tuple of (t, c, k) from splprep
        num_points: number of points to generate

    Returns:
        Nx2 array of contour points
    """
    t, c, k = tck
    spline_x = BSpline(t, c[0], k)
    spline_y = BSpline(t, c[1], k)

    u_new = np.linspace(0, 1, num_points)
    x_smooth = spline_x(u_new)
    y_smooth = spline_y(u_new)

    return np.column_stack([x_smooth, y_smooth])


def sample_bspline_uniform(tck, num_points=200):
    """Alias for evaluate_bspline_contour - same thing."""
    return evaluate_bspline_contour(tck, num_points)


def get_bspline_control_points(tck):
    """Extract control points from the B-spline coefficients."""
    t, c, k = tck
    # c[0] and c[1] are the control point coordinates for x and y
    control_points = np.column_stack([c[0], c[1]])
    return control_points


def get_width_height(contour):
    x = contour[:, 0]
    y = contour[:, 1]
    width = x.max() - x.min()
    height = y.max() - y.min()
    return width, height


def close_contour(contour):
    return np.vstack([contour, contour[0]])


def save_bspline_npz(tck, filename, structure_name="", u=None):
    """
    Save B-spline parameters to an npz file.

    Args:
        tck: tuple of (t, c, k) from splprep
        filename: output filename (.npz extension)
        structure_name: optional name for the structure (for metadata)
        u: optional parameter values from fitting
    """
    t, c, k = tck

    save_dict = {
        't': t,
        'c_x': c[0],
        'c_y': c[1],
        'k': k,
        'structure_name': structure_name
    }

    if u is not None:
        save_dict['u'] = u

    np.savez(filename, **save_dict)
    print(f"Saved B-spline to: {filename}")


def transform_bspline(tck, scale=1.0, translate=(0, 0), flip_y=False):
    """
    Apply geometric transformations to a B-spline.

    Args:
        tck: tuple of (t, c, k) from splprep
        scale: uniform scale factor
        translate: (dx, dy) translation
        flip_y: whether to flip Y coordinate

    Returns:
        New tck tuple with transformed coefficients
    """
    t, c, k = tck

    # Copy coefficients
    c_x = c[0].copy()
    c_y = c[1].copy()

    # Apply scaling
    c_x *= scale
    c_y *= scale

    # Apply Y flip if requested (before translation)
    if flip_y:
        c_y *= -1

    # Apply translation
    dx, dy = translate
    c_x += dx
    c_y += dy

    return (t, [c_x, c_y], k)


def main():
    # -------- SETTINGS --------
    prostate_file = "prostate_bspline.npz"
    urethra_file = "urethra_bspline.npz"

    target_prostate_width = 40.0e-3 # 40mm
    num_sample_points = 200

    do_center = True
    do_flip_y = True

    prostate_out_npz = "prostate_bspline_transformed.npz"
    urethra_out_npz = "urethra_bspline_transformed.npz"
    # --------------------------

    # Load B-splines
    print("Loading B-splines...")
    prostate_tck, prostate_u = load_bspline_npz(prostate_file)
    urethra_tck, urethra_u = load_bspline_npz(urethra_file)

    # Sample contours for analysis (original, unscaled)
    prostate_original = evaluate_bspline_contour(prostate_tck, num_sample_points)
    urethra_original = evaluate_bspline_contour(urethra_tck, num_sample_points)

    # Get prostate dimensions in original coordinates
    p_width, p_height = get_width_height(prostate_original)
    print(f"\nOriginal prostate width  = {p_width:.3f}")
    print(f"Original prostate height = {p_height:.3f}")

    # One shared scale factor based on prostate width
    scale = target_prostate_width / p_width
    print(f"Shared scale factor = {scale:.6f}")

    # Center both contours using prostate centroid
    if do_center:
        # Get prostate centroid from sampled contour
        prostate_centroid = prostate_original.mean(axis=0)
        translation = -prostate_centroid * scale

        if do_flip_y:
            # First scale and flip, then translate
            prostate_tck = transform_bspline(prostate_tck, scale=scale, translate=(0, 0), flip_y=do_flip_y)
            urethra_tck = transform_bspline(urethra_tck, scale=scale, translate=(0, 0), flip_y=do_flip_y)
            # Then translate (after flip)
            translation[1] = -translation[1]
            prostate_tck = transform_bspline(prostate_tck, scale=1.0, translate=translation, flip_y=False)
            urethra_tck = transform_bspline(urethra_tck, scale=1.0, translate=translation, flip_y=False)
        else:
            prostate_tck = transform_bspline(prostate_tck, scale=scale, translate=translation, flip_y=False)
            urethra_tck = transform_bspline(urethra_tck, scale=scale, translate=translation, flip_y=False)

        print("Both B-splines transformed: scaled + centered using prostate centroid.")
    else:
        # Just scale without centering
        if do_flip_y:
            prostate_tck = transform_bspline(prostate_tck, scale=scale, translate=(0, 0), flip_y=do_flip_y)
            urethra_tck = transform_bspline(urethra_tck, scale=scale, translate=(0, 0), flip_y=do_flip_y)
        else:
            prostate_tck = transform_bspline(prostate_tck, scale=scale, translate=(0, 0), flip_y=False)
            urethra_tck = transform_bspline(urethra_tck, scale=scale, translate=(0, 0), flip_y=False)

    # Sample transformed contours for visualization and saving
    prostate_scaled = evaluate_bspline_contour(prostate_tck, num_sample_points)
    urethra_scaled = evaluate_bspline_contour(urethra_tck, num_sample_points)

    # Dimensions after transformation
    p_width_new, p_height_new = get_width_height(prostate_scaled)
    u_width_new, u_height_new = get_width_height(urethra_scaled)

    print(f"\nScaled prostate width  = {p_width_new:.3f} mm")
    print(f"Scaled prostate height = {p_height_new:.3f} mm")
    print(f"Scaled urethra width   = {u_width_new:.3f} mm")
    print(f"Scaled urethra height  = {u_height_new:.3f} mm")

    # Save transformed B-splines as .npz files
    # Note: The original u arrays are lost after transformation because u is tied to the original fitting,
    # but we can save the transformed B-spline without u since only tck is needed for evaluation
    save_bspline_npz(prostate_tck, prostate_out_npz, structure_name="prostate_transformed")
    save_bspline_npz(urethra_tck, urethra_out_npz, structure_name="urethra_transformed")

    # Plot together
    p_plot = close_contour(prostate_scaled)
    u_plot = close_contour(urethra_scaled)

    plt.figure(figsize=(7, 7))
    plt.plot(p_plot[:, 0], p_plot[:, 1], 'b-', linewidth=2, label="Prostate")
    plt.plot(u_plot[:, 0], u_plot[:, 1], 'r-', linewidth=2, label="Urethra")

    # Optional: Also plot control points to see the B-spline structure
    prostate_ctrl_pts = get_bspline_control_points(prostate_tck)
    urethra_ctrl_pts = get_bspline_control_points(urethra_tck)
    plt.plot(prostate_ctrl_pts[:, 0], prostate_ctrl_pts[:, 1], 'bo', markersize=4, alpha=0.5,
             label="Prostate control points")
    plt.plot(urethra_ctrl_pts[:, 0], urethra_ctrl_pts[:, 1], 'ro', markersize=4, alpha=0.5,
             label="Urethra control points")

    plt.gca().set_aspect('equal')
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.title("Scaled prostate and urethra contours (from B-spline)")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nFiles saved:")
    print(f"  - {prostate_out_npz}")
    print(f"  - {urethra_out_npz}")


if __name__ == "__main__":
    main()
