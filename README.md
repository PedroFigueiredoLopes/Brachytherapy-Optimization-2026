# Brachytherapy Optimization
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

> **Academic Project Disclaimer**  
> This repository contains the **brachytherapy optimisation component** developed as part of a class project on nonlinear engineering optimisation (Otimização Não-linear em Engenharia, Universidade de Aveiro, 2026).  
> The implementation is **not clinically validated** and should not be used for real treatment planning. It is a 2D academic simplification intended to demonstrate the application of metaheuristic algorithms (CMA-ES and Black Hole Algorithm) and other thecniques to a physics-based dose optimisation problem.


## The Full Project Report

The complete project report contains the mathematical formulation, sensitivity analysis, benchmark tests (13-bar truss, analytical functions), and a detailed discussion of results for **both** the brachytherapy problem and the bulletproof vest problem.

The full report (Metaheuristic Optimisation of Nonlinear Engineering Problems) is part of a larger compendium with:
- DOI: https://doi.org/10.48528/48c3-x460
- ISBN: 978-989-9253-81-0

> Note: The DOI may not be working due to recent cyber attack that affected University of Aveiro's institutional repositor.

*This repository hosts only the code for the brachytherapy module. The bulletproof vest model (which used FEniCSx and Duffing oscillators) is described in the report and its code is available in a [separate repository](https://github.com/PedroFigueiredoLopes/Bulletproof-Vest-Optimization-2026).*


## Repository Structure

| File | Description |
| :--- | :--- |
| `main.py` | Main orchestration script. Runs multistart optimisation (CMA-ES or Black Hole). |
| `brachytherapy_optimization.py` | Core class: evaluates objective function, clinical constraints, and visualises dose maps. |
| `tg_43.py` | TG-43 line-source dose calculator. Loads radial dose data from an external Excel file. |
| `cma_es.py` | Standalone implementation of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES). |
| `black_hole.py` | Standalone implementation of the Black Hole Algorithm. |
| `distance_field_precomputation.py` | Generates Signed Distance Field (SDF) maps from B-spline contours for fast geometry queries. |
| `scaling_bsplines.py` | Scales and centres B-spline contours to physical dimensions (metres). |
| `bsplines_from_png.py` | Interactive GUI tool to fit closed B-splines to contours traced on a `.png` image. |
| `benchmark.py` | Script used to benchmark integration methods (MC, Sobol, grid, importance sampling) to justify the choice of Sobol integration in the report. |

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for fast dependency management. To set up the environment:

```bash
uv sync
```

This installs all dependencies (`matplotlib`, `scipy`, `pandas`, `xlrd`) defined in `pyproject.toml`.

*Alternatively, if you prefer `pip`*:
```bash
pip install matplotlib pandas scipy xlrd
```

## Data Preparation

### 1. TG-43 Dose Data (Required)
The dose calculator in `tg_43.py` requires a radial dose function file (e.g., `192ir-hdr_varianclassic.xls`).  
- **This file is not included** in the repository.  
- It is publicly available from standard libraries (e.g., ESTRO, IAEA, IROC). And it is the generic found [here](https://www.estro.org/About/ESTRO-Organisation-Structure/Committees/GEC-ESTRO-Committee/GEC-ESTRO-BRAPHYQS/Ir-192-HDR).  
- Place the file in the **root directory**, or modify the path in `tg_43.py`.
- Adapt the `load_dose_calculator` function to use the correct rows, cols, ...

### 2. Contour & SDF Generation (Required)
This repository does not include precomputed SDF maps to keep it lightweight. You must generate them for the prostate and urethra:

1. Place a `prostate.png` image (or your own anatomy slice) in the root directory.
2. Run `python bsplines_from_png.py` to click points and fit closed B-splines.
3. Run `python scaling_bsplines.py` to scale the splines to metres (adjust `target_prostate_width` inside the script).
4. Run `python distance_field_precomputation.py` **twice**: once for the prostate and once for the urethra.  
   - Change the `file_name_out` variable inside the script to `"prostate_sdf.npz"` and `"urethra_sdf.npz"` respectively.
   - Change the input B-spline path in the second call to `load_bspline_npz()` to match the correct contour.


## Running the Optimisation

Once the TG-43 data and SDF maps are in place, configure the optimisation by editing the global variables at the top of `main.py`:

```python
ALGORITHM        = "cma_es"       # "cma_es" or "black_hole"
SOURCE_COUNT     = 6
N_SAMPLES        = 2 ** 14        # Sobol integration points
MAX_ITERATIONS   = 1000
PRESCRIBED_DOSE  = 5              # Gy
```

Then execute:
```bash
python main.py
```
### Customisation Options in `main.py`

- **Loading saved results:** `main.py` allows you to load previously saved results instead of running a new optimisation. Uncomment the appropriate lines in the `if __name__ == '__main__':` block to enable this.
- **Disabling the animation:** The optimisation generates an animated GIF (`mean_trajectory.gif`) showing the evolution of the CMA-ES mean solution. To disable this, comment out the line `create_mean_trajectory_animation(problem, best_history)` in the main function body.
- **Changing the random seed:** A master seed controls reproducibility across multistart runs. To change it, modify the seed value (default `42`) in the line `master_rng = np.random.default_rng(42)`.
- **Note on animations:** The animation feature was implemented specifically for CMA-ES. It can be adapted for the Black Hole Algorithm by modifying the `create_mean_trajectory_animation` function.

> **Performance note:** The optimisation is computationally intensive (optimising ~30 variables with numerical integration). For a quick smoke test, reduce `MAX_ITERATIONS` to `1000` and `N_SAMPLES` to `2**10`.


### Expected outputs:
- Console logs with convergence progress and multistart leaderboard.
- Matplotlib figures: convergence history and final dose distribution.
- An animated GIF (`mean_trajectory.gif`) showing the evolution of the CMA-ES mean solution (if enabled).

## Development Notes & Transparency

This project was developed within a constrained academic timeline. The following notes provide context for reviewers:

- **Auxiliary AI assistance:** AI was used throughout the development, with the `benchmark.py` script (used to select the integration method) being almost completely AI generated.

- **Performance:** The codebase prioritises **clarity and modularity** over performance optimisation. It is intended to demonstrate the implementation of metaheuristic algorithms and physics-based modelling, not to serve as a production-grade treatment planning system.

## Acknowledgments

- **Collaborator:** João Gonçalo Pereira Lopes - contributions to development, algorithm testing, mathematical modelling, and discussions.
- **Python Ecosystem:** Built with NumPy, SciPy, Matplotlib, and the broader open-source community.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
