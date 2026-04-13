# Baseball Bat Vibration Simulation

In this repository, I simulate the **bat-ball collision** described in (Nathan, 2000) for the **standard** and **Torpedo** bat. I impact the bat with the ball across each bat's longitudinal axis to examine how the exit velocity of a baseball changes across the barrel. Then, I analyze the differences in fractional energy and collision time for both bats. 

This repository is structured to replicate the simulations for any bat or ball profiles. I create a class structure for both the bat and ball to allow easy flexibility across a range of bat shapes, stiffnesses, densities, as well as ball stiffness, non-linearity, and coefficient of restitutions (COR). 

The physics governing the bat and ball dynamics are explained in `report`.

## Repository Structure

```
├── run_simulation.py              # Main simulation: sweep exit velocity vs. impact location
├── scripts/                       # Core library modules
│   ├── __init__.py
│   ├── bat_class.py               # BatOsc & Ball classes: ODE integration, collision
│   ├── create_system_matrix.py    # System matrix H construction & eigenanalysis utilities
│   ├── integrators.py             # ODE integrators (RK4 and variants)
│   ├── plot_osc.py                # Geometry helpers: rotation, box rendering, bat visualisation
│   ├── unit_conversions.py        # Unit conversion utilities
│   └── old_rk4routine.py          # Legacy RK4 implementation
├── fourier_analysis.ipynb         # Fourier spectral analysis of bat vibrations
├── SimulationAnalysis.ipynb       # Main analysis notebook: eigenanalysis, collision, visualization
├── midterm_report/                # LaTeX source and working documents
│   ├── midterm_report.tex
│   ├── references.bib
│   ├── plots/                     # Figures used in the report
│   └── drafts/
├── plots/                         # Generated simulation plots
├── data/                          # *(ignored by .gitignore)*
├── docs/                          # *(ignored by .gitignore)*
├── results/                       # *(ignored by .gitignore)*
├── substack_plots/                # *(ignored by .gitignore)*
└── .gitignore
```

**Note:** Directories marked with *(ignored by .gitignore)* are not tracked in version control.

## Key Modules

### `scripts/bat_class.py` — `BatOsc` and `Ball` Classes

Class structure for the bat and ball. Encapsulates the full workflow:

**`BatOsc`**
1. **Initialize** with bat profile data and slice thickness: `BatOsc(bat_prof, dz)`
2. **Set material properties**: `set_bat_features(mass, rho, Y, S)` — also builds the mass matrix $M$ and precomputes $M^{-1}$
3. **Build or load the system matrix**: `get_H_matrix()` — accepts a CSV path, a NumPy array, or computes from scratch
4. **Validate setup**: `validate(require_inits, require_ball, impact_idx)` — raises a descriptive `ValueError` listing every missing step
5. **Set initial conditions**: `set_initial_conditions(state_vec)` — a $4N$-length vector $[y_0, \Phi_0, \dot{y}_0, \dot{\Phi}_0]$ with the initial conditions of each bat slice 
6. **Attach Ball** `bat.ball = ...` - initialize ball (as below) and attach to the bat for the collision
7. **Integrate** via:
   - `integrate_solution(t_span)` — free vibration
   - `integrate_with_ball(t_span, ball, impact_idx)` — full two-phase ball–bat collision (compression → expansion → free vibration), returns `dict` with `'yb'`, `'yb_dot'`, `'t_separation'`, `'max_u'`, `'k2'`, etc.
8. **Reset** between simulations: `reset()` — clears solution attributes for loop-safe re-use

You can also load a bat from a json file using the `bat_from_json` function. The function expects a path name to a json file in the form:

```
{
    "name": "standard", #unique bat name
    "profile_file": "path/to/bat/shape/profile/.dat", #path to the bat shape profile, expects array of tuples [(z_i, R_i)] in cylindrical coordinates
    "bat_length": 0.84, #in metres
    "mass": 0.9, #in kg
    "rho": 700, #density in kg/m^3
    "Y": 1.5e10, #Young's modulus in N/m^2
    "S": 1.0e9, #Shear modulus in N/m^2
    "dz": 0.01 #slice thickness in the bat profile
}
```

**`Ball`**
- Stores `mass`, `radius`, `initial_velocity`, `k1`, `alpha`, `e0`
- `compress(u)` — $F = k_1 u^\alpha$
- `expand(u, k2)` — $F = k_2 u^\alpha$ (energy-loss expansion phase)
- `get_k2(F_max, u_max)` — computes $k_2$ from COR and max compression state
- `reset()` — restores ball to initial state between simulations

You can also load a bat from a json file using the `bat_from_json` function. The function expects the path name to a json file in the form:

```
{
    "mass": 0.146, #ball mass in kg
    "radius": 0.0366, #ball radius in m
    "initial_velocity": 44.704, #initial ball velocity in m/s
    "e0": 0.50, #COR, dimensionless
    "k1": 0.714e7, #ball stiffness in N/m^alpha
    "alpha": 1.36 #non-linearity of the ball, dimensionless
}

```

### `run_simulation.py` — Exit Velocity Sweep Simulation

Self-contained simulation script that performs a complete parameter sweep across impact locations. The workflow includes:

1. **Interactive bat and ball selection** — prompts user to select standard and torpedo bat JSON files, and a ball JSON file from the `data/` directory
2. **Bat initialization** — loads both bats from JSON, builds system matrices $H$, and renders bat profile geometries
3. **Impact location sweep** — iterates over impact indices along the bat (default: 50 to end of bat)
4. **Collision integration** — for each impact location, runs `integrate_with_ball()` for both bat geometries simultaneously, computing exit velocity, max compression, max force, and collision time
5. **Results aggregation** — stores all results in pandas DataFrames and saves to CSV files in the `results/` directory
6. **Visualization** — generates plots of exit velocity vs. impact location comparing standard and torpedo geometries

Run interactively in VS Code or directly from terminal:
```bash
python run_simulation.py
```

Output CSVs are saved to `results/` with columns for impact index, final velocity (`vf`), maximum compression (`max_u`), maximum force (`max_F`), and collision time (`coll_t`).

### `scripts/create_system_matrix.py` — System Matrix Construction & Eigenanalysis

Builds the $2N \times 2N$ block-tridiagonal matrix $H$:

$$H = \begin{pmatrix} H_1 & H_2 \\ H_4 & H_3 \end{pmatrix}$$

where each block is an $N \times N$ tridiagonal matrix constructed from the material constants $\Lambda = S / (\rho \, dz^2)$ and $\Upsilon = Y / (\rho \, dz^2)$, and the cross-section geometry $A_i$, $I_i$. Free-free boundary conditions are applied at both ends of the bat.

See midterm report for a breakdown of the components of $\mathbf{H}$.

Also provides eigenanalysis and plotting utilities:
- `compute_eigenfrequencies(H)` — eigenvalues, eigenvectors, and natural frequencies as a DataFrame
- `find_mode_nodes(eig_df, zs, N)` — nodal positions (zero-crossings) for each mode shape
- `plot_eigenvalue_histogram(H)` — histogram of eigenvalue real parts on a symlog scale
- `plot_mode_shapes(eig_df, zs, N)` — eigenvector mode shapes with nodal markers
- `fourier_analysis(y_sol, t)` — FFT, peak-finding, centroid frequencies, spectrum & heatmap plots
- `compare_frequencies(H, reference_freqs)` — tabular comparison of computed vs. reference frequencies

### `scripts/plot_osc.py` — Visualization Utilities

Provides:
- `rotate(points, angle, centre)` — 2D rotation of point arrays
- `make_box(z, H, dz)` — creates rectangular box geometry for a bat slice
- `plot_bat_disp(zs, Ri, yi, phi_i)` — renders the deformed bat shape

### `scripts/integrators.py` — ODE Integration Methods

Contains numerical ODE solvers used for bat and ball dynamics:
- RK4 (Runge-Kutta 4th order) and variants
- Adaptive step-size control for accuracy
- Collision detection and phase transition handling

### `scripts/unit_conversions.py` — Unit Conversion Utilities

Helper functions for converting between different unit systems and physical quantities used throughout the simulation (e.g., velocity, force, compression, energy).

### `run_simulation.py` — Exit Velocity Sweep

Self-contained simulation script that:
1. Loads bat and ball parameters from JSON files in `data/`
2. Sweeps over impact locations along the bat
3. Runs `integrate_with_ball()` for both standard and torpedo bats at each location
4. Plots exit velocity vs. impact location for both geometries

Run directly or cell-by-cell in the VS Code interactive window:
```bash
python run_simulation.py
```

### Notebooks

- **`SimulationAnalysis.ipynb`** — Main analysis notebook. Covers static bat visualisation (standard + torpedo side-by-side), eigenmode analysis, free-vibration integration, Fourier analysis with frequency comparison against Alan Nathan's reference values, forced-vibration tests (constant & Gaussian pulses), full ball–bat collision integration with phase-by-phase output, and animation generation.
- **`fourier_analysis.ipynb`** — Spectral analysis and Fourier decomposition of bat vibration modes.


## Getting Started

### Prerequisites

- Python 3.8+
- NumPy, SciPy, Matplotlib, pandas, tqdm

Install dependencies:

```bash
pip install numpy scipy matplotlib pandas tqdm
```

### Quick Start

```python
import json
from bat_class import BatOsc, Ball
import numpy as np

# Load parameters from JSON
with open('data/bats/standard_bat_params.json') as f:
    p = json.load(f)

bat_prof = np.loadtxt(p['profile_file'])
bat = BatOsc(bat_prof, p['dz_m'])
bat.set_bat_features(p['mass_kg'], p['rho_kg_m3'], p['Y_N_m2'], p['S_N_m2'])
bat.get_H_matrix()
bat.validate()  # raises if anything is missing

# Set zero initial conditions
bat.set_initial_conditions(np.zeros(4 * bat.N))

# Set up ball
ball = Ball(v=58, e0=0.53, k1=6.53e7, alpha=1.84, mass=0.145, radius=0.0366)

# Run collision at impact slice 75
result = bat.integrate_with_ball(t_span=(0, 0.01), ball=ball, impact_idx=75)
print(f"Exit velocity: {result['yb_dot'][-1]:.2f} m/s")
print(f"Max compression: {result['max_u']*1e3:.2f} mm")
print(f"Separation time: {result['t_separation']*1e3:.3f} ms")
```


## References

- Alan M. Nathan, "Dynamics of the baseball–bat collision," *American Journal of Physics* 68, 979 (2000)
- Van Zandt, L.L., "The dynamical theory of the baseball bat," *American Journal of Physics* 60, 172 (1992)
