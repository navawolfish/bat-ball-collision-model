# Baseball Bat Vibration Simulation

A computational physics project that models the transverse vibration and oscillation dynamics of a baseball bat according to Alan Nathan (2000). The bat is discretized into thin cross-sectional slices and the resulting coupled ODE system is integrated numerically to study how vibrations propagate along the bat after impact. Two bat geometries are compared: a **standard R161 bat** ($N = 84$) and the **Torpedo bat** ($N = 85$).

## Current Progress

### Completed
- System matrix $H$ construction with free-free boundary conditions for both bat geometries
- Validation of $H$ against Alan Nathan's reference matrices (element-wise relative error analysis)
- Eigenmode analysis: eigenfrequencies, mode shapes, and nodal positions
- Free-vibration integration via RK45
- Heatmap visualisation of $y(t, z)$ and $\Phi(t, z)$
- Fourier analysis of displacement solutions with peak-finding and centroid frequency extraction
- Forced-vibration integration with constant and Gaussian force pulses at the impact slice
- Animated bat oscillation (`.mp4`)
- **Coupled ball–bat collision** using the two-phase nonlinear spring model ($F = k_1 u^\alpha$ compression, $F = k_2 u^\beta$ expansion)
- **Ball exit velocity sweep** across all impact locations for standard and torpedo bats
- **`bat_ball_collision_model.py`** — self-contained simulation script comparing standard vs. torpedo bat performance
- JSON parameter files for reproducible bat and ball configuration
- `BatOsc.validate()` for clear setup error messages
- `BatOsc.reset()` and `Ball.reset()` for loop-safe re-integration
- Midterm report with derivations, figures, and error analysis

### To Do
- Energy-loss modelling / hysteresis curve visualisation
- Uncertainty/sensitivity analysis over ball parameters ($k_1$, $\alpha$, COR)

## Physics Overview

A baseball bat is modeled as a **non-uniform Timoshenko beam**. We assume a tapered, free-free beam that accounts for both bending (Young's modulus $Y$) and shear deformation (shear modulus $S$). The bat is sliced into $N$ segments along its longitudinal axis, and for each slice $i$, the equations of motion couple two degrees of freedom:

- **$y_i(t)$** — transverse (vertical) displacement of slice $i$
- **$\Phi_i(t)$** — rotation angle of the cross-section at slice $i$

These are assembled into a state vector $\psi = [y_1, \dots, y_N, \Phi_1, \dots, \Phi_N]$, and the dynamics are governed by:

$$\ddot{\psi} = H \psi + M^^{-1}F(t)$$

where $H$ is a $2N \times 2N$ system matrix encoding the material properties, geometry, and boundary conditions, and $F(t)$ is an optional external force (e.g., ball–bat collision).

### Bat Parameters

| Parameter | Symbol | Standard Bat | Torpedo Bat |
|---|---|---|---|
| Bat length | $L$ | 0.84 m | 0.85 m |
| Mass | $m$ | 0.885 kg | 0.907 kg |
| Wood density | $\rho$ | 649 kg/m³ | 690 kg/m³|
| Young's modulus | $Y$ | $1.814 \times 10^{10}$ N/m² | $1.65 \times 10^{10}$ N/m² |
| Shear modulus | $S$ | $1.05 \times 10^{9}$ N/m² | $0.90 \times 10^{9}$ N/m²|
| Number of slices | $N$ | 84 | 85 |

The radius profiles are loaded from empirical bat profile data (`data/r161.dat` and `data/torpedo.dat`), capturing the realistic tapered geometry.

## Repository Structure

```
├── bat_class.py                  # BatOsc & Ball classes: ODE integration, collision, plotting
├── create_system_matrix.py       # System matrix H construction & eigenanalysis utilities
├── plot_osc.py                   # Geometry helpers: rotation, box rendering, bat visualisation
├── bat_ball_collision_model.py   # Sweep script: exit velocity vs. impact location (standard vs. torpedo)
├── do_integration.ipynb          # Main notebook: integration, eigenanalysis, Fourier, forced/ball collision
├── substack.ipynb                # Supplementary notebook: standing wave exploration
├── testing_H.ipynb               # H matrix validation against Alan Nathan's reference
├── data/
│   ├── r161.dat                  # Standard bat radius profile
│   ├── torpedo.dat               # Torpedo bat radius profile
│   ├── bat_profile_pts.txt       # Bat profile data points
│   ├── standard_bat_params.json  # Standard bat parameters (mass, rho, Y, S, dz)
│   ├── torpedo_bat_params.json   # Torpedo bat parameters
│   ├── ball_params.json          # Ball parameters (v_init, e0, k1, alpha, mb, Rb)
│   ├── H_matrix.csv              # Computed system matrix H (full)
│   ├── H_matrix_Alan.csv         # Alan Nathan's reference H (standard bat, sparse CSV)
│   ├── H_matrix_torpedo.csv      # Alan Nathan's reference H (torpedo bat, sparse CSV)
│   ├── H_matrix_nava.csv         # Nava's computed H (standard bat)
│   ├── H_matrix_nava_torpedo.csv # Nava's computed H (torpedo bat)
│   ├── H_matrix_nonzero.csv      # Sparse (non-zero entries only) representation of H
│   ├── hMatrix_torpedo.txt       # Raw torpedo H matrix data (text format)
│   └── standard_bat_eigenvalues.csv  # Saved eigenvalues for the standard bat
├── midterm_report/
│   ├── midterm_report.tex        # LaTeX source
│   ├── plots/                    # Figures used in the report
│   └── ...
└── .gitignore
```

## Key Modules

### `bat_class.py` — `BatOsc` and `Ball` Classes

The main simulation classes. Encapsulates the full workflow:

**`BatOsc`**
1. **Initialize** with bat profile data and slice thickness: `BatOsc(bat_prof, dz)`
2. **Set material properties**: `set_bat_features(mass, rho, Y, S)` — also builds the mass matrix $M$ and precomputes $M^{-1}$
3. **Build or load the system matrix**: `get_H_matrix()` — accepts a CSV path, a NumPy array, or computes from scratch
4. **Validate setup**: `validate(require_inits, require_ball, impact_idx)` — raises a descriptive `ValueError` listing every missing step
5. **Set initial conditions**: `set_initial_conditions(state_vec)` — a $4N$-length vector $[y_0, \Phi_0, \dot{y}_0, \dot{\Phi}_0]$
6. **Integrate** via:
   - `integrate_solution(t_span)` — free vibration
   - `integrate_solution_with_collision(t_span, F)` — with external force function $F(t)$
   - `integrate_with_ball(t_span, ball, impact_idx)` — full two-phase ball–bat collision (compression → expansion → free vibration), returns `dict` with `'yb'`, `'yb_dot'`, `'t_separation'`, `'max_u'`, `'k2'`, etc.
7. **Reset** between simulations: `reset()` — clears solution attributes for loop-safe re-use

**`Ball`**
- Stores `mass`, `radius`, `initial_velocity`, `k1`, `alpha`, `e0`
- `compress(u)` — $F = k_1 u^\alpha$
- `expand(u, k2)` — $F = k_2 u^\alpha$ (energy-loss expansion phase)
- `get_k2(F_max, u_max)` — computes $k_2$ from COR and max compression state
- `reset()` — restores ball to initial state between simulations

Also provides:
- `F_quad(u, k, alpha)` — standalone quadratic force model

### `create_system_matrix.py` — System Matrix Construction & Eigenanalysis

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

### `plot_osc.py` — Visualization Utilities

Provides:
- `rotate(points, angle, centre)` — 2D rotation of point arrays
- `make_box(z, H, dz)` — creates rectangular box geometry for a bat slice
- `plot_bat_disp(zs, Ri, yi, phi_i)` — renders the deformed bat shape

### `bat_ball_collision_model.py` — Exit Velocity Sweep

Self-contained simulation script that:
1. Loads bat and ball parameters from JSON files in `data/`
2. Sweeps over impact locations along the bat
3. Runs `integrate_with_ball()` for both standard and torpedo bats at each location
4. Plots exit velocity vs. impact location for both geometries

Run directly or cell-by-cell in the VS Code interactive window:
```bash
python bat_ball_collision_model.py
```

### Notebooks

- **`do_integration.ipynb`** — Main analysis notebook. Covers static bat visualisation (standard + torpedo side-by-side), eigenmode analysis, free-vibration integration, Fourier analysis with frequency comparison against Alan Nathan's reference values, forced-vibration tests (constant & Gaussian pulses), full ball–bat collision integration with phase-by-phase output, and animation generation.
- **`substack.ipynb`** — Supplementary notebook for standing wave exploration on the cylindrical bat model, including jshtml animations.
- **`testing_H.ipynb`** — Validates the independently constructed $H$ matrices against Alan Nathan's reference for both bat geometries. Computes element-wise relative errors, identifies discrepant entries by matrix quadrant, and generates publication-quality error plots saved to `midterm_report/plots/`.

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
with open('data/standard_bat_params.json') as f:
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

### Eigenanalysis Quick Start

```python
from create_system_matrix import compute_eigenfrequencies, plot_mode_shapes, find_mode_nodes

eig_df = compute_eigenfrequencies(bat.H, num_modes=10)
nodes = find_mode_nodes(eig_df, bat.zs, bat.N, num_modes=10)
plot_mode_shapes(eig_df, bat.zs, bat.N, num_modes=10, nodes=nodes)
```

## References

- Alan M. Nathan, "Dynamics of the baseball–bat collision," *American Journal of Physics* 68, 979 (2000)
- Van Zandt, L.L., "The dynamical theory of the baseball bat," *American Journal of Physics* 60, 172 (1992)
