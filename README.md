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
- Midterm report with derivations, figures, and error analysis

### To Do
- Coupled ball–bat collision using the nonlinear spring model ($F = k \, u^\alpha$)
- Ball exit velocity as a function of coefficient of restitution (COR)
- Energy-loss modelling for the collision (hysteresis curve)

## Physics Overview

A baseball bat is modeled as a **non-uniform Timoshenko beam**. We assume a tapered, free-free beam that accounts for both bending (Young's modulus $Y$) and shear deformation (shear modulus $S$). The bat is sliced into $N$ segments along its longitudinal axis, and for each slice $i$, the equations of motion couple two degrees of freedom:

- **$y_i(t)$** — transverse (vertical) displacement of slice $i$
- **$\Phi_i(t)$** — rotation angle of the cross-section at slice $i$

These are assembled into a state vector $\psi = [y_1, \dots, y_N, \Phi_1, \dots, \Phi_N]$, and the dynamics are governed by:

$$\ddot{\psi} = H \psi + F(t)$$

where $H$ is a $2N \times 2N$ system matrix encoding the material properties, geometry, and boundary conditions, and $F(t)$ is an optional external force (e.g., ball–bat collision).

### Bat Parameters

| Parameter | Symbol | Standard Bat | Torpedo Bat |
|---|---|---|---|
| Bat length | $L$ | 0.84 m | 0.85 m |
| Mass | $m$ | 0.885 kg | — |
| Wood density | $\rho$ | 649 kg/m³ | varies |
| Young's modulus | $Y$ | $1.814 \times 10^{10}$ N/m² | varies |
| Shear modulus | $S$ | $1.05 \times 10^{9}$ N/m² | varies |
| Number of slices | $N$ | 84 | 85 |

The radius profiles are loaded from empirical bat profile data (`data/r161.dat` and `data/torpedo.dat`), capturing the realistic tapered geometry.

## Repository Structure

```
├── bat_class.py              # BatOsc class: ODE integration, plotting, animation
├── create_system_matrix.py   # System matrix H construction & eigenanalysis utilities
├── plot_osc.py               # Geometry helpers: rotation, box rendering, bat visualisation
├── rk4routine.py             # Standalone script for quick RK4 integration & plotting
├── do_integration.ipynb      # Main notebook: integration, eigenanalysis, Fourier, forced vibration
├── testing_H.ipynb           # H matrix validation against Alan Nathan's reference
├── data/
│   ├── r161.dat              # Standard bat radius profile
│   ├── torpedo.dat           # Torpedo bat radius profile
│   ├── bat_profile_pts.txt   # Bat profile data points
│   ├── H_matrix.csv          # Computed system matrix H (full)
│   ├── H_matrix_Alan.csv     # Alan Nathan's reference H (standard bat, sparse CSV)
│   ├── H_matrix_torpedo.csv  # Alan Nathan's reference H (torpedo bat, sparse CSV)
│   ├── H_matrix_nava.csv     # Nava's computed H (standard bat)
│   ├── H_matrix_nava_torpedo.csv # Nava's computed H (torpedo bat)
│   ├── H_matrix_nonzero.csv  # Sparse (non-zero entries only) representation of H
│   ├── hMatrix_torpedo.txt   # Raw torpedo H matrix data (text format)
│   └── standard_bat_eigenvalues.csv  # Saved eigenvalues for the standard bat
├── midterm_report/
│   ├── midterm_report.tex    # LaTeX source
│   ├── plots/                # Figures used in the report
│   └── ...
└── .gitignore
```

## Key Modules

### `bat_class.py` — `BatOsc` and `Ball` Classes

The main simulation class. Encapsulates the full workflow:

1. **Initialize** with bat profile data and slice thickness
2. **Set material properties** via `set_bat_features(mass, rho, Y, S)`
3. **Build or load the system matrix** via `get_H_matrix()` — accepts a CSV path, a NumPy array, or computes from scratch
4. **Set initial conditions** via `set_initial_conditions(state_vec)` — a $4N$-length vector $[y_0, \Phi_0, \dot{y}_0, \dot{\Phi}_0]$
5. **Integrate** via `integrate_solution(t_span)` (free vibration) or `integrate_solution_with_collision(t_span, F)` (with external force)
6. **Visualize** via `plot_bat(time_idx)` for static snapshots or `animate_bat()` for full animations saved to `.mp4`

Also provides:
- `F_quad(u, k, alpha)` — quadratic (nonlinear spring) force model for ball compression
- `Ball` class (stub) — stores `mass`, `radius`, `initial_velocity`, `k1`, `alpha` for future ball–bat coupling

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

### `rk4routine.py` — Standalone Integration Script

A self-contained script that loads the bat profile, builds/loads the $H$ matrix, solves the ODE system using `scipy.integrate.solve_ivp` (RK45), and produces:
- Heatmap plots of $y(t, z)$ and $\Phi(t, z)$
- Animated centerline oscillation (`.mp4`)

### Notebooks

- **`do_integration.ipynb`** — Main analysis notebook. Covers static bat visualisation (standard + torpedo side-by-side), eigenmode analysis, free-vibration integration, Fourier analysis with frequency comparison against Alan Nathan's reference values, forced-vibration tests (constant & Gaussian pulses), and animation generation.
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
from bat_class import BatOsc
import numpy as np

# Load bat profile
bat_prof = np.loadtxt('data/r161.dat')
bat_length = 0.84  # meters
N = bat_prof.shape[0]
dz = bat_length / N

# Create bat object
bat = BatOsc(bat_prof, dz)
bat.set_bat_features(mass=0.885, rho=649, Y=1.814e10, S=1.05e9)
bat.get_H_matrix('data/H_matrix_Alan.csv')

# Set initial conditions: small displacement at slice 59
y0 = np.zeros(N)
y0[58] = 0.01  # 1 cm displacement
state = np.concatenate([y0, np.zeros(N), np.zeros(N), np.zeros(N)])
bat.set_initial_conditions(state)

# Integrate for 5 ms
sol = bat.integrate_solution((0, 0.005))

# Visualize
bat.plot_bat(time_idx=500, exaggerate=100)
bat.animate_bat(exaggerate=100, interval=10, path='vibration.mp4')
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
