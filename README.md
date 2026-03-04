# Baseball Bat Vibration Simulation

A computational physics project that models the transverse vibration and oscillation dynamics of a baseball bat using Timoshenko beam theory. The bat is discretized into thin cross-sectional slices and the resulting coupled ODE system is integrated numerically to study how vibrations propagate along the bat after impact.

## Physics Overview

A baseball bat is modeled as a **non-uniform Timoshenko beam** — a tapered, free-free beam that accounts for both bending (Young's modulus $Y$) and shear deformation (shear modulus $S$). The bat is sliced into $N$ segments along its longitudinal axis, and for each slice $i$, the equations of motion couple two degrees of freedom:

- **$y_i(t)$** — transverse (vertical) displacement of slice $i$
- **$\Phi_i(t)$** — rotation angle of the cross-section at slice $i$

These are assembled into a state vector $\psi = [y_1, \dots, y_N, \Phi_1, \dots, \Phi_N]$, and the dynamics are governed by:

$$\ddot{\psi} = H \, \psi + F(t)$$

where $H$ is a $2N \times 2N$ system matrix encoding the material properties, geometry, and boundary conditions, and $F(t)$ is an optional external force (e.g., ball–bat collision).

### Bat Parameters

| Parameter | Symbol | Value |
|---|---|---|
| Bat length | $L$ | 0.84 m |
| Mass | $m$ | 0.885 kg |
| Wood density | $\rho$ | 649 kg/m³ |
| Young's modulus | $Y$ | $1.814 \times 10^{10}$ N/m² |
| Shear modulus | $S$ | $1.05 \times 10^{9}$ N/m² |
| Number of slices | $N$ | 84 |

The radius profile is loaded from empirical bat profile data (`data/r161.dat`), capturing the realistic tapered geometry of an R161-model bat.

## Repository Structure

```
├── bat_class.py              # BatOsc class: ODE integration, plotting, animation
├── create_system_matrix.py   # Constructs the system matrix H with boundary conditions
├── plot_osc.py               # Geometry helpers: rotation, box rendering, bat visualization
├── rk4routine.py             # Standalone script for quick RK4 integration & plotting
├── do_integration.ipynb      # Jupyter notebook for interactive integration & analysis
├── data/
│   ├── r161.dat              # Bat radius profile data (diameter vs. position)
│   ├── bat_profile_pts.txt   # Bat profile data points
│   ├── H_matrix.csv          # Computed system matrix H (full)
│   ├── H_matrix_Alan.csv     # Reference system matrix from Alan Nathan's model
│   └── H_matrix_nonzero.csv  # Sparse (non-zero entries only) representation of H
├── midterm_report/
│   ├── sample7.pdf           # Compiled midterm report
│   └── plots/                # Figures and animations used in the report
└── .gitignore
```

## Key Modules

### `bat_class.py` — `BatOsc` Class

The main simulation class. Encapsulates the full workflow:

1. **Initialize** with bat profile data and slice thickness
2. **Set material properties** via `set_bat_features(mass, rho, Y, S)`
3. **Build or load the system matrix** via `get_H_matrix()` — can compute from scratch or load a precomputed CSV
4. **Set initial conditions** via `set_initial_conditions(state_vec)` — a $4N$-length vector $[y_0, \Phi_0, \dot{y}_0, \dot{\Phi}_0]$
5. **Integrate** via `integrate_solution(t_span)` (free vibration) or `integrate_solution_with_collision(t_span, F)` (with external force)
6. **Visualize** via `plot_bat(time_idx)` for static snapshots or `animate_bat()` for full animations saved to `.mp4`

### `create_system_matrix.py` — System Matrix Construction

Builds the $2N \times 2N$ block-tridiagonal matrix $H$:

$$H = \begin{pmatrix} H_1 & H_2 \\ H_4 & H_3 \end{pmatrix}$$

where each block is an $N \times N$ tridiagonal matrix constructed from the material constants $\Lambda = S / (\rho \, dz^2)$ and $\Gamma = Y / (\rho \, dz^2)$, and the cross-section geometry $A_i$, $I_i$. Free-free boundary conditions are applied at both ends of the bat.

### `plot_osc.py` — Visualization Utilities

Provides:
- `rotate(points, angle, centre)` — 2D rotation of point arrays
- `make_box(z, H, dz)` — creates rectangular box geometry for a bat slice
- `plot_bat_disp(zs, Ri, yi, phi_i)` — renders the deformed bat shape

### `rk4routine.py` — Standalone Integration Script

A self-contained script that loads the bat profile, builds/loads the $H$ matrix, solves the ODE system using `scipy.integrate.solve_ivp` (RK45), and produces:
- Heatmap plots of $y(t, z)$ and $\Phi(t, z)$
- Animated centerline oscillation (`.mp4`)

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

## References

- Alan M. Nathan, "Dynamics of the baseball–bat collision," *American Journal of Physics* 68, 979 (2000)
- Van Zandt, L.L., "The dynamical theory of the baseball bat," *American Journal of Physics* 60, 172 (1992)
