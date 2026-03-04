#%% Imports
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os 
import tqdm
from scipy.sparse import diags

from create_system_matrix import create_system_matrices, load_H_matrix
#%% PLOT SETTINGS
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'grid.alpha': 0.3,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'axes.facecolor': '#F1F5F2',
})

colors = plt.get_cmap('tab10').colors
#turn colors into a list
colors = [colors[i] for i in range(len(colors))]
colors = ['#EFA00B', '#439775', '#4B4E6D', '#6A4C93', '#FAC8CD', '#9BC1BC', '#5D737E', '#D9BF77', '#ACD8AA', '#FFE156']
#%% Bat profile
#params
bat_length = 0.84 #in m
mass = 0.885 # in kg
rho = 649 # in kg/m^3
Y = 1.814 * 1e10 # in N/m^2
S = 1.05 * 1e9 # in N/m^2

# %% Fit radius profile
# load
bat_prof= np.loadtxt('data/r161.dat') #diameter
N = bat_prof.shape[0]  # number of slices
dz = bat_length / N # slice thickness in m
Ri = bat_prof[:, 1] * 1e-3 / 2  # in m 

#%% Plot bat
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(bat_prof[:, 0], Ri, c=colors[1], s=30, label='Data Points')

ax.set_title('Bat Radius Profile', fontsize=14)
ax.set_xlabel('z (cm)', fontsize=12)
ax.set_ylabel('R (m)', fontsize=12)
# ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

os.makedirs('plots', exist_ok=True) # make directory for plots if it doesn't exist
plt.savefig('plots/bat_radius_profile.png', dpi=300) # save figure
plt.show()



#%% Create array A for cross-sectional area
Ai = np.pi * Ri**2  # in m^2
Vi = Ai * dz  # in m^3
Ii = 4/np.pi * Ri**4  # in m^4


#%% Time span
t_span = (0, 0.00501) # from 0 to 5 ms
t_eval = np.linspace(t_span[0], t_span[1], 20000) # time evaluation points

pbar = tqdm.tqdm(total=1.2e5, desc="Overall Progress", position=0)
#%% Create the Matrix H
# H = create_system_matrices(N, Ai, Ii, dz, S, Y, rho)
H = load_H_matrix('data/H_matrix_Alan.csv', N)
#%% System ODE
def new_bat_ode(t, x, H = H, N = N, pbar = pbar):
    """
    ODE system for bat vibration using system matrix H

    :param x: state vector (displacement and velocity)
    """

    #call progress bar update
    pbar.update(1)
    # Unpack state vector: x = [y, Phi, y_dot, Phi_dot]
    y = x[0:N]
    Phi = x[N:2*N]
    y_dot = x[2*N:3*N]
    Phi_dot = x[3*N:4*N]

    # Stack velocities for H
    psi = np.concatenate([y, Phi])  # shape (2N,)
    # Compute accelerations using H
    psi_ddot = H.dot(psi)  # shape (2N,)
    y_ddot = psi_ddot[0:N]
    Phi_ddot = psi_ddot[N:2*N]

    # Return derivative: d/dt [y, Phi, y_dot, Phi_dot] = [y_dot, Phi_dot, y_ddot, Phi_ddot]
    return np.concatenate([y_dot, Phi_dot, y_ddot, Phi_ddot])
# %% Run the solver
# Initial conditions
y0 = np.zeros(N)
y_dot0 = np.zeros(N)
Phi0 = np.zeros(N)
Phi_dot0 = np.zeros(N)

#at t=0, give a small force at the 59th slice
y0[58] = 0.1  # 1 cm initial displacement at slice 59
Phi0[58] = 0.00  # 1 cm initial angular displacement at slice 59

# Pack initial state
x0 = np.concatenate([y0, Phi0, y_dot0, Phi_dot0])

# Solve
sol = solve_ivp(new_bat_ode, t_span, x0, method="RK45",
                rtol=1e-6,
atol=1e-9
)

# Unpack solution
y_sol = sol.y[0:N, :]
Phi_sol = sol.y[N:2*N, :]

#%% show results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

im0 = axs[0].imshow(y_sol, aspect='auto', extent=[t_span[0]*1e3, t_span[1]*1e3, 0, bat_length], origin='lower', cmap='plasma')
im1 = axs[1].imshow(Phi_sol, aspect='auto', extent=[t_span[0]*1e3, t_span[1]*1e3, 0, bat_length], origin='lower', cmap='plasma')
axs[0].set_title('Solution for $y(t, z)$')
axs[1].set_title(r'Solution for $\Phi(t, z)$')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Position along Bat (m)')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Position along Bat (m)')
plt.colorbar(im1, ax=axs[1], label='Angular Displacement (rad)')
plt.colorbar(im0, ax=axs[0], label='Displacement (m)')
plt.tight_layout()
# %%
plt.imshow(np.log(my_bat.H), cmap='viridis')
plt.colorbar(label='log(H values)')
# %%

# Exaggeration factor for visualization
EXAGGERATE = 100  # Increase or decrease as needed

# Bat profile (z positions and radii)
z_cm = bat_prof[:, 0]  # in cm
z = z_cm * 1e-2        # convert to meters for plotting
R = Ri                 # already in meters

# Choose time steps to animate (e.g., every 100th frame)
frames = np.arange(0, y_sol.shape[1], 100)

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], 'o-', color=colors[1], lw=3)
ax.set_xlim(z[0], z[-1])
ax.set_ylim(-0.1, 0.1)  # Adjust as needed for exaggeration
ax.set_xlabel('z (m)')
ax.set_ylabel('y (m)')
ax.set_title('Bat Oscillation Animation (Exaggerated)')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    t_idx = frames[i]
    # Exaggerated displacement
    y_disp = y_sol[:, t_idx] * EXAGGERATE
    phi_disp = Phi_sol[:, t_idx] * EXAGGERATE

    # Deformed centerline: y(z) + phi(z)*0 (for small angles, phi is rotation)
    # For visualization, you can show the centerline as:
    y_deformed = y_disp  # or y_disp + R * np.sin(phi_disp) for more effect

    line.set_data(z, y_deformed)
    ax.set_title(f'Bat Oscillation Animation (t={t_eval[t_idx]*1e3:.2f} ms)')
    return line,

ani = animation.FuncAnimation(
    fig, animate, frames=len(frames), init_func=init, blit=True, interval=30
)
ani.save('bat_oscillation.mp4', fps=30)
plt.show()



# %% Make bat osc

bat_osc = BatOsc(bat_prof, dz)
time_idx = 0  # choose a time index to plot
top_lefts = bat_osc.plot_bat(y_sol, Phi_sol, time_idx)

# %%
from bat_class import BatOsc

# %%
my_bat = BatOsc(bat_prof, dz)
my_bat.set_bat_features(mass, rho, Y, S)
my_bat.get_H_matrix('data/H_matrix_Alan.csv')
state_vec = np.concatenate([y0, Phi0, y_dot0, Phi_dot0])
my_bat.set_initial_conditions(state_vec)
sol = my_bat.integrate_solution(t_span, t_eval)
my_bat.plot_bat(time_idx=1000)
# %%
plt.figure(figsize=(10, 6))
plt.imshow(np.log(my_bat.y_sol), cmap='viridis')
 # %%
y_sol = my_bat.y_sol
Phi_sol = my_bat.phi_sol

#%% show results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

im0 = axs[0].imshow(y_sol, aspect='auto', extent=[t_span[0]*1e3, t_span[1]*1e3, 0, bat_length], origin='lower', cmap='plasma')
im1 = axs[1].imshow(Phi_sol, aspect='auto', extent=[t_span[0]*1e3, t_span[1]*1e3, 0, bat_length], origin='lower', cmap='plasma')
axs[0].set_title('Solution for $y(t, z)$')
axs[1].set_title(r'Solution for $\Phi(t, z)$')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Position along Bat (m)')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Position along Bat (m)')
plt.colorbar(im1, ax=axs[1], label='Angular Displacement (rad)')
plt.colorbar(im0, ax=axs[0], label='Displacement (m)')
plt.tight_layout()
# %%
