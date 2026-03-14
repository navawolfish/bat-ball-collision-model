#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

#my imports
from create_system_matrix import create_system_matrices, load_H_matrix
from plot_osc import animate_bat, plot_batsol_heatmap, plot_bat_disp, make_box, rotate, plot_bat
from bat_class import BatOsc, Ball
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
    'axes.grid': True
})

colors = plt.get_cmap('tab10').colors
#turn colors into a list
colors = [colors[i] for i in range(len(colors))]
colors = ['#EFA00B', '#439775', '#4B4E6D', '#6A4C93', '#FAC8CD', '#9BC1BC', '#5D737E', '#D9BF77', '#ACD8AA', '#FFE156']

#%% Step 1: Set Bat Features

## STANDARD BAT ##
standard_file = 'data/standard_bat_params.json' #replace as necessary

with open(standard_file, 'r') as f:
    params_standard = json.load(f)
#load bat profile
bat_profile_standard = np.loadtxt(params_standard['profile_file'])
#convert profile to SI units (m)
bat_profile_standard[:, 0] *= 1e-2 #convert z from cm to m
bat_profile_standard[:, 1] *= 1e-3/2 #convert diameter from mm to m, convert to radius

#create bat instance
bat_standard = BatOsc(bat_profile_standard, params_standard['dz']) #initialize with profile and dz
bat_standard.set_bat_features(params_standard['mass'], params_standard['rho'], params_standard['Y'], params_standard['S']) #set features with mass, density, Young's modulus, and damping coefficient
bat_standard.get_H_matrix() # build system matrix

#plot the bat
plot_bat(bat_standard, title = 'Standard Bat Profile')
#%%
## TORPEDO BAT ##
torpedo_file = 'data/torpedo_bat_params.json' #replace as necessary
with open(torpedo_file, 'r') as f:
    params_torpedo = json.load(f)
#load bat profile
bat_profile_torpedo = np.loadtxt(params_torpedo['profile_file'])

#convert profile to SI units (m)
bat_profile_torpedo[:, 0] *= 1e-2 #convert z from cm to m
bat_profile_torpedo[:, 1] *= 1e-3/2 #convert diameter from mm to m, convert to radius

#create bat instance
bat_torpedo = BatOsc(bat_profile_torpedo, params_torpedo['dz']) #initialize with profile and dz
bat_torpedo.set_bat_features(params_torpedo['mass'], params_torpedo['rho'], params_torpedo['Y'], params_torpedo['S']) #set features with mass, density, Young's modulus, and damping coefficient
bat_torpedo.get_H_matrix() # build system matrix
#plot the bat
plot_bat(bat_torpedo, title = 'Torpedo Bat Profile')
# %% Now initialize the ball
ball_param_file = 'data/ball_params_lowv.json' #replace as necessary

with open(ball_param_file, 'r') as f:
    ball_params = json.load(f)

#ball takes params v, e0, k1, alpha, mb, Rb
ball = Ball(ball_params['v_init'], ball_params['e0'], ball_params['k1'], ball_params['alpha'],ball_params['mb'], ball_params['Rb'])

#attach the ball to the bats for easy access during simulation
bat_standard.ball = ball
bat_torpedo.ball = ball
# %% Set up time span for simulation
tspan = (0, 0.01) #time span for simulation
N_time = 10000
t_eval = np.linspace(tspan[0], tspan[1], N_time) #time points to evaluate at


#%% Set up array of impact locations
impact_idcs = np.arange(45, len(bat_standard.zs), 1) #every 2nd point along the bat, starting from the 10th point to avoid the handle
# %% Now perform the simulations for each impact location and store results
max_vel_s = np.zeros_like(impact_idcs, dtype=object)
max_vel_t = np.zeros_like(impact_idcs, dtype=object)

results = {}
for i, impact_idx in tqdm(enumerate(impact_idcs), disable=False):
    # print(f"Simulating impact at index {impact_idx} (z = {bat_standard.zs[impact_idx]:.3f} m)")

    #reset ball and bats to initial conditions before each simulation
    ball.reset()
    bat_standard.reset()
    bat_torpedo.reset()

    bat_standard.set_initial_conditions(np.zeros(4 * bat_standard.N))
    bat_torpedo.set_initial_conditions(np.zeros(4 * bat_torpedo.N))

    #simulate for standard bat
    # print("Simulating standard bat...")
    sol_standard = bat_standard.integrate_with_ball(tspan, ball, impact_idx, t_eval = t_eval)
    max_vel_s[i] = sol_standard['yb_dot'][-1]
    # print("Exit velocity (standard bat): {:.2f} m/s".format(sol_standard['yb_dot'][-1]))

    #simulate for torpedo bat
    # print("Simulating torpedo bat...")
    sol_torpedo = bat_torpedo.integrate_with_ball(tspan, ball, impact_idx, t_eval = t_eval)
    max_vel_t[i] = sol_torpedo['yb_dot'][-1]

    # print("Exit velocity (torpedo bat): {:.2f} m/s".format(sol_torpedo['yb_dot'][-1]))
    #store results
    results[impact_idx] = {
        'standard': sol_standard,
        'torpedo': sol_torpedo
    }

#%%plot
def mps_to_mph(v):
    return v * 2.23694

fig, ax = plt.subplots(figsize=(10, 6))
plt.title(r'Exit Velocity vs Impact Location, $v_i = %s$ m/s' % ball.initial_velocity)
plt.plot(impact_idcs *1e-2, max_vel_s, label='Standard Bat', marker='o', color = colors[0])
plt.plot(impact_idcs *1e-2, max_vel_t, label='Torpedo Bat', marker='s', color = colors[1])
ax.set_xlabel('Impact Location along Bat (m)')
ax.set_ylabel('Maximum Velocity (m/s)', color='black')
ax.set_ylim(min(np.min(max_vel_s), np.min(max_vel_t)) * 0.9, max(np.max(max_vel_s), np.max(max_vel_t)) * 1.1)
plt.legend()
# Add secondary y-axis on right with mph scale
ax2 = ax.twinx()
ax2.set_ylabel('Maximum Velocity (mph)', color='black')
ymin_mps, ymax_mps = ax.get_ylim()
ax2.set_ylim(mps_to_mph(ymin_mps), mps_to_mph(ymax_mps))
ax2.grid(False)

plt.savefig("impact_location_vs_exit_velocity.pdf")
plt.show()

# %%
