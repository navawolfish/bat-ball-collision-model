#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import os
import pandas as pd

#my imports
from create_system_matrix import create_system_matrices, load_H_matrix
from plot_osc import animate_bat, plot_batsol_heatmap, plot_bat_disp, make_box, rotate, plot_bat, plot_ball_forces
from bat_class import BatOsc, Ball, bat_from_json, ball_from_json
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
make_plots = True

#%% Set global vars for the simulation
DATA_PATH = "data/"
all_bats = os.listdir(os.path.join(DATA_PATH, "bats"))
all_bats = [f for f in all_bats if f.endswith('.json')] #filter to only json files
all_balls = os.listdir(os.path.join(DATA_PATH, "balls"))
all_balls = [f for f in all_balls if f.endswith('.json')] #filter to only json files

#%% SELECT BAT AND BALL FILES
print("Available bat files:")
for i, f in enumerate(all_bats):
    print(f"{i+1} - {f}")

standard_file = input(f"Enter the number for the standard bat: ")
assert int(standard_file) <= len(all_bats), "Invalid input for standard bat file. Please enter a valid number from the list."

torpedo_file = input(f"Enter the number for the torpedo bat: ")
assert int(torpedo_file) <= len(all_bats), "Invalid input for torpedo bat file. Please enter a valid number from the list."
assert int(standard_file) != int(torpedo_file), "Standard and torpedo bat files must be different. Please enter different numbers from the list."

standard_file = DATA_PATH + 'bats/' + all_bats[int(standard_file) - 1]
torpedo_file = DATA_PATH + 'bats/' + all_bats[int(torpedo_file) - 1]

print("Available ball files:")
for i, f in enumerate(all_balls):
    print(f"{i+1} - {f}")

ball_file = input(f"Enter the number for the ball: ")
assert int(ball_file) <= len(all_balls), "Invalid input for ball file. Please enter a valid number from the list."
ball_file = DATA_PATH + 'balls/' + all_balls[int(ball_file) - 1]
#%% Step 1: Set Bat Features
## STANDARD BAT ##
bat_standard = bat_from_json(standard_file) #load from json file (this will also load the profile and set features)
bat_standard.get_H_matrix() # build system matrix


#plot the bat
if make_plots:
    plot_bat(bat_standard, title = 'Standard Bat Profile')
#%%
## TORPEDO BAT ##
bat_torpedo = bat_from_json(torpedo_file) #load from json file (this will also load the profile and set features)
bat_torpedo.get_H_matrix() # build system matrix
#plot the bat
if make_plots:
    plot_bat(bat_torpedo, title = 'Torpedo Bat Profile')
# %% Now initialize the ball
ball = ball_from_json(ball_file) #load ball parameters from json file and create Ball instance

#attach the ball to the bats for easy access during simulation
bat_standard.ball = ball
bat_torpedo.ball = ball
# %% Set up time span for simulation
tspan = (0, 0.002) #time span for simulation
N_time = 40000
t_eval = np.linspace(tspan[0], tspan[1], N_time) #time points 
#%% Set up array of impact locations
impact_idcs = np.arange(45, len(bat_standard.zs), 1)
# %% Now perform the simulations for each impact location and store results
torp_df = pd.DataFrame(columns=['idx', 'vf', 'max_u', 'max_F', 'coll_t'])
standard_df = pd.DataFrame(columns=['idx', 'vf', 'max_u', 'max_F', 'coll_t'])

os.makedirs("results/AdleyOld", exist_ok=True)
os.makedirs("results/AdleyNewMod2", exist_ok=True)

#clear existing files in results folders
# for f in os.listdir("results/AdleyOld"):
#     os.remove(os.path.join("results/AdleyOld", f))
# for f in os.listdir("results/AdleyNewMod2"):
#     os.remove(os.path.join("results/AdleyNewMod2", f))

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
    standard_df.loc[i, 'vf'] = sol_standard['yb_dot'][-1]
    standard_df.loc[i, 'max_u'] = bat_standard.ball.max_u
    standard_df.loc[i, 'max_F'] = bat_standard.ball.max_F
    standard_df.loc[i, 'coll_t'] = bat_standard.ball.t_separation

    # print("Exit velocity (standard bat): {:.2f} m/s".format(sol_standard['yb_dot'][-1]))

    #simulate for torpedo bat
    # print("Simulating torpedo bat...")
    sol_torpedo = bat_torpedo.integrate_with_ball(tspan, ball, impact_idx, t_eval = t_eval)
    torp_df.loc[i, 'vf'] = sol_torpedo['yb_dot'][-1]
    torp_df.loc[i, 'max_u'] = bat_torpedo.ball.max_u
    torp_df.loc[i, 'max_F'] = bat_torpedo.ball.max_F
    torp_df.loc[i, 'coll_t'] = bat_torpedo.ball.t_separation

    # print("Exit velocity (torpedo bat): {:.2f} m/s".format(sol_torpedo['yb_dot'][-1]))
    #store results
    results[impact_idx] = {
        'standard': sol_standard,
        'torpedo': sol_torpedo
    }

    #store results in a csv file for later analysis
    # bat_standard.to_pkl(f"results/AdleyOld/bat_impact_{impact_idx}.pkl")
    # bat_torpedo.to_pkl(f"results/AdleyNewMod2/bat_impact_{impact_idx}.pkl")


#%% summary results
summary_df = pd.DataFrame({
    'idx': impact_idcs,
    'standard_vf': standard_df['vf'].values,
    'torpedo_vf': torp_df['vf'].values,
})

summary_df.to_csv("results/impact_location_vs_exit_velocitySIM2.csv", index=False)
torp_df.to_csv("results/torpedo_bat_results.csv", index=False)
standard_df.to_csv("results/standard_bat_results.csv", index=False)
#%%plot
def mps_to_mph(v):
    """Convert velocity from meters per second to miles per hour."""
    return v * 2.23694

fig, ax = plt.subplots(figsize=(10, 6))
plt.title(r'Exit Velocity vs Impact Location')


plt.plot(impact_idcs *1e-2, summary_df['standard_vf'], label='Standard Bat', marker='o', color = colors[0])
plt.plot(impact_idcs *1e-2, summary_df['torpedo_vf'], label='Torpedo Bat', marker='s', color = colors[1])


ax.set_xlabel('Impact Location along Bat (m)')
ax.set_ylabel('Maximum Velocity (m/s)', color='black')
ax.set_ylim(min(np.min(standard_df['vf']), np.min(torp_df['vf'])) * 0.9, max(np.max(standard_df['vf']), np.max(torp_df['vf'])) * 1.1)
plt.legend()

#add box on top left with ball params
ball_params_text = f"Ball Parameters:\nMass: {ball.mass:.3f} kg\nRadius: {ball.radius:.3f} m\nInitial Velocity: {ball.initial_velocity:.1f} m/s\nCOR: {ball.e0:.2f}"
plt.text(0.05, 0.95, ball_params_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

# Add secondary y-axis on right with mph scale
ax2 = ax.twinx()
ax2.set_ylabel('Maximum Velocity (mph)', color='black')
ymin_mps, ymax_mps = ax.get_ylim()
ax2.set_ylim(mps_to_mph(ymin_mps), mps_to_mph(ymax_mps))
ax2.grid(False)

plt.savefig("impact_location_vs_exit_velocitySIM2.pdf")
plt.show()
# %%
