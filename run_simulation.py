#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd

#my imports
from scripts.plot_osc import plot_bat
from scripts.bat_class import bat_from_json, ball_from_json
from scripts.unit_conversions import *
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
PLOT_PATH = "plots/"
os.makedirs(PLOT_PATH, exist_ok=True)

all_bats = os.listdir(os.path.join(DATA_PATH, "bats"))
all_bats = [f for f in all_bats if f.endswith('.json')] #filter to only json files
all_balls = os.listdir(os.path.join(DATA_PATH, "balls"))
all_balls = [f for f in all_balls if f.endswith('.json')] #filter to only json files

#%% SELECT BAT FILES
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
#%% SELECT BALL FILE
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
tspan = (0, 0.01) #time span for simulation
N_time = 40000
t_eval = np.linspace(tspan[0], tspan[1], N_time) #time points 
#%% Set up array of impact locations
impact_idcs = np.arange(50, len(bat_standard.zs), 1)
# %% Now perform the simulations for each impact location and store results
torp_df = pd.DataFrame(columns=['idx', 'vf', 'max_u', 'max_F', 'coll_t'])
standard_df = pd.DataFrame(columns=['idx', 'vf', 'max_u', 'max_F', 'coll_t'])
torp_df['idx'] = impact_idcs
standard_df['idx'] = impact_idcs

os.makedirs("results/AdleyOld", exist_ok=True)
os.makedirs("results/AdleyNewMod2", exist_ok=True)


results = {}
for i, impact_idx in tqdm(enumerate(impact_idcs), disable=False, desc="Simulating impacts", total=len(impact_idcs)):
    #reset ball and bats to initial conditions before each simulation
    ball.reset()
    bat_standard.reset()
    bat_torpedo.reset()

    #set bats initially at rest
    bat_standard.set_initial_conditions(np.zeros(4 * bat_standard.N))
    bat_torpedo.set_initial_conditions(np.zeros(4 * bat_torpedo.N))

    #integrate the system for the standard bat
    sol_standard = bat_standard.integrate_with_ball(tspan, ball, impact_idx, t_eval = t_eval)
    if sol_standard is None: #if the simulation failed
        print(f"Simulation failed for standard bat at impact index {impact_idx}. Skipping to next impact location.")
        continue

    #store results in dataframe
    standard_df.loc[i, 'vf'] = sol_standard['yb_dot'][-1]
    standard_df.loc[i, 'max_u'] = bat_standard.ball.max_u
    standard_df.loc[i, 'max_F'] = bat_standard.ball.max_F
    standard_df.loc[i, 'coll_t'] = bat_standard.ball.t_separation

    #integrate the system for the torpedo bat
    sol_torpedo = bat_torpedo.integrate_with_ball(tspan, ball, impact_idx, t_eval = t_eval)
    if sol_torpedo is None: #if the simulation failed 
        print(f"Simulation failed for torpedo bat at impact index {impact_idx}. Skipping to next impact location.")
        continue

    #store results in dataframe
    torp_df.loc[i, 'vf'] = sol_torpedo['yb_dot'][-1]
    torp_df.loc[i, 'max_u'] = bat_torpedo.ball.max_u
    torp_df.loc[i, 'max_F'] = bat_torpedo.ball.max_F
    torp_df.loc[i, 'coll_t'] = bat_torpedo.ball.t_separation

    #store results in dictionary for easy access later if needed
    results[impact_idx] = {
        'standard': sol_standard,
        'torpedo': sol_torpedo
    }

    
#%% summary results
RESULT_PATH = "results/" #folder to save results csv files
os.makedirs(RESULT_PATH, exist_ok=True)

#change to alans desired conventions
L = len(bat_standard.zs) * bat_standard.dz #length of bat
standard_df['ztip'] = m_to_inches(L - standard_df['idx'] * 1e-2) #inches from tip instead of m from knob
torp_df['ztip'] = m_to_inches(L - torp_df['idx'] * 1e-2) #inches from tip instead of m from knob

standard_df['vf_mph'] = mps_to_mph(standard_df['vf']) #convert exit velocity to mph
torp_df['vf_mph'] = mps_to_mph(torp_df['vf']) #convert exit velocity to mph


sim_key = input("Enter the desired unique simulation key to save the results: ")
torp_df.to_csv(f"{RESULT_PATH}{sim_key}_torp_results.csv", index=False)
standard_df.to_csv(f"{RESULT_PATH}{sim_key}_stan_results.csv", index=False)
#%%plot final results
fig, ax = plt.subplots(figsize=(8, 5))
plt.title(r'Exit Velocity vs Impact Location')


plt.plot(standard_df['ztip'], standard_df['vf_mph'], label='Standard Bat', marker='o', color = colors[0])
plt.plot(torp_df['ztip'], torp_df['vf_mph'], label='Torpedo Bat', marker='s', color = colors[1])


ax.set_xlabel('Impact Location (inches from tip)')
ax.set_ylabel('Exit Velocity (mph)', color='black')
ax.set_xlim(0, max(np.max(standard_df['ztip']), np.max(torp_df['ztip'])))
plt.legend()

#add box on top left with ball params
ball_params_text = f"Ball Parameters:\nMass: {kg_to_oz(ball.mass):.2f} oz\nRadius: {ball.radius*1e2:.2f} cm\nInitial Velocity: {mps_to_mph(ball.initial_velocity):.1f} mph\nCOR: {ball.e0:.2f}"
plt.text(0.60, 0.05, ball_params_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

plt.savefig(f"{PLOT_PATH}impact_location_vs_exit_velocity_{sim_key}.pdf")
plt.show()
# %%
