#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from create_system_matrix import create_system_matrices, load_H_matrix
#%%PLT
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
# %%
def rotate(points, angle, centre=(0, 0)):
    """
    Rotate a set of 2D points by a given angle.

    :param points: Nx2 array of points to rotate
    :param angle: rotation angle in radians
    :return: Nx2 array of rotated points
    """
    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],
                                 [np.sin(-angle),  np.cos(-angle)]])
    translated_points = points - np.array(centre)
    rotated_points = translated_points @ rotation_matrix.T
    return rotated_points + np.array(centre)

def make_box(z, H, dz=0.01):
    """
    Create a rectangular box at z (right-coord) with height H and width dz.

    :param z: center z-coordinate
    :param H: height of the box
    :param dz: width of the box
    :return: Nx2 array of box corner points
    """

    box = np.array([[z - dz, -H / 2],
                    [z, -H / 2],
                    [z,  H / 2],
                    [z - dz,  H / 2],
                    [z - dz, -H / 2]])  # close the box
    return box

def plot_bat_disp(zs, Ri, yi, phi_i, dz = 0.01, return_fig=False):
    """
    Plot the displacement of the bat along its length.

    :param zs: array of z-coordinates along the bat
    :param Ri: array of rotation angles at each z-coordinate
    :param yi: array of vertical displacements at each z-coordinate
    :param phi_i: array of phase angles at each z-coordinate
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for z, R, y, phi in zip(zs, Ri, yi, phi_i):
        box = make_box(z, R, dz=dz)
        # Apply vertical shift to box coordinates
        box = box.copy()
        box[:, 1] += y
        centre = (z - dz / 2, y)
        box = rotate(box, phi, centre=centre)
        ax.plot(box[:, 0], box[:, 1], color = colors[0], alpha=0.3)
        ax.scatter(*centre, color = colors[1], s = 10)
    if return_fig:
        return fig, ax
    ax.set_aspect('equal')
    ax.set_title('Bat Displacement along Length')
    plt.show()

def plot_batsol_heatmap(bat_sol):
    """ 
    Plot the heatmap of the bat solution (displacement and rotation) over time.
    :param bat_sol: BatOsc object containing the solution of the bat oscillation
    """

    #ensure bat_sol has the necessary attributes
    if not hasattr(bat_sol, 'y_sol') or not hasattr(bat_sol, 'phi_sol'):
        raise ValueError("bat_sol must have attributes 'y_sol' and 'phi_sol' for plotting.")
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # y_sol subplot
    im0 = axs[0].imshow(bat_sol.y_sol, cmap='hot', interpolation='nearest', aspect='auto')
    axs[0].set_ylabel('Node')
    axs[0].set_title('Displacement (y_sol)')
    plt.colorbar(im0, ax=axs[0], label='Displacement')
    # phi_sol subplot
    im1 = axs[1].imshow(bat_sol.phi_sol, cmap='cool', interpolation='nearest', aspect='auto')
    axs[1].set_xlabel('Time Index')
    axs[1].set_ylabel('Node')
    axs[1].set_title('Rotation (phi_sol)')
    plt.colorbar(im1, ax=axs[1], label='Rotation')
    # Set x-ticks to correspond to time points in standard_bat.t
    num_ticks = 10
    if hasattr(bat_sol, 't') and len(bat_sol.t) > 1:
        xtick_locs = np.linspace(0, len(bat_sol.t)-1, num_ticks, dtype=int)
        xtick_labels = [f"{bat_sol.t[i]:.4f}" for i in xtick_locs]
        axs[1].set_xticks(xtick_locs)
        axs[1].set_xticklabels(xtick_labels, rotation=45)
    plt.tight_layout()
    return fig, axs


# %% BAT PLOTTING FUNCTIONS
def get_box(bat_sol, idx, y_shift=0.0, phi=0.0):
        """
        Returns the box coordinates for slice idx, shifted by y_shift and rotated by phi.
        :param idx: index of slice
        :param y_shift: vertical shift
        :param phi: rotation angle (radians, from vertical)
        :return: Nx2 array of box corner points
        """
        z = bat_sol.zs[idx]
        H = bat_sol.radii[idx] * 2
        box = make_box(z, H, dz=bat_sol.dz)
        # Apply vertical shift to box coordinates
        box = box.copy()
        box[:, 1] += y_shift
        centre = (z + bat_sol.dz / 2, y_shift)
        return rotate(box, phi, centre=centre)

def plot_bat(bat_sol, time_idx = 0, exaggerate=1.0, exaggerate_rotation=1.0, new_fig = True, highlight=-1, title = ''):
    """
    Plots the bat at a specific time index.
    :param time_idx: time index to plot
    """
    if new_fig:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect('equal')
        ax.set_ylabel('Vertical Position (m)')
        ax.set_xlabel('Longitudinal Position (m)')  
        ax.set_ylim(max(bat_sol.radii)*5 * -1, max(bat_sol.radii)*5)

    if not hasattr(bat_sol, 'y_sol') or not hasattr(bat_sol, 'phi_sol'):
        print('No solution found, plotting static bat.')
        #just plot static bat
        for i in range(bat_sol.N):
            box = get_box(bat_sol, i)
            ax.plot(box[:, 0], box[:, 1], color = colors[0], alpha = 0.5)
        if highlight >= 0:
            box = get_box(bat_sol, highlight)
            ax.plot(box[:, 0], box[:, 1], color = colors[2], alpha = 1.0, linewidth=2)
        if title == '':
            ax.set_title('Static Bat Profile')
        else:
            ax.set_title(title)
        if new_fig:
            plt.show()
        return

    else:
        top_left = [] #track top left corner of each box
        for i in range(bat_sol.N):
            box = get_box(bat_sol, i, y_shift= exaggerate * bat_sol.y_sol[i, time_idx], phi=exaggerate_rotation * bat_sol.phi_sol[i, time_idx])
            top_left.append((box[0, 0], box[0, 1])) 
            ax.plot(box[:, 0], box[:, 1], color = colors[1], alpha=0.3)
            #scatter centre point
            ax.scatter(bat_sol.bat_prof[i, 0]*1e-2 - bat_sol.dz/2, exaggerate * bat_sol.y_sol[i, time_idx], color='r', s=5)
        ax.set_title(f'Bat Profile at Time Index {time_idx}')
        if new_fig:
            plt.show()
        return
def animate_bat(bat_sol, exaggerate=1.0, exaggerate_rotation=1.0, interval=10, path=None, idx=-1):
        """
        Animates the bat profile over time using plot_bat at intervals of 'interval' time indices.
        :param exaggerate: exaggeration factor for displacement
        :param interval: number of time indices between frames
        """

        if not hasattr(bat_sol, 'y_sol') or not hasattr(bat_sol, 'phi_sol'):
            print('No solution found, cannot animate bat.')
            return
        if interval < 1:
            raise ValueError("interval must be >= 1")

        frame_indices = np.arange(0, bat_sol.y_sol.shape[1], interval)
        num_frames = len(frame_indices)

        base_ylim = max(bat_sol.radii) * 1.2
        max_disp = np.max(np.abs(exaggerate * bat_sol.y_sol))
        y_lim = max(base_ylim, max_disp + base_ylim)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect('equal')
        if exaggerate != 1.0:
            ax.set_ylabel(f'{exaggerate}x Vertical Position (m)')
        else:
            ax.set_ylabel('Vertical Position (m)')
        ax.set_xlabel('Longitudinal Position (m)')
        ax.set_ylim(-y_lim, y_lim)
        ax.set_xlim(min(bat_sol.zs)-bat_sol.dz, max(bat_sol.zs)+3*bat_sol.dz)

        def update(frame):
            t_idx = frame_indices[frame]
            ax.clear()
            ax.set_aspect('equal')
            if exaggerate != 1.0:
                ax.set_ylabel(f'{exaggerate}x Vertical Position (m)')
            else:
                ax.set_ylabel('Forward-Backward Position (m)')
            ax.set_xlabel('Longitudinal Position (m)')
            ax.set_ylim(-y_lim, y_lim)
            ax.set_xlim(min(bat_sol.zs)-bat_sol.dz, max(bat_sol.zs)+3*bat_sol.dz)
            for i in range(bat_sol.N):
                y_val = exaggerate * bat_sol.y_sol[i, t_idx]
                phi_val = exaggerate_rotation * bat_sol.phi_sol[i, t_idx]
                box = get_box(bat_sol, i, y_shift=y_val, phi=phi_val)
                ax.plot(box[:, 0], box[:, 1], color=colors[1], alpha=0.3)
                if i == idx:
                    ax.plot(box[:, 0], box[:, 1], color=colors[2], alpha=1.0, linewidth=2, label = f'Impact Location')
                    ax.legend(loc = 'lower left')

                ax.scatter((bat_sol.bat_prof[i, 0] - 1)*1e-2 + bat_sol.dz/2, y_val, color='r', s=5)
            ax.set_title(f'Bat Profile at Time {bat_sol.t[t_idx]*1000:.2f} ms') #update to actually be value of t in ms

        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, repeat=False)
        # Save the animation as an mp4 file
        if path is not None:
            ani.save(path, writer='ffmpeg', dpi=150)
        plt.show()

#%% BALL PLOTTING FUNCTIONS

