#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from create_system_matrix import create_system_matrices, load_H_matrix
from scipy.integrate import solve_ivp
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
if __name__ == "__main__":
    """
    This script is for visualising the oscillation of the bat. It creates a simple rectangular box representing the bat, rotates it by a small angle, and plots the original and rotated positions to illustrate the effect of the rotation on the height of the bat. It also creates a series of boxes along the z-axis with varying heights to simulate the oscillation of the bat over time, applying small random rotations to each box to show how the height changes with rotation. The final plot shows the oscillation pattern of the bat as it rotates.
    """
    H = 6 # set sample height of the bat
    L = 2 # set sample length of the bat

    # define the corners of the rectangle representing the bat
    line_top = np.array([[-L/2, H/2], [L/2, H/2]])
    line_bottom = np.array([[-L/2, -H/2], [L/2, -H/2]])
    line_right = np.array([[L/2, H/2], [L/2, -H/2]])
    line_left = np.array([[-L/2, H/2], [-L/2, -H/2]])


    # rotate the rectangle by 30 degrees (pi/6 radians)
    rline_top = rotate(line_top, np.pi/6)
    rline_bottom = rotate(line_bottom, np.pi/6)
    rline_right = rotate(line_right, np.pi/6)
    rline_left = rotate(line_left, np.pi/6)

    diag = np.array([[0, 0], [np.sin(np.pi/6)*H/2, np.cos(np.pi/6)*H/2]])
    #plot lines

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(line_top[:, 0], line_top[:, 1], 'b-')
    ax.plot(line_bottom[:, 0], line_bottom[:, 1], 'b-')
    ax.plot(line_right[:, 0], line_right[:, 1], 'b-')
    ax.plot(line_left[:, 0], line_left[:, 1], 'b-')

    ax.plot(rline_top[:, 0], rline_top[:, 1], 'r-')
    ax.plot(rline_bottom[:, 0], rline_bottom[:, 1], 'r-')
    ax.plot(rline_right[:, 0], rline_right[:, 1], 'r-')
    ax.plot(rline_left[:, 0], rline_left[:, 1], 'r-')

    ax.plot(diag[:, 0], diag[:, 1], 'g--', label='Height after Rotation')
    ax.set_xlim(-H, H)
    ax.set_ylim(-H, H)
    ax.set_aspect('equal')
    ax.set_title('Rectangle before Rotation')
    ax.grid(True)
    plt.show()

    box = make_box(2, H, 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(box[:, 0], box[:, 1], 'b-')
    rotated_box = rotate(box, np.pi/6, centre=(2 + 1/2, 0))

    centre = (2 + 1/2, 0)
    diag = np.array([centre, (centre[0] + np.sin(np.pi/6)*H/2, centre[1] + np.cos(np.pi/6)*H/2)])
    ax.plot(diag[:, 0], diag[:, 1], 'g--', label='Height after Rotation')

    ax.scatter(centre[0], centre[1], color='g', label='Centre of Rotation')
    ax.plot(rotated_box[:, 0], rotated_box[:, 1], 'r-')
    # ax.set_xlim(-H, H)
    # ax.set_ylim(-H, H)
    ax.set_aspect('equal')
    ax.set_title('Box before and after Rotation')
    ax.grid(True)
    plt.show()
    #%%
    zs = np.arange(0, 84)
    Hs = np.sin(zs/84 * np.pi) * 80
    plt.plot(zs, Hs)

    shifts = np.array([0, 0.02, 0, 0.03, 0.1, 0.05] *14)

    #turn into boxes and rotate
    fig, ax = plt.subplots(figsize=(10, 6))
    for z, H in zip(zs, Hs):
        box = make_box(z, H, dz=1)
        rotated_box = rotate(box, shifts[z], centre=(z + 0.5, 0))
        # ax.plot(box[:, 0], box[:, 1], 'b-', alpha=0.3)
        ax.plot(rotated_box[:, 0], rotated_box[:, 1], 'r-', alpha=0.3)
    ax.set_aspect('equal')


# %%
