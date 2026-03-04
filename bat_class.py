
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from GIT_FOLDER.create_system_matrix import create_system_matrices, load_H_matrix
from scipy.integrate import solve_ivp
from GIT_FOLDER.plot_osc import rotate, make_box
import matplotlib.animation as animation
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
#%% Helpers
def new_bat_ode(t, x, H, N, pbar):
    """
    ODE system for bat vibration using system matrix H

    :param x: state vector (displacement and velocity)
    """

    #call progress bar update
    if pbar is not None:
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

def bat_ode_with_force(t, x, H, N, F, pbar = None):
    """
    ODE system for bat vibration using system matrix H with external forces
    :param t: time
    :param x: state vector (displacement and velocity)
    :param H: system matrix
    :param N: number of slices
    :param F: external force function
    :param pbar: progress bar object (optional)
    """

    #call progress bar update
    if pbar is not None:
        pbar.update(1)
    # Unpack state vector: x = [y, Phi, y_dot, Phi_dot]
    y = x[0:N]
    Phi = x[N:2*N]
    y_dot = x[2*N:3*N]
    Phi_dot = x[3*N:4*N]

    # Stack velocities for H
    psi = np.concatenate([y, Phi])  # shape (2N,)
    # Compute accelerations using H
    psi_ddot = H.dot(psi)  + F(t).squeeze() # shape (2N,)
    y_ddot = psi_ddot[0:N]
    Phi_ddot = psi_ddot[N:2*N]

    # Return derivative: d/dt [y, Phi, y_dot, Phi_dot] = [y_dot, Phi_dot, y_ddot, Phi_ddot]
    return np.concatenate([y_dot, Phi_dot, y_ddot, Phi_ddot])

class BatOsc:
    def __init__(self, bat_prof, dz):
        """
        Stores bat profile and dz.
        :param bat_prof: Nx2 array, columns are [z, diameter]
        :param dz: slice thickness (float)
        """
        self.bat_prof = bat_prof
        self.dz = dz
        self.N = bat_prof.shape[0]
        self.zs = bat_prof[:, 0] * 1e-2  # convert cm to m
        self.radii = bat_prof[:, 1] * 1e-3 / 2  # convert diameter to radius in m

    def get_box(self, idx, y_shift=0.0, phi=0.0):
        """
        Returns the box coordinates for slice idx, shifted by y_shift and rotated by phi.
        :param idx: index of slice
        :param y_shift: vertical shift
        :param phi: rotation angle (radians, from vertical)
        :return: Nx2 array of box corner points
        """
        z = self.zs[idx]
        H = self.radii[idx] * 2
        box = make_box(z, H, dz=self.dz)
        # Apply vertical shift to box coordinates
        box = box.copy()
        box[:, 1] += y_shift
        centre = (z + self.dz / 2, y_shift)
        return rotate(box, phi, centre=centre)
    
    def set_initial_conditions(self, state_vec):
        """
        Sets the initial conditions for the bat oscillation.
        :param state_vec: 4N-length array, first N are y displacements, next N are phi angles,
        """
        y0 = state_vec[0:self.N]
        phi0 = state_vec[self.N:2*self.N]
        dy0 = state_vec[self.N*2:3*self.N]
        dphi0 = state_vec[3*self.N:4*self.N]
        self.inits = np.array([y0, phi0, dy0, dphi0])
    
    def set_bat_features(self, mass, rho, Y, S):
        """
        Sets the bat features.
        :param mass: mass of the bat (kg)
        :param rho: density (kg/m^3)
        :param Y: Young's modulus (N/m^2)
        :param S: tensile strength (N/m^2)
        """
        self.mass = mass
        self.rho = rho
        self.Y = Y
        self.S = S
    
    def get_H_matrix(self, H_matrix=None):
        """
        Sets the system matrix H.
        :param H_matrix: 2N x 2N array
        """
        if type(H_matrix) == str:
            self.H = load_H_matrix(H_matrix, self.N)
        elif isinstance(H_matrix, np.ndarray):
            self.H = H_matrix
        else:
            self.H =create_system_matrices(self.N,
                                   Ai = np.pi * (self.radii)**2,
                                   Ii = (np.pi / 4) * (self.radii)**4,
                                   dz = self.dz,
                                   S = self.S,
                                   Y = self.Y,
                                   rho = self.rho)
        return


    def integrate_solution(self, t_span, t_eval = None):
        """ 
        Integrates the bat oscillation ODE without external forces.

        :param t_span: tuple of (t0, tf)
        :param t_eval: array of time points to evaluate the solution at
        :return: solution object from solve_ivp
        """
        
        # asserts
        assert hasattr(self, 'H'), "H matrix not set. Call get_H_matrix() first."
        assert hasattr(self, 'inits'), "Initial conditions not set. Call set_initial_conditions() first."

        #solve
        if t_eval is not None:
            sol = solve_ivp(lambda t, x: new_bat_ode(t, x, self.H, self.N, None), t_span, self.inits.flatten(), method="RK45",
                            t_eval=t_eval,
                            rtol=1e-6,
            atol=1e-9
            )
        else:
            sol = solve_ivp(lambda t, x: new_bat_ode(t, x, self.H, self.N, None), t_span, self.inits.flatten(), method="RK45",
                        rtol=1e-6,    atol=1e-9)

        # Unpack solution
        y_sol = sol.y[0:self.N, :]
        phi_sol = sol.y[self.N:2*self.N, :]
        self.y_sol = y_sol
        self.phi_sol = phi_sol
        self.t = sol.t
        return sol
    
    def integrate_solution_with_collision(self, t_span, F = None, t_eval = None):
        """ 
        Integrates the bat oscillation ODE with external forces.
        :param t_span: tuple of (t0, tf)
        :param F: external force function of time, returns 2N-length array
        :param t_eval: array of time points to evaluate the solution at
        :return: solution object from solve_ivp
        """
        if F is None:
            F = lambda t: np.zeros(2*self.N)
        # asserts
        assert hasattr(self, 'H'), "H matrix not set. Call get_H_matrix() first"
        assert hasattr(self, 'inits'), "Initial conditions not set. Call set_initial_conditions() first."

        #solve
        if t_eval is not None:
            sol = solve_ivp(lambda t, x: bat_ode_with_force(t, x, self.H, self.N, F, None), t_span, self.inits.flatten(), method="RK45", t_eval=t_eval, rtol=1e-6, atol=1e-9)
        else:
            sol = solve_ivp(lambda t, x: bat_ode_with_force(t, x, self.H, self.N, F, None), t_span, self.inits.flatten(), method="RK45", rtol=1e-6, atol=1e-9)
        # Unpack solution
        y_sol = sol.y[0:self.N, :]
        phi_sol = sol.y[self.N:2*self.N, :]
        self.y_sol = y_sol
        self.phi_sol = phi_sol
        self.t = sol.t
        return sol


    def plot_bat(self, time_idx = 0, exaggerate=1.0, exaggerate_rotation=1.0, new_fig = True, highlight=-1):
        """
        Plots the bat at a specific time index.
        :param time_idx: time index to plot
        """
        if new_fig:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_aspect('equal')
            ax.set_ylabel('Vertical Position (m)')
            ax.set_xlabel('Longitudinal Position (m)')  
            ax.set_ylim(max(self.radii)*5 * -1, max(self.radii)*5)

        if not hasattr(self, 'y_sol') or not hasattr(self, 'phi_sol'):
            print('No solution found, plotting static bat.')
            #just plot static bat
            for i in range(self.N):
                box = self.get_box(i)
                ax.plot(box[:, 0], box[:, 1], color = colors[0], alpha = 0.5)
            if highlight >= 0:
                box = self.get_box(highlight)
                ax.plot(box[:, 0], box[:, 1], color = colors[2], alpha = 1.0, linewidth=2)
            ax.set_title('Static Bat Profile')
            if new_fig:
                plt.show()
            return

        else:
            top_left = [] #track top left corner of each box
            for i in range(self.N):
                box = self.get_box(i, y_shift= exaggerate * self.y_sol[i, time_idx], phi=exaggerate_rotation * self.phi_sol[i, time_idx])
                top_left.append((box[0, 0], box[0, 1])) 
                ax.plot(box[:, 0], box[:, 1], color = colors[1], alpha=0.3)
                #scatter centre point
                ax.scatter(self.bat_prof[i, 0]*1e-2 - self.dz/2, exaggerate * self.y_sol[i, time_idx], color='r', s=5)
            ax.set_title(f'Bat Profile at Time Index {time_idx}')
            if new_fig:
                plt.show()
            return 
    def animate_bat(self, exaggerate=1.0, exaggerate_rotation=1.0, interval=10, path=None, idx=-1):
        """
        Animates the bat profile over time using plot_bat at intervals of 'interval' time indices.
        :param exaggerate: exaggeration factor for displacement
        :param interval: number of time indices between frames
        """

        if not hasattr(self, 'y_sol') or not hasattr(self, 'phi_sol'):
            print('No solution found, cannot animate bat.')
            return
        if interval < 1:
            raise ValueError("interval must be >= 1")

        frame_indices = np.arange(0, self.y_sol.shape[1], interval)
        num_frames = len(frame_indices)

        base_ylim = max(self.radii) * 1.2
        max_disp = np.max(np.abs(exaggerate * self.y_sol))
        y_lim = max(base_ylim, max_disp + base_ylim)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect('equal')
        if exaggerate != 1.0:
            ax.set_ylabel(f'{exaggerate}x Vertical Position (m)')
        else:
            ax.set_ylabel('Vertical Position (m)')
        ax.set_xlabel('Longitudinal Position (m)')
        ax.set_ylim(-y_lim, y_lim)
        ax.set_xlim(min(self.zs)-self.dz, max(self.zs)+3*self.dz)

        def update(frame):
            t_idx = frame_indices[frame]
            ax.clear()
            ax.set_aspect('equal')
            if exaggerate != 1.0:
                ax.set_ylabel(f'{exaggerate}x Vertical Position (m)')
            else:
                ax.set_ylabel('Vertical Position (m)')
            ax.set_xlabel('Longitudinal Position (m)')
            ax.set_ylim(-y_lim, y_lim)
            ax.set_xlim(min(self.zs)-self.dz, max(self.zs)+3*self.dz)
            for i in range(self.N):
                y_val = exaggerate * self.y_sol[i, t_idx]
                phi_val = exaggerate_rotation * self.phi_sol[i, t_idx]
                box = self.get_box(i, y_shift=y_val, phi=phi_val)
                ax.plot(box[:, 0], box[:, 1], color=colors[1], alpha=0.3)
                if i == idx:
                    ax.plot(box[:, 0], box[:, 1], color=colors[2], alpha=1.0, linewidth=2, label = f'Impact Location')
                    ax.legend(loc = 'lower left')

                ax.scatter((self.bat_prof[i, 0] - 1)*1e-2 + self.dz/2, y_val, color='r', s=5)
            ax.set_title(f'Bat Profile at Time {self.t[t_idx]*1000:.2f} ms') #update to actually be value of t in ms

        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, repeat=False)
        # Save the animation as an mp4 file
        if path is not None:
            ani.save(path, writer='ffmpeg', dpi=150)
        plt.show()
# %%
