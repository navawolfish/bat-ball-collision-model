
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from create_system_matrix import create_system_matrices, load_H_matrix
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from scipy.interpolate import interp1d
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
#%% ODE system for bat vibration using system matrix H
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


def bat_ode_with_ball(t, x, H, N, M_inv_diag, impact_idx, ball, phase, pbar=None):
    """
    ODE system for bat vibration coupled with ball collision.
    State vector: x = [y, Phi, y_dot, Phi_dot, yb, yb_dot]
    :param t: time
    :param x: state vector (4N + 2)
    :param H: system matrix (2N x 2N)
    :param N: number of slices
    :param impact_idx: index of the impact location on the bat
    :param ball: Ball object
    :param phase: 'compress' or 'expand' — determines which force law to use
    :param pbar: progress bar object (optional)
    """
    if pbar is not None:
        pbar.update(1)

    # Unpack state vector
    y = x[0:N]
    Phi = x[N:2*N]
    y_dot = x[2*N:3*N]
    Phi_dot = x[3*N:4*N]
    yb = x[4*N]
    yb_dot = x[4*N + 1]

    # Ball compression: positive when ball is compressed against bat
    u = ball.radius - (yb - y[impact_idx])

    # Compute contact force
    if u > 0:
        if phase == 'compress':
            Fk = ball.compress(u)
        else:  # phase == 'expand'
            Fk = ball.expand(u, ball.k2)
    else:
        Fk = 0.0

    # Bat accelerations
    psi = np.concatenate([y, Phi])
    F = np.zeros(2 * N)
    F[impact_idx] = -Fk  # raw force; M_inv converts to acceleration
    psi_ddot = H.dot(psi) + M_inv_diag * F
    y_ddot = psi_ddot[0:N]
    Phi_ddot = psi_ddot[N:2*N]

    # Ball acceleration (Newton's 3rd law: force decelerates ball, pushes it back up)
    yb_ddot = Fk / ball.mass

    return np.concatenate([y_dot, Phi_dot, y_ddot, Phi_ddot, [yb_dot], [yb_ddot]])


def _event_ball_vel_zero(t, x, H, N, M_inv_diag, impact_idx, ball, phase, pbar=None):
    """Event function: triggers when ball velocity crosses zero (max compression)."""
    return x[4*N + 1]  # yb_dot = 0

_event_ball_vel_zero.terminal = True
_event_ball_vel_zero.direction = 1  # detect negative -> positive crossing (ball starts reversing)


def _event_separation(t, x, H, N, M_inv_diag, impact_idx, ball, phase, pbar=None):
    """Event function: triggers when ball separates from bat (u <= 0)."""
    yb = x[4*N]
    y_impact = x[impact_idx]
    u = ball.radius - (yb - y_impact)
    return u  # crosses zero when ball separates

_event_separation.terminal = True
_event_separation.direction = -1  # detect positive -> negative crossing
#%% ball force profile for collision

def F_quad(u, k, alpha):
    """ 
    Quadratic force profile for collision. Modelling the ball as a lossy spring with stiffness k and deformation u, with a nonlinearity alpha to capture the fact that the ball gets stiffer as it deforms more.
    """
    return k * u**alpha





# BatOsc class to store bat profile, initial conditions, and features, and to perform integration and plotting

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
        ### FIX UNIT CONVERSIONS
        self.zs = bat_prof[:, 0] # set z coordinates from profile (assumed to be in meters)
        self.radii = bat_prof[:, 1] #radius from profile (assumed to be in meters)
    def set_ball(self, ball):
        """
        Sets the ball parameters for collision modelling.
        :param ball: Ball object containing mass, radius, and force profile parameters
        """
        self.ball = ball
    
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
        #also set M matrix
        Ai = np.pi * (self.radii)**2
        Ii = (np.pi / 4) * (self.radii)**4
        self.M = np.diag(np.concatenate([Ai*rho * self.dz, Ii * rho / self.dz]))
        self.M_inv_diag = 1.0 / np.diag(self.M)  # precompute for ODE speed

    
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

    def validate(self, require_inits=True, require_ball=False, impact_idx=None):
        """
        Checks that the bat is fully set up before integration. Raises a descriptive
        ValueError listing every missing step.
        :param require_inits: whether initial conditions must be set
        :param require_ball: whether a ball must be attached
        :param impact_idx: if provided, checks the index is within bounds
        """
        errors = []
        if not hasattr(self, 'mass'):
            errors.append("Material features not set — call set_bat_features(mass, rho, Y, S).")
        if not hasattr(self, 'H'):
            errors.append("System matrix H not built — call get_H_matrix().")
        if require_inits and not hasattr(self, 'inits'):
            errors.append("Initial conditions not set — call set_initial_conditions(state_vec).")
        if require_ball and not hasattr(self, 'ball'):
            errors.append("No ball attached — call set_ball(ball) or pass ball to integrate_with_ball().")
        if impact_idx is not None and not (0 <= impact_idx < self.N):
            errors.append(f"impact_idx={impact_idx} is out of bounds for bat with N={self.N} slices.")
        if errors:
            msg = "BatOsc setup incomplete:\n" + "\n".join(f"  • {e}" for e in errors)
            raise ValueError(msg)

#%% Integration functions
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

    def integrate_with_ball(self, t_span, ball, impact_idx, t_eval=None, verbose = False):
        """
        Integrates bat + ball collision in two phases:
          Phase 1 (compression): runs until yb_dot = 0 (max compression)
          Phase 2 (expansion): computes k2 from max compression state, runs until separation (u <= 0)
        Then continues free vibration for the remainder of t_span.

        State vector: [y, Phi, y_dot, Phi_dot, yb, yb_dot]  (length 4N + 2)

        :param t_span: tuple (t0, tf)
        :param ball: Ball object with compress/expand/get_k2 methods
        :param impact_idx: index of impact slice
        :param t_eval: optional array of output time points
        :return: dict with keys 't', 'y_sol', 'phi_sol', 'yb', 'yb_dot',
                 'max_u', 'max_F', 'k2', 't_max_compress', 't_separation'
        """
        assert hasattr(self, 'H'), "H matrix not set. Call get_H_matrix() first"
        assert hasattr(self, 'inits'), "Initial conditions not set. Call set_initial_conditions() first."

        N = self.N

        # Store per-slice masses on ball so the ODE can divide force by slice mass
        Ai = np.pi * self.radii**2
        ball.slice_masses = self.rho * Ai * self.dz

        # Build initial state: append ball position and velocity to bat state
        bat_state = self.inits.flatten()  # length 4N
        y_impact_0 = bat_state[impact_idx]
        x0 = np.concatenate([bat_state, [ball.radius + y_impact_0, -ball.initial_velocity]])
        # yb(0) = R_ball + y_impact(0), so u = R_ball - (yb - y_impact) = 0 (just touching)
        # yb_dot(0) = -v (moving toward bat, negative direction)

        ode_args = (self.H, N, self.M_inv_diag, impact_idx, ball, 'compress')

        # --- Phase 1: compression (until yb_dot crosses zero) ---
        sol1 = solve_ivp(
            bat_ode_with_ball, (t_span[0], t_span[1]), x0,
            args=ode_args, method='RK45',
            events=[_event_ball_vel_zero],
            rtol=1e-8, atol=1e-10, dense_output=True, max_step=1e-5
        )

        # Collect phase 1 results
        all_t = [sol1.t]
        all_y = [sol1.y]

        if sol1.t_events[0].size > 0:
            t_max_compress = sol1.t_events[0][0]
            x_at_max = sol1.y_events[0][0]

            # Compute max deformation and force at max compression
            yb_max = x_at_max[4*N]
            y_impact_max = x_at_max[impact_idx]
            max_u = ball.radius - (yb_max - y_impact_max)
            max_F = ball.compress(max_u)
            k2 = ball.get_k2(max_F, max_u)
            ball.k2 = k2  # store for the expansion ODE
            if verbose:
                print(f"Max compression at t = {t_max_compress*1e3:.3f} ms: "
                    f"u_max = {max_u*1e3:.3f} mm, F_max = {max_F:.1f} N, k2 = {k2:.2e}")

            # --- Phase 2: expansion (until ball separates from bat) ---
            ode_args_expand = (self.H, N, self.M_inv_diag, impact_idx, ball, 'expand')
            sol2 = solve_ivp(
                bat_ode_with_ball, (t_max_compress, t_span[1]), x_at_max,
                args=ode_args_expand, method='RK45',
                events=[_event_separation],
                rtol=1e-8, atol=1e-10, dense_output=True, max_step=1e-5
            )

            # Skip first time point of sol2 (duplicate of transition)
            all_t.append(sol2.t[1:])
            all_y.append(sol2.y[:, 1:])

            if sol2.t_events[0].size > 0:
                t_separation = sol2.t_events[0][0]
                x_at_sep = sol2.y_events[0][0]
                if verbose:
                    print(f"Ball separates at t = {t_separation*1e3:.3f} ms, "
                          f"ball exit v = {x_at_sep[4*N+1]:.2f} m/s")

                # --- Phase 3: free vibration (no ball) ---
                if t_separation < t_span[1]:
                    bat_state_free = x_at_sep[:4*N]
                    sol3 = solve_ivp(
                        lambda t, x: new_bat_ode(t, x, self.H, N, None),
                        (t_separation, t_span[1]), bat_state_free,
                        method='RK45', rtol=1e-8, atol=1e-10, max_step=1e-5
                    )
                    # Pad ball DOFs: ball flies out with constant velocity post-separation
                    dt_post = sol3.t[1:] - t_separation
                    yb_final = x_at_sep[4*N] + x_at_sep[4*N+1] * dt_post
                    yb_dot_final = x_at_sep[4*N+1] * np.ones(sol3.t[1:].shape)
                    sol3_padded = np.vstack([
                        sol3.y[:, 1:],
                        yb_final[np.newaxis, :],
                        yb_dot_final[np.newaxis, :]
                    ])
                    all_t.append(sol3.t[1:])
                    all_y.append(sol3_padded)
            else:
                t_separation = None
                if verbose:
                    print("Warning: ball did not separate before t_final")
        else:
            t_max_compress = None
            max_u = max_F = k2 = None
            t_separation = None
            if verbose:
                print("Warning: ball velocity never crossed zero (no max compression detected)")

        # Concatenate all phases
        t_full = np.concatenate(all_t)
        y_full = np.concatenate(all_y, axis=1)

        # If t_eval requested, interpolate
        if t_eval is not None:
            interp = interp1d(t_full, y_full, kind='linear', axis=1, fill_value='extrapolate')
            y_full = interp(t_eval)
            t_full = t_eval

        # Store results
        self.y_sol = y_full[0:N, :]
        self.phi_sol = y_full[N:2*N, :]
        self.t = t_full
        
        #now store ball results as well
        if not hasattr(self, 'ball'):
            self.ball = ball
        self.ball.yb = y_full[4*N, :]
        self.ball.yb_dot = y_full[4*N+1, :]
        self.ball.u = ball.radius - (self.ball.yb - self.y_sol[impact_idx, :])  # deformation over time
        # ensure u does not go negative
        self.ball.u[self.ball.u < 0] = 0 # ball cannot pull on the bat, only compress
        self.ball.k2 = k2 # store k2 in ball object for reference
        self.ball.exit_v = self.ball.yb_dot[-1] if t_separation is not None else None

        return {
            't': t_full,
            'y_sol': y_full[0:N, :],
            'phi_sol': y_full[N:2*N, :],
            'yb': y_full[4*N, :],
            'yb_dot': y_full[4*N+1, :],
            'max_u': max_u,
            'max_F': max_F,
            'k2': k2,
            't_max_compress': t_max_compress,
            't_separation': t_separation,
        }
    def reset(self):
        """Resets the solution attributes to allow for fresh integration."""
        if hasattr(self, 'y_sol'):
            del self.y_sol
        if hasattr(self, 'phi_sol'):
            del self.phi_sol
        if hasattr(self, 't'):
            del self.t
        if hasattr(self, 'ball'):
            if hasattr(self.ball, 'yb'):
                del self.ball.yb
            if hasattr(self.ball, 'yb_dot'):
                del self.ball.yb_dot
            if hasattr(self.ball, 'u'):
                del self.ball.u
            if hasattr(self.ball, 'k2'):
                del self.ball.k2
            if hasattr(self.ball, 'exit_v'):
                del self.ball.exit_v
        return

# %% Ball class to store ball parameters and force profile for collision modelling
class Ball: 
    def __init__(self, v, e0, k1, alpha, mass, radius):
        self.mass = mass
        self.radius = radius
        self.initial_velocity = v #m/s, positive towards the bat
        self.e0 = e0 #COR
        self.k1 = k1 #stiffness of the ball, from 2000 paper, can be tuned to match observed collision times and forces
        self.alpha = alpha # lossiness of the ball, higher alpha means more nonlinear stiffening as the ball deforms
        self.beta = (1 + alpha) / e0**2 - 1 # from 2000 paper

    def compress(self, u):
        """ 
        Force on the ball as a function of deformation u. Uses the quadratic force profile with parameters k1 and alpha.
        """
        F = F_quad(u, self.k1, self.alpha)
        return F

    def expand(self, u, k2):
        """ 
        Force on the ball as it expands back after maximum compression. Uses the same force profile but with a different stiffness k2 to capture the fact that the ball is stiffer during expansion than compression. Because the ball is lossy, the maximum force during expansion is higher than during compression, which is captured by the beta parameter.
        """
        F = F_quad(u, k2, self.beta) # example values for max_F and max_u, can be tuned
        return F
    
    def get_k2(self, max_F, max_u):
        """ 
        Computes the effective stiffness k2 of the ball at maximum deformation, which occurs when the force equals the maximum force observed in collisions (max_F). This is used to ensure that the force profile matches observed collision forces.
        """
        k2 = max_F / max_u**self.beta
        return k2 
    def reset(self):
        """Resets any dynamic attributes of the ball to allow for fresh simulation."""
        if hasattr(self, 'yb'):
            del self.yb
        if hasattr(self, 'yb_dot'):
            del self.yb_dot
        if hasattr(self, 'u'):
            del self.u
        if hasattr(self, 'k2'):
            del self.k2
        if hasattr(self, 'exit_v'):
            del self.exit_v
        return
    