# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from create_system_matrix import create_system_matrices, load_H_matrix
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pickle
import json
from integrators import new_bat_ode, bat_ode_with_force, bat_ode_with_ball, _event_ball_max_comp, _event_separation
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

#%% ALL ATTRIBUTES
#running list of all attributes for BatOsc and Ball
ball_attr = ['mass', #ball mass in kg
             'radius', #ball natural radius in m
             'initial_velocity', #initial velocity towards the bat in m/s
             'e0', #coefficient of restitution (dimensionless)
             'k1', #stiffness of the ball during compression (N/m^alpha)
             'alpha', #nonlinearity of the ball force profile (dimensionless)
             #AFTER INTEGRATION 
             'k2', #stiffness of the ball during expansion (N/m^beta)
             'yb', #ball position over time (m)
             'yb_dot', #ball velocity over time (m/s)
             'u', #ball deformation over time (m)
             'F', #force exerted by the ball over time (N)
             'exit_v', #exit velocity of the ball (m/s)
             'max_u', #maximum ball deformation (m)
             'max_F', #maximum force exerted by the ball (N)
             't_max_compress', #time at maximum compression
             't_separation', #time of separation
             't_collision' #time of collision
             ] 
ball_attr_postint = ball_attr[6:] #attributes that are only defined after integration (dynamic attributes)


bat_attr = ['bat_prof', #bat profile, 2D array with columns [z, radius]
             'dz', #slice thickness (float)
             'N', #number of slices (int)
             'zs', #z coordinates of slice centers (array of length N)
             'radii', #radii of slice centers (array of length N)
             'mass', #mass of the bat (kg)
             'rho', #density (kg/m^3)
             'Y', #Young's modulus (N/m^2)
             'S', #tensile strength (N/m^2)
             'M', #mass matrix
             'M_inv_diag', #inverse diagonal of mass matrix
             'H', #system matrix
             #AFTER INTEGRATION
             'inits', #initial conditions (array of length 4N: [y0 (N), phi0 (N), dy0 (N), dphi0 (N)])
             'y_sol', #solution for y displacements (array of shape N x len(t))
             'phi_sol', #solution for phi angles (array of shape N x len(t))
             't' #time vector
             ] 
bat_attr_postint = bat_attr[12:] #attributes that are only defined after integration (dynamic attributes)

#%% helper functions
def F_quad(u, k, alpha):
    """ 
    Quadratic force profile for collision. Modelling the ball as a lossy spring with stiffness k and deformation u, with a nonlinearity alpha to capture the fact that the ball gets stiffer as it deforms more.
    """
    return k * u**alpha

def get_0_F(F, u, threshold = 1e-7):
    max_comp_idx = np.argmax(u)
    F_exp = F[max_comp_idx:]
    zero_crossings = np.where(np.abs(F_exp) < threshold)[0][0] #take first decay pt
    return zero_crossings

#%% CLASS DEFS
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
    
    def integrate_with_ball(
            self,
            t_span,
            ball,
            impact_idx,
            t_eval=None,
            verbose=False,
            method='auto',
            rtol=1e-8,
            atol=1e-10,
            max_step=1e-5,
            continue_free_vibration=True,
        ):

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
        :param method: solve_ivp method ('auto', 'RK45', 'Radau', 'BDF', ...)
        :param rtol: relative tolerance passed to solve_ivp
        :param atol: absolute tolerance passed to solve_ivp
        :param max_step: max step size passed to solve_ivp
        :param continue_free_vibration: if False, skips post-separation bat solve
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

        # Stiff systems (very large |H|) are much more robust with implicit methods.
        if method == 'auto':
            stiffness_indicator = np.max(np.abs(self.H))
            solve_method = 'Radau' if stiffness_indicator > 1e12 else 'RK45'
            if verbose:
                print(f"Using solver method '{solve_method}' (|H|_max={stiffness_indicator:.2e})")
        else:
            solve_method = method

        ode_args = (self.H, N, self.M_inv_diag, impact_idx, ball, 'compress') # arguments for the ODE function during compression phase

        # --- Phase 1: compression (until yb_dot crosses zero) ---
        sol1 = solve_ivp(
            bat_ode_with_ball, (t_span[0], t_span[1]), x0,
            args=ode_args, method=solve_method,
            events=[_event_ball_max_comp],
            rtol=rtol, atol=atol, dense_output=True, max_step=max_step
        )

        #sol1 has the form [y [0:N], Phi [N:2*N], y_dot [2*N:3*N], Phi_dot [3*N:4*N], y_b [4*N], yb_dot [4*N+1]]
        if sol1.status == -1:
            print("Integration failed, returning None. Message:", sol1.message)
            return None
        
        #store solution
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
                args=ode_args_expand, method=solve_method,
                events=[_event_separation],
                rtol=rtol, atol=atol, dense_output=True, max_step=max_step
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
                    if continue_free_vibration:
                        bat_state_free = x_at_sep[:4*N]
                        sol3 = solve_ivp(
                            lambda t, x: new_bat_ode(t, x, self.H, N, None),
                            (t_separation, t_span[1]), bat_state_free,
                            method=solve_method, rtol=rtol, atol=atol, max_step=max_step
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
                        dt_post = t_span[1] - t_separation
                        yb_final = x_at_sep[4*N] + x_at_sep[4*N+1] * dt_post
                        y_end = np.concatenate([
                            x_at_sep[:4*N],
                            [yb_final],
                            [x_at_sep[4*N+1]]
                        ])[:, np.newaxis]
                        all_t.append(np.array([t_span[1]]))
                        all_y.append(y_end)
            else:
                t_separation = None
                if verbose:
                    print("Warning: ball did not separate before t_final")
        else:
            t_max_compress = None
            max_u = max_F = k2 = None
            t_separation = None
            y = sol1.y
            # get y_ball
            ydot_b = y[4*N + 1, :]  # ball velocity over time
            # Compute max deformation and force at max compression
            ydot_b_min = np.min(ydot_b)  # should be a positive value (max velocity in reverse direction)
            u = ball.radius - (y[4*N, :] - y[impact_idx, :])  # deformation over time

            #get maximum u
            max_u = np.max(u)
            max_F = ball.compress(max_u)

            if verbose:
                print(f"Warning: ball velocity never crossed zero (no max compression detected). The maximum deformation u detected is {max_u:.2f}, with yb_dot = {ydot_b_min:.2f} m/s at that point.")

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

        #ball attributes
        self.ball.yb = y_full[4*N, :]
        self.ball.yb_dot = y_full[4*N+1, :]
        self.ball.u = ball.radius - (self.ball.yb - self.y_sol[impact_idx, :])  # deformation over time

        assert np.abs(np.max(self.ball.u) - max_u) < 1e-10, "Inconsistent maximum deformation. From deformation array: max_u = {:.6f}, from ball.max_u = {:.6f}. Check that ball.u is computed correctly from yb and y_sol.".format(np.max(self.ball.u), max_u)
        
        # ensure u does not go negative
        self.ball.u[self.ball.u < 0] = 0 # ball cannot pull on the bat, only compress
        self.ball.t = t_full
        self.ball.t_separation = t_separation
        self.ball.k2 = k2 # store k2 in ball object for reference
        self.ball.exit_v = self.ball.yb_dot[-1] if t_separation is not None else None
        self.ball.max_u = max_u
        self.ball.max_F = max_F
        self.ball.t_max_compress = t_max_compress
        self.ball.F = ball.F_from_u() if hasattr(ball, 'k2') else None # compute force over time if k2 is available

        #collision time
        t_collision = t_full[get_0_F(self.ball.F, self.ball.u)]
        self.ball.t_collision = t_collision

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
            't_collision': t_collision
        }
    def reset(self):
        """Resets the solution attributes to allow for fresh integration."""

        #dynamic vars
        for attr in ['y_sol', 'phi_sol', 't']:
            if hasattr(self, attr):
                delattr(self, attr)

        if hasattr(self, 'ball'): # also reset ball dynamic attributes if ball is attached
            for attr in ball_attr_postint:
                if hasattr(self.ball, attr):
                    delattr(self.ball, attr)
        return

    def to_pkl(self, filename, include_solution=False):
        """Saves the bat parameters and solution to a pickle file."""
        data = {
            'bat_prof': self.bat_prof,
            'dz': self.dz,
            'mass': self.mass,
            'rho': self.rho,
            'Y': self.Y,
            'S': self.S,
            'H': self.H,
            'inits': self.inits,
        }
        if hasattr(self, 'y_sol') and include_solution:
            data['y_sol'] = self.y_sol
        if hasattr(self, 'phi_sol') and include_solution:
            data['phi_sol'] = self.phi_sol
        if hasattr(self, 't') and include_solution:
            data['t'] = self.t

        for attr in ['y_sol', 'phi_sol', 't']: #dynamic variables to save if they exist
            if hasattr(self, attr):
                data[attr] = getattr(self, attr)

        if hasattr(self, 'ball'): # also save ball parameters and results if ball is attached
            ball_data = {
                'mass': self.ball.mass,
                'radius': self.ball.radius,
                'initial_velocity': self.ball.initial_velocity,
                'e0': self.ball.e0,
                'k1': self.ball.k1,
                'alpha': self.ball.alpha,
            }
            for attr in ball_attr_postint: #ball dynamic variables to save if they exist
                if hasattr(self.ball, attr):
                    ball_data[attr] = getattr(self.ball, attr)
            data['ball'] = ball_data

        with open(filename, 'wb') as f:
            pickle.dump(data, f)
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

    def expand(self, u):
        """ 
        Force on the ball as it expands back after maximum compression. Uses the same force profile but with a different stiffness k2 to capture the fact that the ball is stiffer during expansion than compression. Because the ball is lossy, the maximum force during expansion is higher than during compression, which is captured by the beta parameter.
        """
        assert hasattr(self, 'k2'), "k2 not set. Call get_k2(max_F, max_u) after max compression is observed to set k2 before using expand()."
        F = F_quad(u, self.k2, self.beta) # example values for max_F and max_u, can be tuned
        return F
    
    def get_k2(self, max_F, max_u):
        """ 
        Computes the effective stiffness k2 of the ball at maximum deformation, which occurs when the force equals the maximum force observed in collisions (max_F). This is used to ensure that the force profile matches observed collision forces.
        """
        k2 = max_F / max_u**self.beta
        return k2 
    def F_from_u(self):
        """ 
        Computes the force on the ball for any deformation u, using the compress force profile if u is increasing (compression) and the expand force profile if u is decreasing (expansion). This allows for a unified force calculation during integration.
        """
        ## check that the ball has the proper attributes
        if not hasattr(self, 'k2'):
            raise ValueError("Ball must have k2 attribute set to compute force. Call get_k2(max_F, max_u) after max compression is observed to set k2.")
        if not hasattr(self, 'u') or len(self.u) == 0:
            raise ValueError("Ball must have u attribute (deformation over time) to compute force. Ensure integrate_with_ball() has been called to set u(t).")
        
        comp_idx = np.argmax(self.u) # index of maximum compression, where we switch from compress to expand
        F = np.zeros_like(self.u)
        u_comp = self.u[:comp_idx+1]  # include the max point
        u_exp = self.u[comp_idx:]     # expansion starts at max

        assert np.abs(self.compress(self.max_u) - self.expand(self.max_u)) < 1e-10, "Compress and expand forces must match at max_u. Check that k2 is computed correctly from max_F and max_u."

        F[:comp_idx+1] = self.compress(u_comp)
        F[comp_idx:] = self.expand(u_exp)
        return F
        
    def reset(self):
        """Resets any dynamic attributes of the ball to allow for fresh simulation."""
        for attr in ball_attr_postint:
            if hasattr(self, attr):
                delattr(self, attr)
        return
    def to_pkl(self, filename):
        """Saves the ball parameters to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        return


#%% LOADING FUNCTIONS from json and pkl
def bat_from_json(filename):
    """Loads bat parameters and solution from a JSON file and returns a BatOsc instance. Does not include solution data, initial conditions, or H matrix"""
    with open(filename, 'r') as f:
        data = json.load(f)
    if 'profile_file' in data.keys():
        # If the JSON contains a reference to a profile file, load the profile from that file
        profile_path = data['profile_file']
        candidate_paths = [profile_path]
        if not os.path.isabs(profile_path):
            candidate_paths.append(os.path.join(os.path.dirname(filename), profile_path))

        resolved_profile_path = None
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                resolved_profile_path = candidate
                break

        if resolved_profile_path is None:
            raise FileNotFoundError(
                f"Profile file '{profile_path}' not found. Tried: {candidate_paths}"
            )
        
        try:
            data['bat_prof'] = np.loadtxt(resolved_profile_path)
        except Exception as e:
            raise ValueError(f"Failed to load bat profile from '{resolved_profile_path}': {e}")
    else:
        raise ValueError("JSON must contain either 'profile_file' or 'bat_prof' key with the bat profile data.")
    bat = BatOsc(data['bat_prof'], data['dz'])
    bat.set_bat_features(data['mass'], data['rho'], data['Y'], data['S'])
    return bat


def ball_from_json(filename):
    """Loads ball parameters from a JSON file and returns a Ball instance. JSON must contain keys: initial_velocity, e0, k1, alpha, mass, radius"""
    with open(filename, 'r') as f:
        data = json.load(f)
    ball = Ball(
        v=data['initial_velocity'],
        e0=data['e0'],
        k1=data['k1'],
        alpha=data['alpha'],
        mass=data['mass'],
        radius=data['radius']
    )
    return ball


def ball_from_pkl(filename):
    """Loads ball parameters from a pickle file and returns a Ball instance."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Allow loading from either a ball-only pickle or a bat pickle containing
    # nested ball data under the 'ball' key.
    if isinstance(data, dict) and 'ball' in data:
        data = data['ball']

    ball = Ball(
        v=data['initial_velocity'],
        e0=data['e0'],
        k1=data['k1'],
        alpha=data['alpha'],
        mass=data['mass'],
        radius=data['radius']
    )
    # Load dynamic attributes if they exist
    for attr in ball_attr_postint:
        if attr in data:
            setattr(ball, attr, data[attr])
    return ball



def bat_from_pkl(filename, solution = False):
    """Loads bat parameters and solution from a pickle file and returns a BatOsc instance."""
    if solution:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

    bat = BatOsc(data['bat_prof'], data['dz'])
    bat.set_bat_features(data['mass'], data['rho'], data['Y'], data['S'])
    bat.H = data['H']
    bat.inits = data['inits']
    # Load solution attributes if they exist
    for attr in bat_attr_postint:
        if attr in data:
            setattr(bat, attr, data[attr])

    # Reattach ball if it was serialized inside the bat pickle.
    if 'ball' in data and isinstance(data['ball'], dict):
        ball_data = data['ball']
        ball = Ball(
            v=ball_data['initial_velocity'],
            e0=ball_data['e0'],
            k1=ball_data['k1'],
            alpha=ball_data['alpha'],
            mass=ball_data['mass'],
            radius=ball_data['radius']
        )
        for attr in ball_attr_postint:
            if attr in ball_data:
                setattr(ball, attr, ball_data[attr])
        bat.ball = ball
    return bat