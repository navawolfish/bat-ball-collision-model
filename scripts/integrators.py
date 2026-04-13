import numpy as np
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
            Fk = ball.expand(u)
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


def _event_ball_max_comp(t, y, H, N, M_inv_diag, impact_idx, ball, phase, pbar=None):
    """Event function: triggers when ball reaches max compression (u_dot = 0)."""
    yb_dot = y[4*N + 1]                    # ball velocity
    y_dot_impact = y[2*N + impact_idx]       # bat velocity at impact location
    u_dot = y_dot_impact - yb_dot          # rate of deformation change
    return u_dot  # crosses zero at max compression (switch from compressing to expanding)

_event_ball_max_comp.terminal = True
_event_ball_max_comp.direction = -1  # detect negative -> positive crossing (ball starts reversing)


def _event_separation(t, y, H, N, M_inv_diag, impact_idx, ball, phase, pbar=None):
    """Event function: triggers when ball separates from bat (u <= 0)."""
    yb = y[4*N]
    y_impact = y[impact_idx]
    u = ball.radius - (yb - y_impact)
    return u  # crosses zero when ball separates

_event_separation.terminal = True
_event_separation.direction = -1  # detect positive -> negative crossing
