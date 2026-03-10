#%% Imports
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import os 
import tqdm
from scipy.sparse import diags

from create_system_matrix import create_system_matrices

#%% Bat profile
#params
bat_length = 0.84 #in m
mass = 0.885 # in kg
rho = 649 # in kg/m^3
Y = 1.814 # in N/m^2
Y = Y * 1e10  # convert to N/m^2
S = 0.105 # in N/ m^2
S = S * 1e9 # convert to N/m^2
dz = 0.01  # slice thickness in m



#%% Define the ODE system
def get_yddoti(yis, Phiis, Ais, S = S, rho = rho, dz = dz):
    """ Compute the second derivative of y at slice i
    :param yis: list of y values at [i-1, i, i+1]
    :param phiis: list of phi values at [i-1, i]
    :param Ais: list of cross-sectional areas at [i-1, i]
    :param S: Stretching modulus
    :param rho: density
    :param dz: slice thickness
    :return: yddoti: second derivative of y at slice i (* rho * dz^2 / S)
    """
    #unpack inputs
    coeff_y = S / (rho * dz**2) # coefficient to convert to physical units

    # yis = [ylast, ycurr, ynext]
    ylast = yis[0]
    ycurr = yis[1]
    ynext = yis[2]

    # phiis = [philast, phicurr]
    Philast = Phiis[0]
    Phicurr = Phiis[1]

    #Ais = [Alast, Acurr]
    Alast = Ais[0]
    Acurr = Ais[1]

    #compute yddoti
    yddoti = ylast + ynext - 2*ycurr + (-Alast + Acurr) / (Acurr) * (ycurr - ylast + Philast) + (Phicurr - Philast)
    
    return yddoti

def Phi_ddoti(yis, Phiis, Ais, S = S, rho = rho, dz = dz, Y = Y):
    """ Compute the second derivative of Phi at slice i
    :param yis: list of y values at [i-1, i, i+1]
    :param phiis: list of phi values at [i-1, i, i+1]
    :param Ais: list of cross-sectional areas at [i-1, i]
    :param S: Stretching modulus
    :param rho: density
    :param dz: slice thickness
    :param Y: Young's modulus
    :return: Phi_ddoti: second derivative of Phi at slice i (* rho * dz^2 / S)
    """

    # yis = [ylast, ycurr, ynext]
    ylast = yis[0]
    ycurr = yis[1]
    ynext = yis[2]

    # Phiis = [Philast, Phicurr, Phinext]
    Philast = Phiis[0]
    Phicurr = Phiis[1]
    Phinext = Phiis[2]

    #Ais = [Alast, Acurr]
    Alast = Ais[0]
    Acurr = Ais[1]

    #Iis, moment of inertia for a cylindrical cross-section
    Ilast = 4/np.pi * Alast**2
    Icurr = 4/np.pi * Acurr**2
    I_term = (Icurr - Ilast) / Icurr # term due to change in moment of inertia

    #compute Phi_ddoti
    #term1
    coeff1 = Y / (rho * dz**2) # coefficient 1 to convert to physical units
    term1 = coeff1 * (Phinext - 2*Phicurr + Philast + I_term * (Phicurr - Philast))

    #term2
    coeff2 = - S / (2 *rho * Icurr) # coefficient 2 to convert to physical units
    term2a = Acurr * (ynext - ylast + Phicurr + Philast)
    term2b = (Alast - Acurr)*(ycurr - ylast + Philast)

    Phi_ddoti = coeff1 * term1 + coeff2 * (term2a + term2b)
    return Phi_ddoti

def bat_ode(t, x, S = S, Y = Y, pbar = ):
    """
    ODE system for bat vibration

    :param x: state vector (displacement and velocity)
    """

    #call progress bar update
    pbar.update(1)
    # Unpack state vector -> x = [y, dy, Phi, dPhi], x is a 4N-dim vector
    # Unpack state vector (flattened)
    y = x[0:N]
    y_dot = x[N:2*N]
    Phi = x[2*N:3*N]
    Phi_dot = x[3*N:4*N]

    # Initialize acceleration arrays
    y_ddot = np.zeros(N)
    Phi_ddot = np.zeros(N)

    for i in range(N):
        if i == 0:
            # Boundary condition at the handle (free end), ghost nodes
            ylast = np.array(y[0] - Phi[0] * dz)
            y_pass = np.array([ylast, y[0], y[1]])
            phi_pass = np.array([Phi[0], Phi[0], Phi[1]])
            A_pass = np.array([A[0], A[0]])
        elif i == N - 1:
            # Boundary condition at the barrel tip (free end), ghost nodes
            Phinext = Phi[N-1]
            ynext = y[N-1] + Phinext * dz

            y_pass = np.array([y[N-2], y[N-1], ynext])
            phi_pass = np.array([Phi[N-2], Phi[N-1], Phi[N-1]])
            A_pass = np.array([A[N-1], A[N-1]])
        else:
            y_pass = np.array([y[i-1], y[i], y[i+1]])
            phi_pass = np.array([Phi[i-1], Phi[i], Phi[i+1]])
            A_pass = np.array([A[i-1], A[i]])

        #pass for yddoti calculation
        y_ddot[i] = get_yddoti(y_pass, phi_pass, A_pass)
        #pass for Pheiddoti calculation
        Phi_ddot[i] = Phi_ddoti(y_pass, phi_pass, A_pass)

    # Return derivative: d/dt [y, y_dot, Phi, Phi_dot] = [y_dot, y_ddot, Phi_dot, Phi_ddot]
    return np.concatenate([y_dot, y_ddot, Phi_dot, Phi_ddot])

# %%
