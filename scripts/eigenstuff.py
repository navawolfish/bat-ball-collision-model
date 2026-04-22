#%% IMPORTS
import numpy as np
import pandas as pd
from scipy.linalg import eigh

#%% EIGEN ANALYSIS
def compute_eigenfrequencies(K, M, num_modes=10, dz=0.01):
    """Compute eigenvalues, eigenvectors, and frequencies from system matrix H.
    
    :param K: stiffness matrix (2N x 2N)
    :param M: mass matrix (2N x 2N)
    :param num_modes: number of modes to return (sorted by eigenvalue descending)
    :return: DataFrame with columns ['eigenvalue', 'frequency_Hz', 'eigenvector'],
             sorted by eigenvalue descending
    """
    evals, evecs = eigh(K, M)
    #check there are 2 rigid modes with eigenvalue ~0
    rigid_mode_indices = np.where(np.isclose(evals, 0, atol=1))[0]
    assert len(rigid_mode_indices) == 2, f"Expected 2 rigid modes with eigenvalue ~0, but found {len(rigid_mode_indices)}. Lowest 2 eigenvalues: {np.sort(np.abs(evals))[:2]}"

    eig_df = pd.DataFrame({
        'eigenvalue': evals,
        'frequency_Hz': np.where(evals.real < 0, np.sqrt(-evals.real) / (2 * np.pi), 0),
        'eigenvector': list(evecs.T)
    })

    eig_df.sort_values('eigenvalue', ascending=False, inplace=True)
    eig_df.reset_index(drop=True, inplace=True)

    v1 = eig_df['eigenvector'].iloc[0]  # first mode vector
    v2 = eig_df['eigenvector'].iloc[1]  # second mode vector
    
    t_mode, r_mode = rigid_modes([v1, v2], N=len(evecs) // 2, dz=dz)
    eig_df.at[0, 'eigenvector'] = t_mode
    eig_df.at[1, 'eigenvector'] = r_mode
    eig_df['eigenvector'] = eig_df['eigenvector'].apply(lambda v: v / np.sum(np.abs(v * dz)))  #normalize integral to 1

    eig_df = norm_to_M(eig_df, M) #normalize eigenvectors with respect to mass matrix M

    return eig_df


def rigid_modes(vecs, N, dz = 0.01):
    """
    DOCSTRING
    """
    v1 = vecs[0]  # first mode vector
    v2 = vecs[1]  # second mode vector

    W1, Phi1 = v1[:N], v1[N:]
    W2, Phi2 = v2[:N], v2[N:]

    PhiMatrix = np.column_stack([Phi1, Phi2])
    _, _, Vt = np.linalg.svd(PhiMatrix)
    a, b = Vt[-2:][0]  # last two right singular vectors = null space directions

    t_mode = a*v1 + b*v2   # translation mode

    # Rotation: orthogonal complement in span{v1,v2}
    cd = np.array([-b, a])
    r_mode = cd[0]*v1 + cd[1]*v2   # rotation mode

    # Normalize each mode to sum to 1 for easier comparison
    t_mode = t_mode / np.sum(np.abs(t_mode * dz))**2
    r_mode = r_mode / np.sum(np.abs(r_mode * dz))**2

    return t_mode, r_mode


def norm_to_M(eig_df, M):
    """
    Normalize eigenvectors with respect to mass matrix M.
    """
    modes = np.stack(eig_df['eigenvector'].values, axis=1)  # (2N, num_modes)

    # Normalize wrt M: compute norms vectorized
    norms = np.sqrt(np.sum(modes * (M @ modes), axis=0))
    standard_modes = modes / norms

    # Assign normalized modes back to dataframe
    eig_df['eigenvector'] = [standard_modes[:, i] for i in range(standard_modes.shape[1])]
    return eig_df

#%% AREA NORMALIZATION
def get_Abar(bat):
    stan_Ai = bat.radii**2 * np.pi
    ymodes = bat.modes[:bat.N, :]  # only take the y-displacement part of the modes

    Abar = (ymodes.T @ (stan_Ai[:, None] * bat.dz * ymodes)) #area norm matrix

    Abar = np.mean(np.diag(Abar)) #average across modes to get single Abar value

    return Abar
#%% Modal Amplitudes
def get_an(y, modes, M):
    """Project the solution y onto the mode shape to get the modal amplitude an."""
    return modes.T @ M @ y

def modal_amps(bat):
    assert hasattr(bat, 'y_sol'), "y_sol not found in bat object. Run integration to get y_sol before computing modal amplitudes."
    assert hasattr(bat, 'ydot_sol'), "ydot_sol not found in bat object. Run integration to get ydot_sol before computing modal amplitudes."

    My = bat.M[:bat.N, :bat.N]  # mass matrix for y displacements
    modesy = bat.modes[:bat.N, :]  # mode shapes for y displacements
    an = get_an(bat.y_sol, modesy, My)  # modal amplitudes for y displacements
    adotn = get_an(bat.ydot_sol, modesy, My)  # modal amplitudes for y velocities
    return an, adotn

def get_bat_energies(bat):
    an, adotn = modal_amps(bat)
    omega_n = 2 * np.pi * np.array(bat.freqs)
    E_n = 0.5 * bat.rho * bat.Abar * (adotn**2 + omega_n[:, None]**2 * an**2)
    return E_n