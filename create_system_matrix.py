#%% Imports
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import os 
import tqdm
from scipy.sparse import diags
import pandas as pd

def load_H_matrix(file_path, N = 84):
    """ Loads the system matrix H from a file. If the file is a CSV, it converts from sparse format to full matrix. The CSV file should have columns [i, j, H(i, j)] for non-zero entries.
    :param file_path: path to the file
    :return: system matrix H (2N x 2N)
    """
    if not file_path.endswith('.csv'):
        H = np.loadtxt(file_path, delimiter=',')
        return H
    Hdf = pd.read_csv(file_path)
    # convert df with rows [i, j, H(i, j)] to full matrix
    assert Hdf.shape[1] == 3, "CSV file must have exactly three columns: [i, j, H(i, j)]"
    Hdf.columns = ['i', 'j', 'H(i, j)']
    for idx, row in Hdf.iterrows():
        i = int(row['i']) - 1
        j = int(row['j']) - 1
        value = row['H(i, j)']
        if idx == 0:
            H = np.zeros((2*N, 2*N))
        H[i, j] = value
    return H

def create_internal_matrices(N, Ai, Ii, dz, S, Y, rho):
    """
    Docstring for create_internal_matrices
    
    :param N: Description
    :param Ai: Description
    :param Ii: Description
    :param dz: Description
    :param S: Description
    :param Y: Description
    :param rho: Description
    """
    #need 4 initial matrices
    H1 = np.zeros((N, N)) # system matrix
    H2 = np.zeros((N, N)) # system matrix
    H3 = np.zeros((N, N)) # system matrix
    H4 = np.zeros((N, N)) # system matrix

    #final matrix looks like [[H1, H2], [H3, H4]]


    #helper constants
    Lam = S / (rho * dz**2)
    Ups = Y / (rho * dz**2)

    # Helpful note: A[:-1] is analogous to A_{i - 1} and A[1:] is analogous to A_{i}, since i starts at index 2 in the equations. This allows us to create the diagonals for the sparse matrices without explicit loops.

    # H1 matrix
    a = Lam * Ai[:-1]/Ai[1:]   # i, i-1 entry, sub-diagonal
    b = -Lam * (1 + Ai[:-1]/Ai[1:])  # i, i entry, main diagonal
    b = np.insert(b, 0, 0) #add fillers to b to make it length N
    c = Lam * np.ones(N-1) # i, i+1 entry, super-diagonal

    #Quadrant 1
    H1 = diags(
        diagonals=[a, b, c],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="csr"
    )

    # H2 matrix
    a = -Lam * Ai[:-1] / (2 * Ai[1:])  # i, j-1 entry, sub-diagonal
    b = Lam / 2 * (1 - Ai[:-1]/Ai[1:])  # i, j entry, main diagonal
    #add fillers to b to make it length N
    b = np.insert(b, 0, 0)
    c = Lam / 2 * np.ones(N-1)  # i, j+1 entry, super-diagonal

    H2 = diags(
        diagonals=[a, b, c],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="csr"
    )

    

    # H3 matrix
    a = Lam * Ai[:-1] * dz**2 / (2 * Ii[1:])  # j, i-1 entry, sub-diagonal
    c = -Lam * Ai[1:] * dz**2 / (2 * Ii[1:])  # j, i+1 entry, super-diagonal
    b = -c*(1 - Ai[:-1]/Ai[1:])  # j, i
    #add fillers to b to make it length N
    b = np.insert(b, 0, 0)

    H3 = diags(
        diagonals=[a, b, c],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="csr"
    )
    
    # H4 matrix
    a = Ups * (Ii[:-1] / Ii[1:]) - Lam * Ai[:-1] * dz**2 / (4 * Ii[1:]) # j, j-1 entry, sub-diagonal
    b = -Ups * (1 + Ii[:-1] / Ii[1:]) - Lam * Ai[1:] * dz**2 / (4 * Ii[1:]) * (1 + Ai[:-1]/Ai[1:])  # j, j entry, main diagonal
    #add fillers to b to make it length N
    b = np.insert(b, 0, 0)
    c = Ups * np.ones(N-1) - Lam * Ai[1:] * dz**2 / (4 * Ii[1:])  # j, j+1 entry, super-diagonal

    H4 = diags(
        diagonals=[a, b, c],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="csr"
    )

    # Combine H matrices into a single system matrix. H1 is top-left, H2 is top-right, H3 is bottom-left, H4 is bottom-right
    # System matrix H
    H_top = np.hstack((H1.toarray(), H2.toarray()))
    H_bottom = np.hstack((H3.toarray(), H4.toarray()))
    H = np.vstack((H_top, H_bottom))

    return H

# %%
def edit_boundary(H, Ai, Ii, dz, S, Y, rho):
    """ Edits the system matrix H to apply boundary conditions 
    :param H: system matrix (2N x 2N)
    :param Ai: cross-sectional area array, N-length
    :param Ii: moment of inertia array, N-length
    :param dz: slice thickness, in m
    :param S: tensile strength, in N/m^2
    :param Y: Young's modulus, in N/m^2
    :param rho: density, in kg/m^3
    """
    N = Ai.shape[0]
    #helper constants
    Lam = S / (rho * dz**2)
    Ups = Y / (rho * dz**2)

    #First Row
    H[0, 0] = -Lam
    H[0, 1] = Lam
    H[0, N] = Lam / 2
    H[0, N + 1] = Lam / 2

    #Nth Row
    H[N-1, N-2] = Lam * (Ai[-2] / Ai[-1])
    H[N-1, N-1] = -H[N-1, N-2]
    H[N-1, 2*N -2:2*N] = -H[N-1, N-2] / 2


    #N+1 Row
    # H[N, 0] = Lam * Ai[-1]*dz**2 / (4 * Ii[-1])
    # H[N, 1] = -H[N, 0]
    # H[N, N] =  Ups - H[N, 0] 
    # H[N, N +1] = -Ups - H[N, 0]
    H[N, 0] = Lam * Ai[0] * dz**2 / (2*Ii[0])
    H[N, 1] = -Lam * Ai[0] * dz**2 / (2*Ii[0])
    H[N, N] = -Ups - Lam * Ai[-1] * dz**2 / (4*Ii[-1])
    H[N, N+1] = Ups - Lam * Ai[-1] * dz**2 / (4*Ii[-1])
 
    #2N Row
    H[2*N -1, N -2] = Lam * Ai[-2]*dz**2 / (2 * Ii[-1])
    H[2*N -1, N -1] = -H[2*N -1, N -2]
    H[2*N -1, 2*N -2] = Ups * Ii[-2] / Ii[-1] - H[2*N -1, N -2] / 2
    H[2*N -1, 2*N -1] = -Ups * Ii[-2] / Ii[-1] - H[2*N -1, N -2] / 2
    return H

def create_system_matrices(N, Ai, Ii, dz, S, Y, rho):
    """ Creates the system matrix H with boundary conditions applied
    :param N: number of slices
    :param Ai: cross-sectional area array, N-length
    :param Ii: moment of inertia array, N-length
    :param dz: slice thickness, in m
    :param S: tensile strength, in N/m^2
    :param Y: Young's modulus, in N/m^2
    :param rho: density, in kg/m^3
    :return: system matrix H (2N x 2N)
    """
    H = create_internal_matrices(N, Ai, Ii, dz, S, Y, rho)
    H = edit_boundary(H, Ai, Ii, dz, S, Y, rho)

    #save H matrix
    np.savetxt('data/H_matrix.csv', H, delimiter=',')
    return H


# %%
if __name__ == "__main__":
    # Bat profile
    #params
    bat_length = 0.84 #in cm
    mass = 0.885 # in kg
    rho = 649 # in kg/m^3
    z_CM = 0.564 # in m
    r_Ups = 0.23 # in m
    Y = 1.814 # in N/m^2
    Y = Y * 1e10  # convert to N/m^2
    S = 1.05 # in N/ m^2
    S = S * 1e9 # convert to N/m^2
    N = 84  # number of slices
    dz = bat_length / N # slice thickness in m
    
    # load bat profile
    bat_prof= np.loadtxt('data/r161.dat')
    bat_prof[:, 1] = bat_prof[:, 1] / 2 # convert diameter to radius
    Ai = np.pi * (bat_prof[:, 1] * 1e-3)**2  # in m^2
    Vi = Ai * dz  # in m^3
    Ii = np.pi/4 * (bat_prof[:, 1] * 1e-3)**4  # in m^4



    # Create system matrix
    H = create_system_matrices(N, Ai, Ii, dz, S, Y, rho)

    # Extract non-zero entries for display
    H_nonzero = pd.DataFrame(H).stack().reset_index()
    H_nonzero['level_0'] = H_nonzero['level_0'] + 1
    H_nonzero['level_1'] = H_nonzero['level_1'] + 1

    H_nonzero.columns = ['i', 'j', 'H(i, j)']
    H_nonzero = H_nonzero[abs(H_nonzero['H(i, j)']) > 1e-10].reset_index(drop=True)
    H_nonzero.to_csv('data/H_matrix_nonzero.csv', index=False)
    # Print the system matrix
    print("System Matrix H:")
    print(H)
# %%
