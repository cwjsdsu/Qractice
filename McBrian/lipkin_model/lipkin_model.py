import numpy as np



def hamiltonian(e,V,Jz,JpJm):
    return -e*Jz - V/2. * JpJm


''' 
 Find the eigenvalues of the Lipkin Model Hamiltonian
    N - system size
    e - parameter of hamiltonian
    V - parameter of hamiltonian
'''
def diagonalization(N,e,V):
    j_max = float(N)/2
    j = np.array([j_max])# array of all j values

    for i in range(j.size):
        m = np.arange(-j[i],j[i]+1,1.0)
        H = np.zeros((m.size,m.size))
        for i_m in range(m.size):
            H[i_m][i_m] =  e * m[i_m]
            if(i_m+2<m.size): # (J_+)**2 term from quasispin H
                H[i_m][i_m+2] = -V/2. * np.sqrt(j[i]*(j[i]+1)-(m[i_m]*(m[i_m]+1)))
                H[i_m][i_m+2] *= np.sqrt(j[i]*(j[i]+1)-((m[i_m]+1)*(m[i_m]+2)))
            if(i_m-2>=0):# (J_-)**2
                H[i_m][i_m-2] = -V/2. * np.sqrt(j[i]*(j[i]+1)-(m[i_m]*(m[i_m]-1)))
                H[i_m][i_m-2] *= np.sqrt(j[i]*(j[i]+1)-((m[i_m]-1)*(m[i_m]-2)))

    return np.linalg.eigvalsh(H)



''' Useful function thats used all over
 Returns the expectation value of results
    eig_val - eigenvalues of system
    results - results of circuit: number of measurements for each eigenvalue
    n_shots - total number of circuit measurements
'''
def exp_value(eig_val, results, n_shots):
    avg = 0.
    # for every result that was measured
    for a in results.keys():
        # obtain index by converting binary measurement to integer
        b = int(a,2)    # e.g. '10' eigenvalue stored at index 2 of eig
        # weighted sum of eigenvalues and number of measurments
        avg += eig_val[b]*results[a]
    return avg / n_shots
