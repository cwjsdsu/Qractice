import numpy as np
import qiskit as qk
import mean_field

'''
 Generator Coordinate/Solving generator coordinate wave function
 
    - Results start deviating from diagonalization when n_angles < N.
    - Also for n_angles < N, eigenvalues of H start to be degenerate
'''
def numerical(N,e,chi, theta):
    n_angles = theta.size
    h = np.zeros((n_angles,n_angles))
    inv_root_norm = get_inverse_root_norm(N,theta)

    for i in range(n_angles):
        for j in range(n_angles):
            # Fill H array
            h[i][j] = -N*e/2 * np.cos(theta[i] - theta[j])**(N-2)
            temp = np.cos(theta[i])**2 * np.sin(theta[j])**2 + np.cos(theta[j])**2 * np.sin(theta[i])**2
            h[i][j] = h[i][j] * (np.cos(theta[i] - theta[j])*np.cos(theta[i] + theta[j]) + chi*(temp))

    return np.linalg.eigvalsh(inv_root_norm @ h @ inv_root_norm)


def get_inverse_root_norm(N,theta):
    eps = 0.0001
    n_angles = theta.size
    norm = np.zeros((n_angles,n_angles))
    inv_root_norm = np.zeros((n_angles,n_angles))

    for i in range(n_angles):
        for j in range(n_angles):
            norm[i][j] = np.cos(theta[i] - theta[j]) **N

    v,u = np.linalg.eigh(norm)

    for i in range(n_angles):
        for j in range(n_angles):
            for r in range(n_angles):
                if (np.abs(v[r]) > eps):
                    # casting to real to avoid error
                    # does not cause issues because eigenvalues of real symm matrices
                    # should be real anyways
                    inv_root_norm[i][j] += np.real(u[i][r] * u[j][r] / np.sqrt(v[r]))
    return inv_root_norm


# Variational Circuit: Generator Coordinate method ---------------

def Jz_circuit(theta_i, theta_j, n_shots,backend):
    qc = qk.QuantumCircuit(2,2)
    phi = theta_j-theta_i

    qc.h(0)
    qc.ry(2*theta_i, 1)

    qc.cry(2*phi, 0,1)

    qc.h(0)

    qc.measure(0,0)
    qc.measure(1,1)

    exp_values = qk.execute(qc, backend, shots=n_shots)
    results = exp_values.result().get_counts()

    return mean_field.exp_value([1.,-1.,-1.,1], results, n_shots)


def JpJm_circuit(theta_i, theta_j, n_shots,backend):
    qc = qk.QuantumCircuit(3,3)
    phi = theta_j - theta_i

    qc.h(0)
    qc.ry(2*theta_i, 1)
    qc.ry(2*theta_i, 2)

    qc.cry(2*phi, 0,1)
    qc.cry(2*phi, 0,2)
    qc.cx(1,2)
    
    qc.h(0)
    qc.h(1)
    
    qc.measure(0,0)
    qc.measure(1,1)
    qc.measure(2,2)

    exp_values = qk.execute(qc, backend, shots=n_shots)
    results = exp_values.result().get_counts()

    return mean_field.exp_value([1.,-1.,-1.,1.,0,0,0,0], results, n_shots)
