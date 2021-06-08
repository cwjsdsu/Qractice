import numpy as np

'''
 Generator Coordinate/Solving generator coordinate wave function
 
    - Results start deviating from diagonalization when n_angles < N.
    - Also for n_angles < N, eigenvalues of H start to be degenerate
'''
def numerical(N,e,chi):
    eps = 0.0001
    n_angles = 13
    theta = np.linspace(0,np.pi,n_angles)
    norm = np.zeros((n_angles,n_angles))
    h = np.zeros((n_angles,n_angles))

    for i in range(n_angles):
        for j in range(n_angles):
            # Fill norm kernel
            norm[i][j] = np.cos(theta[i] - theta[j]) **N
            # Fill H array
            h[i][j] = -N*e/2 * np.cos(theta[i] - theta[j])**(N-2) 
            temp = np.cos(theta[i])**2 * np.sin(theta[j])**2 + np.cos(theta[j])**2 * np.sin(theta[i])**2
            h[i][j] = h[i][j] * (np.cos(theta[i] - theta[j])*np.cos(theta[i] + theta[j]) + chi*(temp))
            
    v,u = np.linalg.eig(norm)
    invroot_norm = np.zeros((n_angles,n_angles))

    for i in range(n_angles):
        for j in range(n_angles):
            for r in range(n_angles):
                if (np.abs(v[r]) > eps):
                    # casting to real to avoid error
                    # does not cause issues because eigenvalues of real symm matrices
                    # should be real anyways
                    invroot_norm[i][j] += np.real(u[i][r] * u[j][r] / np.sqrt(v[r])) 

    H = invroot_norm @ h @ invroot_norm
    
    return np.linalg.eigvalsh(H)


'''
Same function as numerical but on a different angle interval to only
get states of good parity. 
'''
def numerical_parity_proj(N,e,chi):
    eps = 0.0001
    n_angles = 13
    theta = np.linspace(-np.pi/2,np.pi/2,n_angles)
    norm = np.zeros((n_angles,n_angles))
    h = np.zeros((n_angles,n_angles))

    for i in range(n_angles):
        for j in range(n_angles):
            # Fill norm kernel
            norm[i][j] = np.cos(theta[i] - theta[j]) **N
            # Fill H array
            h[i][j] = -N*e/2 * np.cos(theta[i] - theta[j])**(N-2)
            temp = np.cos(theta[i])**2 * np.sin(theta[j])**2 + np.cos(theta[j])**2 * np.sin(theta[i])**2
            h[i][j] = h[i][j] * (np.cos(theta[i] - theta[j])*np.cos(theta[i] + theta[j]) + chi*(temp))

    v,u = np.linalg.eig(norm)
    invroot_norm = np.zeros((n_angles,n_angles))

    for i in range(n_angles):
        for j in range(n_angles):
            for r in range(n_angles):
                if (np.abs(v[r]) > eps):
                    invroot_norm[i][j] += np.real(u[i][r] * u[j][r] / np.sqrt(v[r]))

    H = invroot_norm @ h @ invroot_norm
    return np.linalg.eigvalsh(H)
