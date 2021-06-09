import numpy as np
import qiskit as qk
from lipkin_model import exp_value

'''
 Mean Field Lipkin Model Hamiltonian
 from Eqn 18 of notes
'''
def hamiltonian(N, e, V, Jz, JpJm):
    return -N*e*Jz - 0.5*N*(N-1)*V*JpJm


''' Computes Eqn 36 from Lipkin Notes
    Energy equation from Generator Coordinate/Back to parity-projection
'''
def analytic_parity_proj(theta,N,e,V,p):
    a = -N*e/2 * (np.cos(2*theta) + (N-1)*V/(2*e) * np.sin(2*theta)**2)
    return a * (1 + p*np.cos(2*theta)**(N-2))/(1 + p*np.cos(2*theta)**N)



# Variational Circuits: Uniform Approximation ----------------

def Jz_circuit(theta, n_shots, backend):
    qc = qk.QuantumCircuit(1,1)

    qc.ry(2*theta, 0)

    qc.measure(0,0)

    exp_values = qk.execute(qc, backend, shots=n_shots)
    results = exp_values.result().get_counts()
    return 0.5*exp_value([1.,-1.], results, n_shots)

def JpJm_circuit(theta, n_shots, backend):
    qc = qk.QuantumCircuit(2,2)
    
    qc.ry(2*theta, 0)
    qc.ry(2*theta, 1)

    qc.cx(0,1)
    qc.h(0)    
    
    qc.measure(0,0)
    qc.measure(1,1)
    
    exp_val = qk.execute(qc, backend, shots=n_shots)
    results = exp_val.result().get_counts()
    return exp_value([1.,-1.,0,0], results, n_shots)

# Eqn 21 in notes
def Jz_analytic(theta):
    return 0.5*np.cos(2*theta)

# Eqn 23 in notes
def JpJm_analytic(theta):
    return 0.5*(np.sin(2*theta))**2


