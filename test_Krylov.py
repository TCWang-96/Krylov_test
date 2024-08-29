import numpy as np
import scipy as sc
import pyscf

from qiskit.primitives import Estimator
from qiskit.quantum_info import Operator, Statevector, SparsePauliOp, Pauli

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.units import DistanceUnit

from qiskit_algorithms import TimeEvolutionProblem, TrotterQRTE
from qiskit.synthesis import LieTrotter, ProductFormula

from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate

from qiskit_aer import AerSimulator

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile

from pathlib import Path
import matplotlib.pyplot as plt

def run_pyscf(atom_test, basis="sto-3g"):
    """Run PySCF calculation for a given atom and basis"""
    mol = pyscf.M(
        atom = atom_test,
        basis = basis
    )
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    reference_fci = cisolver.kernel()[0]
    return reference_fci

def choose_molecule(orb_number, distance):
    """Choose molecule based on the number of orbitals and distance"""
    atoms = "; ".join([f'H 0. 0. {i * distance}' for i in range(orb_number)])
    return atoms

def read_xyz_file(file_path):
    """Read molecule structure from .xyz file and return atom specification for PySCF"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    num_atoms = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        atom = parts[0]
        x, y, z = map(float, parts[1:])
        atoms.append(f'{atom} {x} {y} {z}')
    
    return "; ".join(atoms)

def exact(op, v):
    """Exact calculation of expectation value v^\dagger @ op @ v; op is the operator, v is the state vector"""
    """If v is the HF state, result is the HF energy"""
    op_array = op.to_matrix()
    result = np.array(v).conj().T @ op_array @ np.array(v)
    return result

def exact_exp(op, v, time_step):
    """Exact calculation of time evolution expectation value v^\dagger @ exp(-iHt) @ v. op is the operator H, v is the state vector, time is t"""
    op_array = op.to_matrix()
    op_array_exp = sc.linalg.expm(-1j * time_step * op_array)
    result = np.array(v).conj().T @ op_array_exp @ np.array(v)
    return result

class TimeEvolutionSolver:
    def __init__(self, trotter_step):
        self.trotter_step = trotter_step

    def evolve_trotter(self, op, vector, delta_t):
        #Return the statevector of real-time evolved state using Trotterization
        spec_trotter = LieTrotter(reps = self.trotter_step)
        trotter = TrotterQRTE(product_formula=spec_trotter)
        problem = TimeEvolutionProblem(op, initial_state=vector, time=delta_t)
        result = trotter.evolve(problem)
        statevector_evolved = Statevector(result.evolved_state)

        return statevector_evolved
    
    def evolve_trotter_shots(self, op, vector, delta_t):
        #This shots is implemented in Estimator of TrotterQRTE, but it is used to calculated the expectation of aux operator in TrotterQRTE
        estimator = Estimator(options={'shots':1000})
        spec_trotter = LieTrotter(reps = self.trotter_step)
        trotter = TrotterQRTE(product_formula=spec_trotter, estimator=estimator)
        problem = TimeEvolutionProblem(op, initial_state=vector, time=delta_t)
        result = trotter.evolve(problem)
        statevector_evolved = Statevector(result.evolved_state)

        return statevector_evolved

    def evolve_trotter_circuit(self, op, vector, delta_t):
        spec_trotter = LieTrotter(reps = self.trotter_step)
        trotter = TrotterQRTE(product_formula=spec_trotter)
        problem = TimeEvolutionProblem(op, initial_state=vector, time=delta_t)
        result = trotter.evolve(problem)

        return result.evolved_state

def one_trotter_slice(H, dt):
    #generate circuit of one slice (one time step) of Trotterization exp(-i H dt)
    trotter_slice = PauliEvolutionGate(H, dt, synthesis=LieTrotter())
    circuit = QuantumCircuit(H.num_qubits)
    circuit.append(trotter_slice, range(H.num_qubits))
    circuit = circuit.decompose(reps=2)
    #circuit_stat(circuit)
    return circuit

def circuit_stat(circuit):
    print(
        f"""
        Trotter step with Lie-Trotter
        -----------------------------

                        Depth: {circuit.depth()}
                    Gate count: {len(circuit)}
            Nonlocal gate count: {circuit.num_nonlocal_gates()}
                Gate breakdown: {", ".join([f"{k.upper()}: {v}" for k, v in circuit.count_ops().items()])}
    """
    )
    return

class MatrixElementsCalculation:
    def __init__(self, trotter_step, op, vector, delta_t, diag_el, nuclear_constant, reference_value, num_qubits, max_shots, initial_state,*, hermitian:bool, toeplitz:bool):
        self.trotter_step = trotter_step
        self.op = op
        self.vector = vector
        self.delta_t = delta_t
        self.H_array = self.op.to_matrix()
        self.diag_el = diag_el
        self.nuclear_constant = nuclear_constant
        self.reference_value = reference_value
        self.num_qubits = num_qubits
        self.max_shots = max_shots
        self.initial_state = initial_state
        self.hermitian = hermitian
        self.toeplitz = toeplitz

    def state_evolution(self,time):
        """Calculate evolved state using Trotterization"""
        solver = TimeEvolutionSolver(self.trotter_step)
        return solver.evolve_trotter(self.op, self.vector, time)

    def state_evolution_circuit(self,time):
        """Calculate evolved state using Trotterization and circuit"""
        solver = TimeEvolutionSolver(self.trotter_step)
        return solver.evolve_trotter_circuit(self.op, self.vector, time)

    def matrix_elements_noshots(self, N):
        """Calculate matrix elements of F and S matrices using Trotterization; This corresponds to exact (linear algebra) results, which corresponds to infinite shots."""
        F = np.zeros((N, N), dtype=complex)
        S = np.zeros((N, N), dtype=complex)

        if N == 1:
            F[0,0] = self.diag_el
            S[0,0] = 1. + 0.j
            return F, S
        
        #This part can be also written as a loop over N
        prev_F, prev_S = self.matrix_elements_noshots(N-1)

        if self.hermitian and self.toeplitz:
            #Calculate the first row of an Toeplitz matrix
            for j in range (N):
                if j < N-1:
                    S[0,j] = prev_S[0,j]
                else:
                    state_j = self.state_evolution(j*self.delta_t)
                    S[0,j] = np.array(self.vector).conj().T @ np.array(state_j)
                    
            #Fill S matrix with calculated elements
            for i in range(1,N):
                for j in range(i,N):
                    S[i,j] = S[0,j-i]
                for j in range(i):
                    S[i,j] = np.conjugate(S[j,i])

            for i in range(N):
                for j in range(i,N):
                    if i < N-1 and j < N-1:
                        F[i,j] = prev_F[i,j]
                    else:
                        state_j = self.state_evolution(j*self.delta_t)
                        state_i = self.state_evolution(i*self.delta_t)
                        F[i,j] = np.array(state_i).conj().T @ self.H_array @ np.array(state_j)
                for j in range(i):
                    F[i,j] = np.conjugate(F[j,i])
        else:
            #Here I assume F and S are Hermitian by default
            for i in range(N):
                for j in range(i,N):
                    if i < N - 1 and j < N - 1:
                        S[i,j] = prev_S[i,j]
                        F[i,j] = prev_F[i,j]
                    else:
                        state_j = self.state_evolution(j*self.delta_t)
                        state_i = self.state_evolution(i*self.delta_t)
                        S[i,j] = np.array(state_i).conj().T @ np.array(state_j)
                        F[i,j] = np.array(state_i).conj().T @ self.H_array @ np.array(state_j)
                for j in range(i):
                    S[i,j] = np.conjugate(S[j,i])
                    F[i,j] = np.conjugate(F[j,i])

            
        energy = gen_eig_thres(F, S, N, 1e-12)
        print(((N, abs(np.min(energy) + self.nuclear_constant - self.reference_value))))

        return F, S
    
    def matrix_elements_shots(self, N):
        F = np.zeros((N, N), dtype=complex)
        S = np.zeros((N, N), dtype=complex)

        if N == 1:
            F[0,0] = self.diag_el
            S[0,0] = 1. + 0.j
            return F, S
        
        #This part can be written as a loop over N
        prev_F, prev_S = self.matrix_elements_shots(N-1)
 
        if self.hermitian and self.toeplitz:
            #Calculate the first row of an Toeplitz matrix
            for j in range (N):
                if j < N-1:
                    S[0,j] = prev_S[0,j]
                else:
                    S[0,j], error_level = overlap_matrix_S(self.op,self.initial_state, 0*self.delta_t,j*self.delta_t,self.trotter_step, self.max_shots)
                    
            #Fill S matrix with calculated elements
            for i in range (1,N):
                for j in range(i,N):
                    S[i,j] = S[0,j-i]

                for j in range(i):
                    S[i,j] = np.conjugate(S[j,i])

            for i in range(N):
                for j in range(i,N):
                    if i < N-1 and j < N-1:
                        F[i,j] = prev_F[i,j]
                    else:
                        F[i,j] = overlap_matrix_F(self.op,self.initial_state, i*self.delta_t,j*self.delta_t,self.trotter_step, self.max_shots)
                for j in range(i):
                    F[i,j] = np.conjugate(F[j,i])
        else:
            for i in range(N):
                for j in range(N):
                    if i < N - 1 and j < N - 1:
                        S[i,j] = prev_S[i,j]
                        F[i,j] = prev_F[i,j]
                    else:
                        S[i,j], error_level = overlap_matrix_S(self.op,self.initial_state, i*self.delta_t,j*self.delta_t,self.trotter_step, self.max_shots)
                        F[i,j] = overlap_matrix_F(self.op,self.initial_state, i*self.delta_t,j*self.delta_t,self.trotter_step, self.max_shots)


        energy = gen_eig_thres(F, S, N, 1e-12)
        
        #This corresponds to svd decomposition and rcond cutoff 
        #target = np.matmul(np.linalg.pinv(S,rcond=1e-14),F)
        #energy,coef = sc.linalg.eig(target)
        #results.append((M, abs(np.min(energy) + nuclear_constant - reference_fci)))
        print(((N, abs(np.min(energy) + self.nuclear_constant - self.reference_value))))
        return F, S 

def gen_eig(A, B):
    #Solving the generalized eigenvalue problem Av = BvE using pseudo-inverse of B.
    target = np.matmul(np.linalg.pinv(B, rcond=1e-14), A)
    energy, coef = sc.linalg.eig(target)
    return energy

def gen_eig_thres(A, B, N, eps):
    #Solving the generalized eigenvalue problem Av = BvE, with A and B thresholded by eps in function eig_thres.
    value, vector = eig_thres(B, eps)
    v_dagger = np.conjugate(vector).T
    A_new = v_dagger @ A @ vector
    B_new = v_dagger @ B @ vector
    target = np.matmul(np.linalg.pinv(B_new, rcond=1e-16), A_new)
    energy, coef = sc.linalg.eig(target)
    return energy

def eig_thres(A, eps = 1e-6):
    #Eigendecomposition of A and thresholding of eigenvalues. Discard eigenvalues below eps and then reconstruct A with the remaining eigenvalues and eigenvectors.
    value, vector = np.linalg.eigh(A)
    norm = np.linalg.norm(A)
    eps = eps * norm
    vector = vector[:,value > eps]
    value = value[value > eps]
    return value, vector
           
def overlap_matrix_S(H, state, i, j, trotter_step, shots=1024):
    #Calculate the matrix element of S using Hadamard test circuit
    #introducing shots can be regarded as introducing error (noise)
    #Neglecting Toeplitz strurcture can be regarded as introducing error (noise)
    simulator = AerSimulator()
    dti = i / trotter_step
    dtj = j / trotter_step

    q_reg = QuantumRegister(H.num_qubits + 1)
    c_reg = ClassicalRegister(1)
    circuit_main_re = QuantumCircuit(q_reg, c_reg)
    circuit_main_im = QuantumCircuit(q_reg, c_reg)

    ref = state
    #Real part of the Hadamard test
    circuit_main_re.append(ref, list(range(1, H.num_qubits+1)))
    circuit_main_re.h(0)

    qc_temp = QuantumCircuit(H.num_qubits, name='UdgU')
    qc_temp_left = QuantumCircuit(H.num_qubits, name = 'Uj') 
    for n in range(trotter_step):
        qc_temp_left.append(one_trotter_slice(H,dtj).to_gate(), list(range(H.num_qubits)))
    
    qc_temp_right = QuantumCircuit(H.num_qubits, name = 'Uidg') 
    for n in range(trotter_step):
        qc_temp_right.append(one_trotter_slice(H,dti).inverse().to_gate(), list(range(H.num_qubits)))
    
    qc_temp = qc_temp_left.compose(qc_temp_right)
    VU = qc_temp.to_gate().control(1)
    circuit_main_re.append(VU,list(range(H.num_qubits+1)))

    circuit_main_re.h(0)
    circuit_main_re.measure(q_reg[0],c_reg[0])

    circ_shots_re = transpile(circuit_main_re, simulator)
    result_re = simulator.run(circ_shots_re,shots = shots).result()
    counts_re = result_re.get_counts(circ_shots_re)

    if '0' in counts_re and '1' in counts_re:
        mean_val_re = (counts_re['0'] - counts_re['1']) / shots
        error_re = np.sqrt(2 * counts_re['0'] * counts_re['1'] / shots) / shots
    else:
        mean_val_re = 1 if counts_re.get('0', 0) > 0 else -1
        error_re = 0

    #Imaginary part of the Hadamard test
    circuit_main_im.append(ref, list(range(1, H.num_qubits+1)))

    circuit_main_im.h(0)
    circuit_main_im.sdg(0)
    circuit_main_im.append(VU,list(range(H.num_qubits+1)))
    circuit_main_im.h(0)
    circuit_main_im.measure(q_reg[0],c_reg[0])

    circ_shots_im = transpile(circuit_main_im, simulator)
    result_im = simulator.run(circ_shots_im,shots = shots).result()
    counts_im = result_im.get_counts(circ_shots_im)

    #Collect meausrement results and calculate expecatation value (and error)
    if '0' in counts_im and '1' in counts_im:
        mean_val_im = (counts_im['0'] - counts_im['1']) / shots
        error_im = np.sqrt(2 * counts_im['0'] * counts_im['1'] / shots) / shots
    else:
        mean_val_im = 1 if counts_im.get('0', 0) > 0 else -1
        error_im = 0
    mle = mean_val_re + 1j*mean_val_im
    error_norm = np.sqrt(error_re**2 + error_im**2)

    return mle, error_norm

def overlap_matrix_F(H, state, i, j, trotter_step, shots=1024):
    #Calculate matrix element of F using Hadamard test circuit
    simulator = AerSimulator()
    dti = i / trotter_step
    dtj = j / trotter_step

    q_reg = QuantumRegister(H.num_qubits + 1)
    c_reg = ClassicalRegister(1)

    ref = state

    qc_temp_left = QuantumCircuit(H.num_qubits, name = 'Uj') 
    for n in range(trotter_step):
        qc_temp_left.append(one_trotter_slice(H,dtj).to_gate(), list(range(H.num_qubits)))
    
    qc_temp_right = QuantumCircuit(H.num_qubits, name = 'Uidg') 
    for n in range(trotter_step):
        qc_temp_right.append(one_trotter_slice(H,dti).inverse().to_gate(), list(range(H.num_qubits)))

    sum_re = 0
    sum_im = 0
    sum_error = 0
    sum_error_im = 0
    for term, weight in H.to_list():
        qc_temp_mid = qc_temp_left.copy()
        qc_temp_mid.append(Pauli(term), list(range(H.num_qubits)))
        qc_temp = qc_temp_mid.compose(qc_temp_right)
        VU = qc_temp.to_gate().control(1)
        
        #Real part of the Hadamard test
        circuit_main_re = QuantumCircuit(q_reg, c_reg)
        circuit_main_re.append(ref, list(range(1, H.num_qubits+1)))
        circuit_main_re.h(0)

        circuit_main_re.append(VU,list(range(H.num_qubits+1)))
        circuit_main_re.h(0)
        circuit_main_re.measure(q_reg[0],c_reg[0])
        
        circ_shots_re = transpile(circuit_main_re, simulator)
        result_re = simulator.run(circ_shots_re,shots = shots).result()
        counts_re = result_re.get_counts(circ_shots_re)
        if '0' in counts_re and '1' in counts_re:
            mean_val_re = (counts_re['0'] - counts_re['1']) / shots
            error_re = np.sqrt(2 * counts_re['0'] * counts_re['1'] / shots) / shots
        else:
            mean_val_re = 1 if counts_re.get('0', 0) > 0 else -1
            error_re = 0

        sum_re += mean_val_re * weight
        sum_error += error_re * weight * weight

        #Imaginary part of the Hadamard test
        circuit_main_im = QuantumCircuit(q_reg, c_reg)
        circuit_main_im.append(ref, list(range(1, H.num_qubits+1)))
        circuit_main_im.h(0)
        circuit_main_im.sdg(0)

        circuit_main_im.append(VU,list(range(H.num_qubits+1)))
        circuit_main_im.h(0)
        circuit_main_im.measure(q_reg[0],c_reg[0])

        circ_shots_im = transpile(circuit_main_im, simulator)
        result_im = simulator.run(circ_shots_im,shots = shots).result()
        counts_im = result_im.get_counts(circ_shots_im)

        #Collect measurement results and calcualte expectation value (and error)
        if '0' in counts_im and '1' in counts_im:
            mean_val_im = (counts_im['0'] - counts_im['1']) / shots
            error_im = np.sqrt(2 * counts_im['0'] * counts_im['1'] / shots) / shots
        else:
            mean_val_im = 1 if counts_im.get('0', 0) > 0 else -1
            error_im = 0

        sum_im += mean_val_im * weight
        sum_error_im += error_im * weight * weight
    mle = sum_re + 1j * sum_im 

    return mle


def main():
    #Input parameters
    distance = 0.8  #bond length in Angstrom
    orb_number = 4  #number of orbitals
    n_qubits = 8   #number of qubits
    basis = "sto-3g"    #basis set
    trotter_step = 4 #number of Trotter steps when Trotterization is used
    max_iter = 10 #size of Krylov subspace M. Results are expected to converge as M increases.
    time_step = 0.1 #time step t for time evolution operator (exp(-iHt))
    
    shots_number = 1024 # number of shots when performing shot simulation in function overlap_matrix_S and overlap_matrix_F

    atom_test = choose_molecule(orb_number, distance)

    alpha_number = orb_number // 2 #number of alpha electrons

    reference_fci = run_pyscf(atom_test, basis = basis)
    print('E(FCI) = %.12f' % reference_fci)

    #PySCFDriver block
    driver = PySCFDriver(
        atom=atom_test,
        basis=basis,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    #Build molecular Hamiltonian
    problem = driver.run()
    hamiltonian = problem.hamiltonian
    nuclear_constant = hamiltonian.nuclear_repulsion_energy 
    
    #Mapping to qubit Hamiltonian
    second_q_op = hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    jw_op = mapper.map(second_q_op)

    #Prepare HF state and calculate HF energy
    hf_state = HartreeFock(orb_number, (alpha_number, alpha_number), JordanWignerMapper())
    
    #Set HF state as the initial (reference) state
    initial_state = Statevector(hf_state)
    #This hf_energy has not included the nuclear constant
    hf_energy = exact(jw_op,initial_state)
    
    H_array = jw_op.to_matrix()

    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')

    calculation = MatrixElementsCalculation(trotter_step, jw_op, initial_state, time_step, hf_energy, nuclear_constant, reference_fci, n_qubits, shots_number, hf_state, hermitian=False, toeplitz=False)
    #calculation.matrix_elements_shots(max_iter)
    calculation.matrix_elements_noshots(max_iter)

# Main
if __name__ == "__main__":
    main()