# Krylov_test

This is a demo to perform benchmark calculation to solve ground state energy of H-chain using Quantum Krylov subspace method.

Running this file will give output of array ($M$, $$|E_{sub} - E_{exact}|$$).

$M$ is the number of subspace basis used.

$$|E_{sub} - E_{exact}|$$ is the relative error between energy result given by subspace method and exact result.

Exact result $E_{exact}$  is given by exact diagonalization of the Hamiltonian (full configuration method).

Subspace results $E_{sub}$ are given by solving the generalized eigenvalue problem
```math
Fc = ESc
```

Here the matrix elements of $F$ and $S$ are given by quantum computers (simulators).

Chemical accuracy of 0.001 Hartree is expected to be reached using $$M \sim 10$$

To successfully run this demo I used Python version ```3.10.9```.

## Input parameters
* ```distance```: bond length of hydrogen chain
* ```orb_number```: number of orbitals. It equals to the number of hydrogen atoms in this case
* ```basis```: basis set, which is set to be "sto-3g" by default
* ```n_qubits```: number of qubits. In the case of ```basis = "sto-3g"```, ```n_qubits``` should be twice of ```orb_number```
* ```trotter_step```: number of Trotter steps when Trotterzation approximation is used
* ```max_iter```: maximum size of Krylov subspace $$M$$
* ```time_step```: time step $$t$$ for time evolution operator $$\exp(-iHt)$$
* ```shots_number```: number of shots used when performing shot simulations

When simulation is using ```calculation.matrix_elements_noshots(max_iter)```, calculation of ```orb_number > 4``` and ```n_qubits > 8``` might be time-consuming.

When simulation is using ```calculation.matrix_elements_shots(max_iter)```, ```orb_number = 2``` and ```n_qubits = 4``` should be used, otherwise larger qubit numbers will require long running time.


