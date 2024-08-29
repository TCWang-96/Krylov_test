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

