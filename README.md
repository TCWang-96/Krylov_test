# Krylov_test

This is a demo to perform benchmark calculation to solve ground state energy of H-chain using Quantum Krylov subspace method.

Running this file will give output of array (M,|E_sub - E_exact|).

M is the number of subspace basis used.
|E_sub - E_exact| is the relative error between ground energy result given by subspace method and exact result.

Exact result E_exact is given by exact diagonalization of the Hamiltonian (full configuration method).

Subspace result it given by solving the generalized eigenvalue problem Fc = ESc.
Here the matrix elements of F and S are given by quantum computers (simulators).

