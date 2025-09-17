using Pkg
Pkg.activate(pwd())
Pkg.instantiate()
using LinearAlgebra, FermionicHilbertSpaces, Revise, Random, Arpack, SparseArrays, Plots, LinearMaps, TestItems, Test

includet("src//quantum_dot_system.jl")
includet("src//hilbert_space.jl")
includet("src//hamiltonian.jl")
includet("src//states.jl")
includet("src//time_evolution.jl")
includet("src//measurements.jl")
includet("src//effective_measurements.jl")