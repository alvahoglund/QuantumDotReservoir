using Pkg
Pkg.activate(pwd())
Pkg.instantiate()
using LinearAlgebra, FermionicHilbertSpaces, Revise, Random, Arpack, SparseArrays, Plots, LinearMaps, Test, NonCommutativeProducts

includet("src//quantum_dot_system.jl")
includet("src//states.jl")
includet("src//time_evolution.jl")
includet("src//measurements.jl")
includet("src//effective_measurements.jl")
includet("src//hamiltonian_tight_binding.jl")
includet("src//scrambling_map.jl")