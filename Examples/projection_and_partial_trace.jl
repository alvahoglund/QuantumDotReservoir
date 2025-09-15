## ====================================================
using Pkg
Pkg.activate(pwd())
Pkg.instantiate()
using LinearAlgebra, FermionicHilbertSpaces, Revise, Random, Arpack, SparseArrays, Plots, LinearMaps, TestItems, Test

includet("..//src//quantum_dot_system.jl")
includet("..//src//hilbert_space.jl")
includet("..//src/hamiltonian.jl")
includet("..//src/states.jl")
includet("..//src/time_evolution.jl")
includet("..//src/measurements.jl")
includet("..//src//effective_measurements.jl")
# =====================================================

H = hilbert_space(labels([1,2,3,4]), NumberConservation())
Hqn = hilbert_space(labels([1,2,3,4]), NumberConservation(3))
Hsub = hilbert_space(labels([3,4]), NumberConservation())
Hsubqn = hilbert_space(labels([3,4]), NumberConservation(1))
@fermions f

#########################

ham = hamiltonian(H, f)
partial_trace(ham, H =>Hsub)

#First option
ham = matrix_representation(hamiltonian(H, f), H)
hamsub = partial_trace(ham, H, Hsub)
index = FermionicHilbertSpaces.indices(Hsubqn, Hsub)
ham1 = hamsub[index, index]

#Second option
index = FermionicHilbertSpaces.indices(Hqn, H)
hamqn = ham[index, index]
ham2 =  partial_trace(hamqn, Hqn, Hsubqn)

#Third option 
ham = matrix_representation(hamiltonian(H, f), H)
ham3 = partial_trace(ham, H, Hsubqn)

ham1 ≈ ham2 # False
ham1 ≈ ham3 #True
