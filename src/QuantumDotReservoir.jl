# using Pkg
# Pkg.activate(pwd())
# Pkg.instantiate()
module QuantumDotReservoir
using LinearAlgebra, FermionicHilbertSpaces, Revise, Random, Arpack, SparseArrays, Plots, LinearMaps, Test, NonCommutativeProducts, Statistics
using Distributions: Normal

export tight_binding_system, hamiltonians
export random_separable_state, random_product_state, triplet_plus, singlet, ground_state, def_state
export charge_measurements, effective_measurement, scrambling_map, expectation_value
export state_time_evolution, operator_time_evolution

include("quantum_dot_system.jl")
include("states.jl")
include("time_evolution.jl")
include("measurements.jl")
include("effective_measurements.jl")
include("hamiltonian_tight_binding.jl")
include("scrambling_map.jl")

end