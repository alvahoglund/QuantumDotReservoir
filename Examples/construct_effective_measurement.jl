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

## Define the system
nbr_dots_main = 2
nbr_dots_res = 3
qn_reservoir = 3
qd_system = quantum_dot_system(nbr_dots_main, nbr_dots_res, qn_reservoir)

## Hamiltonians
hams = hamiltonians(hamiltonian_so_b, qd_system)

## Reservoir state  
ρ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)
ind = sector_index(qd_system.qn_reservoir, qd_system.H_reservoir)
@show ρ_res[ind,ind]

## Operator
op = matrix_representation(nbr_op(3, qd_system.f), qd_system.H_total)
t =1.0
ham_tot = matrix_representation(hams.hamiltonian_total, qd_system.H_total)
op_t = sparse(operator_time_evolution(op, t, ham_tot, qd_system.qn_total, qd_system.H_total))
op_t_eff = effective_measurement(op_t, ρ_res, qd_system.H_main_qn, qd_system.H_reservoir, qd_system.H_total)