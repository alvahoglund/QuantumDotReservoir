using Pkg
Pkg.activate(pwd())
Pkg.instantiate()
using LinearAlgebra, FermionicHilbertSpaces, Revise, Random, Arpack, SparseArrays, Plots, LinearMaps, TestItems

includet("..//src//hilbert_space.jl")
includet("..//src/hamiltonian.jl")
includet("..//src/states.jl")
includet("..//src/time_evolution.jl")
includet("..//src/measurements.jl")
includet("..//src//effective_measurements.jl")

## Define the system
main_sites = [1, 2]
reservoir_sites = [3, 4, 5]
qn_main = 2
qn_reservoir = 3
qn_total = qn_main + qn_reservoir
@fermions f

# Hilbert spaces 
H_main_qn = tensor_product(hilbert_space(labels(main_sites[1]), NumberConservation(1)), hilbert_space(labels(main_sites[2]), NumberConservation(1)))
H_main = hilbert_space(keys(H_main_qn), NumberConservation())

H_reservoir = hilbert_space(labels(reservoir_sites), NumberConservation())

H_total = hilbert_space(labels(vcat(main_sites, reservoir_sites)), NumberConservation())

## Hamiltonian
ham = matrix_representation(hamiltonian(H_total, f), H_total)

## Reservoir state  
ham_res = partial_trace(ham, H_total => H_reservoir)
ρ_res = ground_state(ham_res, H_reservoir, qn_reservoir)
ind = sector_index(qn_reservoir, H_reservoir) 
@show ρ_res[ind,ind]

## Operator
op = matrix_representation(nbr_op(3, f), H_total)
t =1.0
op_t = sparse(operator_time_evolution(op, t, ham, qn_total, H_total))
op_t_eff = effective_meaurment(op_t, ρ_reservoir, H_main, H_reservoir, H_total)


