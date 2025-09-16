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

## Define the system
nbr_dots_main = 2
nbr_dots_res = 4
qn_reservoir = 1
qd_system = quantum_dot_system(nbr_dots_main, nbr_dots_res, qn_reservoir)

## Hamiltonian
hams = hamiltonians(hamiltonian_so_b, qd_system)
ham_tot = matrix_representation(hams.hamiltonian_total, qd_system.H_total)

## Reservoir state  
ρ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)

## Measurements 
function s()
    measurements = vcat(
        map(i -> matrix_representation(nbr_op(i, qd_system.f), qd_system.H_total), qd_system.sites_total),
        map(i -> matrix_representation(nbr2_op(i, qd_system.f), qd_system.H_total), qd_system.sites_total)
    )
    t= 1
    prop = propagator(t, ham_tot, qd_system.qn_total, qd_system.H_total)
    process_measurements = op -> effective_measurement(
        operator_time_evolution(sparse(prop), sparse(op)), ρ_res, qd_system
    ) 

    eff_measurements = map(process_measurements, measurements)

    scrambling_map = vcat([vec(m)' for m in eff_measurements]...)
    return scrambling_map
end

s()

@profview s()