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
ham = matrix_representation(hamiltonian(qd_system.H_total, qd_system.f), qd_system.H_total)

## Reservoir state  
ρ_res = reservoir_ground_state(qd_system, ham)
ind = sector_index(qd_system.qn_reservoir, qd_system.H_reservoir) 
@show ρ_res[ind,ind]

H_total_qn = FermionicHilbertSpaces.sector(qd_system.qn_total, qd_system.H_total)

## Measurements 
function s()
    measurements = vcat(
        map(i -> matrix_representation(nbr_op(i, qd_system.f), qd_system.H_total), qd_system.sites_total),
        map(i -> matrix_representation(nbr2_op(i, qd_system.f), qd_system.H_total), qd_system.sites_total)
    )
    t=1.0
    prop = propagator(t, ham, qd_system.qn_total, qd_system.H_total)
    process_measurements = op -> effective_measurement(
        operator_time_evolution(sparse(prop), sparse(op)), ρ_res, qd_system
    ) 

    eff_measurements = map(process_measurements, measurements)

    scrambling_map = vcat([vec(m)' for m in eff_measurements]...)
    return scrambling_map
end

@profview s()