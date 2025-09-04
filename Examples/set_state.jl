using Pkg
Pkg.activate(pwd())
Pkg.instantiate()
using LinearAlgebra, FermionicHilbertSpaces, Revise, Random, Arpack, SparseArrays, Plots

includet("..//src//hilbert_space.jl")
includet("..//src/hamiltonian.jl")
includet("..//src/states.jl")
includet("..//src/time_evolution.jl")
includet("..//src/measurements.jl")

## Define the system
main_sites = [1, 2]
reservoir_sites = [3, 4, 5]
qn_main = 2
qn_reservoir = 3
qn_total = qn_main + qn_reservoir

## Hilbert spaces
H_main = hilbert_space(labels(main_sites), NumberConservation())
H_reservoir = hilbert_space(labels(reservoir_sites), NumberConservation())
H_total = hilbert_space(labels(vcat(main_sites, reservoir_sites)), NumberConservation())

## Hamiltonian
@fermions  f
ham = hamiltonian(H_total, f)
ham = matrix_representation(ham, H_total)

## Main system state
ρ_main = def_state(f, triplet_minus, H_main)
#ham_main = partial_trace(ham, H_total => H_main)
#ρ_main = ground_state(ham_main, H_main, qn_main)
i_sub = sector_index(qn_main, H_main)
@show ρ_main[i_sub, i_sub]

## Reservoir state 
ham_reservoir = partial_trace(ham, H_total => H_reservoir)
ρ_reservoir = ground_state(ham_reservoir, H_reservoir, qn_reservoir)
i_sub = sector_index(qn_reservoir, H_reservoir)
@show ρ_reservoir[i_sub, i_sub]

## Total state
ρ_total = tensor_product((ρ_main, ρ_reservoir), (H_main, H_reservoir) => H_total)

# Total system and state with set qn
i_sub = sector_index(qn_total, H_total)
ρ = ρ_total[i_sub, i_sub]
H = FermionicHilbertSpaces.sector(qn_total, H_total)
ham = ham[i_sub, i_sub]

## Evolve state
t = 1
ρt_range = [state_time_evolution(ρ, t, ham) for t in range(0, 2π, 100)]

## Measure charge over time
gr()
p = plot()
for i in 1:5
    m = matrix_representation(nbr_op(i, f), H)
    y = [expectation_value(m, ρt) for ρt in ρt_range]
    plot!(p, y, label="Site $i")  # plot! modifies p in-place
end
display(p)
