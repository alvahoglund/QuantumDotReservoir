## Define the system
nbr_dots_main = 2
nbr_dots_reservoir = 2
qn_reservoir = 1
qd_system = tight_binding_system(nbr_dots_main, nbr_dots_reservoir, qn_reservoir)

## Hamiltonians
hams = hamiltonians(qd_system)

## Main system state
ρ_main = def_state(qd_system.f, triplet_minus, qd_system.H_main)
i_sub = FermionicHilbertSpaces.indices(qd_system.qn_main, qd_system.H_main)
@show ρ_main[i_sub, i_sub]

## Reservoir state 
ρ_reservoir = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)

## Total state
ρ_total = tensor_product((ρ_main, ρ_reservoir), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total)

# Hamiltonian with conserved qn
ham_total = matrix_representation(hams.hamiltonian_total, qd_system.H_total)

## Evolve state
ρt_range = [state_time_evolution(ρ_total, t, ham_total) for t in range(4*2π, 6*2π, 400)]

## Measure charge over time
gr()
p = plot()
for i in qd_system.coordinates_total
    m = matrix_representation(nbr_op(i, qd_system.f), qd_system.H_total)
    y = [expectation_value(m, ρt) for ρt in ρt_range]
    plot!(p, y, label="Site $i")  # plot! modifies p in-place
end
display(p)