## Define the system
nbr_dots_main = 2
nbr_dots_reservoir = 1
qn_reservoir = 2
qd_system = quantum_dot_system(nbr_dots_main, nbr_dots_reservoir, qn_reservoir)

## Hamiltonians
hams = hamiltonians(hamiltonian_so_b, qd_system)

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
t = 1
ρt_range = [state_time_evolution(ρ_total, t, ham_total) for t in range(0, 2π, 100)]

## Measure charge over time
gr()
p = plot()
for i in 1:nbr_dots_main+nbr_dots_reservoir
    m = matrix_representation(nbr_op(i, qd_system.f), qd_system.H_total)
    y = [expectation_value(m, ρt) for ρt in ρt_range]
    plot!(p, y, label="Site $i")  # plot! modifies p in-place
end
display(p)
