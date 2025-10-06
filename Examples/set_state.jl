## Define the system
nbr_dots_main = 2
nbr_dots_reservoir = 2
qn_reservoir = 3
qd_system = tight_binding_system(nbr_dots_main, nbr_dots_reservoir, qn_reservoir)

## Hamiltonians
seed = 5
hams = hamiltonians(qd_system, seed)

## Main system state
ρ_main = def_state(triplet_0, qd_system.H_main, qd_system.f)
i_sub = FermionicHilbertSpaces.indices(qd_system.qn_main, qd_system.H_main)
@show ρ_main[i_sub, i_sub]

## Reservoir state 
ρ_reservoir = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)
i_sub = FermionicHilbertSpaces.indices(qd_system.qn_reservoir, qd_system.H_reservoir)
@show ρ_reservoir[i_sub, i_sub]

expectation_value(ρ_reservoir, matrix_representation(nbr_op((2,2), qd_system.f), qd_system.H_reservoir))
hams.dot_params_reservoir.ϵ
## Total state
ρ_total = tensor_product((ρ_main, ρ_reservoir), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total)

# Hamiltonian with conserved qn
ham_total = matrix_representation(hams.hamiltonian_total, qd_system.H_total)

## Evolve state
ρt_range = [state_time_evolution(ρ_total, t, ham_total) for t in range(0, 4*2π, 200)]

## Measure charge over time
gr()
p = plot()
for i in qd_system.coordinates_total
    m = matrix_representation(nbr_op(i, qd_system.f), qd_system.H_total)
    y = [expectation_value(m, ρt) for ρt in ρt_range]
    plot!(p, y, label="Site $i", title = "Charge expectation values in quantum dots") 
end
display(p)

gr()
p2 = plot()
ms_main = [matrix_representation(nbr_op(coordinate, qd_system.f), qd_system.H_total) for coordinate in qd_system.coordinates_main]
y2_main = [sum(expectation_value(m, ρt) for m in ms_main) for ρt in ρt_range]
plot!(p2, y2_main, label="Main")
ms_reservoir =[matrix_representation(nbr_op(coordinate, qd_system.f), qd_system.H_total) for coordinate in qd_system.coordinates_reservoir]
y2_reservoir = y2_main = [sum(expectation_value(m, ρt) for m in ms_reservoir) for ρt in ρt_range]
plot!(p2, y2_reservoir, label="Reservoir", title = "Total charge expectation values")
display(p2)


i_sub = FermionicHilbertSpaces.indices(qd_system.qn_reservoir, qd_system.H_reservoir)
ham_res = matrix_representation(hams.hamiltonian_reservoir, qd_system.H_reservoir)
vals, vecs = eigen(Matrix(ham_res))
@show ham_res[i_sub, i_sub]