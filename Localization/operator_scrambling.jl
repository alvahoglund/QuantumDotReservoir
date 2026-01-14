
nbr_dots_main, nbr_dots_res, qn_reservoir = 2, 4, 1
qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
seed = 1
hams = hamiltonians(qd_system, seed)

idx = FermionicHilbertSpaces.indices(qd_system.qn_total, qd_system.H_total)

ρ_main = def_state(triplet_0, qd_system.H_main, qd_system.f)
ρ_reservoir = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)
ρ_total = tensor_product((ρ_main, ρ_reservoir), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total)

Vzz = embed(pauli_string(σz, σz, qd_system), qd_system.H_main => qd_system.H_total)[idx, idx]
Vxz = embed(pauli_string(σx, σz, qd_system), qd_system.H_main => qd_system.H_total)[idx, idx]
Vyx = embed(pauli_string(σy, σx, qd_system), qd_system.H_main => qd_system.H_total)[idx, idx]
Vyy = embed(pauli_string(σy, σy, qd_system), qd_system.H_main => qd_system.H_total)[idx, idx]
V_list = [Vzz, Vxz, Vyx, Vyy]

op_0 = matrix_representation(p1(qd_system.coordinates_reservoir[end], qd_system.f), qd_system.H_total)
ham_tot = matrix_representation(hams.hamiltonian_total, qd_system.H_total)

t_range = range(0,7*π, 100)
op_t_func(t) = operator_time_evolution(op_0, t, ham_tot, qd_system.qn_total, qd_system.H_total)[idx, idx]

op_t_range = [op_t_func(t) for t in t_range]

F_func(op_t, V) = tr(op_t'*V'*op_t*V)
overlap_func(op_t, V) = tr(op_t'*V)^2
C_f(op_t, V) = tr((op_t*V - V*op_t)'*(op_t*V - V*op_t))

#plot_scrambling = plot(layout = (length(V_list), 2))

#for (i, v_i) in enumerate([Vzz, Vxz, Vyx, Vyy])
#    plot!(plot_scrambling[i,1], t_range, [real(overlap_func(op_t, V)) for op_t in op_t_range])
#    plot!(plot_scrambling[i,2], t_range, [real(C_f(op_t, V)) for op_t in op_t_range])
#end

labels_V = ["Z ⊗ Z", "X ⊗ Z", "Y ⊗ X", "Y ⊗ Y"]
plot_scrambling = plot(layout = (2, 1))
for (i, v_i) in enumerate([Vzz, Vxz, Vyx, Vyy])
    plot!(plot_scrambling[1,1], t_range, [real(overlap_func(op_t, v_i)) for op_t in op_t_range], label = labels_V[i])
    plot!(plot_scrambling[2,1], t_range, [real(C_f(op_t, v_i)) for op_t in op_t_range], label = labels_V[i])
end

plot!(plot_scrambling[1,1], title = "|tr{V'*N(t)}|^2", legend = :topright)
plot!(plot_scrambling[2,1], title = "tr{[N(t), V]'*[N(t), V])}", legend = :topright)

display(plot_scrambling)