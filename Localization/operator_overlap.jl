## Define the system
nbr_dots_main = 2
nbr_dots_res = 3
qn_reservoir = 0
qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)

## Hamiltonians
hams = hamiltonians_equal_param(qd_system)

## Reservoir state  
ρ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)

## Operator
op = matrix_representation(p1((1,1), qd_system.f), qd_system.H_total)
ham_tot = matrix_representation(hams.hamiltonian_total, qd_system.H_total)

t_range = range(0.01,6*π, 100)
op_t_range = [sparse(operator_time_evolution(op, t, ham_tot, qd_system.qn_total, qd_system.H_total)) for t in t_range]
op_t_eff_range = [effective_measurement(op_t, ρ_res, qd_system.H_main_qn, qd_system.H_reservoir, qd_system.H_total) for op_t in op_t_range]

# Pauli Strings Overlaps 
idx = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)
pauli_string_list = map(ps -> Matrix(ps[idx, idx]), pauli_strings(qd_system))

conc_s(a,b) = "$(a) ⊗ $(b)"
pauli_string_label =  [conc_s(σi, σj) for σi in ["σ0", "σx", "σy", "σz"] for σj in ["σ0", "σx", "σy", "σz"]]
y_clean(y) = map(x -> abs(x) < 1e-10 ? 0.0 : x, y)


function  plot_overlap()
    ps_overlap_plot = plot(layout = (4,4), size = (1000, 1000))

    for (i,ps) in enumerate(pauli_string_list)
        exp_val = map(op_t_eff -> abs(1/4*tr(op_t_eff*ps))^2, op_t_eff_range)
        plot!(ps_overlap_plot[(i - 1) ÷ 4 + 1, (i - 1) % 4 + 1],t_range, y_clean(exp_val), title = pauli_string_label[i], legend = false)
    end
    plot!(ps_overlap_plot, suptitle = "Dots in Reservoir: $(nbr_dots_res), Electrons in Reservoir: $(qn_reservoir)")

    display(ps_overlap_plot)
end


function plot_com()
    ps_commutator_plot = plot(layout = (4,4), size = (1000, 1000))
    C_f(op_t, V) = 1/4*real(tr((op_t*V - V*op_t)'*(op_t*V - V*op_t)))

    for (i,ps) in enumerate(pauli_string_list)
        com_val = map(op_t_eff -> C_f(op_t_eff, ps), op_t_eff_range)
        plot!(ps_commutator_plot[(i - 1) ÷ 4 + 1, (i - 1) % 4 + 1],t_range, y_clean(com_val), title = pauli_string_label[i], legend = false)
    end

    display(ps_commutator_plot)
end

plot_overlap()