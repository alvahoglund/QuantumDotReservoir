def_system(nbr_dots_res, qn_reservoir) = tight_binding_system(2, nbr_dots_res, qn_reservoir)

function get_ρ_res(qd_system, hams, n) 
    ρ = eig_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir, n)
    ρ = ρ/tr(ρ)
    return ρ
end
function get_eff_op_t(qd_system, hams, ρ_res, op_symbolic, t_range)
    op = matrix_representation(op_symbolic, qd_system.H_total)
    ham_tot = matrix_representation(hams.hamiltonian_total, qd_system.H_total)

    op_t_range = [sparse(operator_time_evolution(op, t, ham_tot, qd_system.qn_total, qd_system.H_total)) for t in t_range]
    op_t_eff_range = [effective_measurement(op_t, ρ_res, qd_system.H_main_qn, qd_system.H_reservoir, qd_system.H_total) for op_t in op_t_range]
    return op_t_eff_range 
end

function get_pauli_strings(qd_system)
    idx = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)
    pauli_string_list = map(ps -> Matrix(ps[idx, idx]), pauli_strings(qd_system))
    return pauli_string_list
end

conc_s(a,b) = "$(a) ⊗ $(b)"
get_pauli_string_labels() = [conc_s(σi, σj) for σi in ["σ0", "σx", "σy", "σz"] for σj in ["σ0", "σx", "σy", "σz"]]

y_clean(y) = map(x -> abs(x) < 1e-10 ? 0.0 : x, y)

overlap(op_t_eff, pauli_string) = real(tr(1/4*op_t_eff*pauli_string)^2)

function get_overlaps(qd_system, eff_op_t_range)
    pauli_string_list =get_pauli_strings(qd_system)
    overlap_list =  [map(op_t_eff -> overlap(op_t_eff,ps), eff_op_t_range) for ps in pauli_string_list]
    return overlap_list
end

function plot_overlap(qd_system, t_range, overlap_list, ρ_res_labels)
    pauli_string_labels = get_pauli_string_labels()

    ps_overlap_plot = plot(layout = (4,4), size = (1000, 1300))

    for (i,ps) in enumerate(pauli_string_labels)
        overlap = overlap_list[i]
        plot!(ps_overlap_plot[(i - 1) ÷ 4 + 1, (i - 1) % 4 + 1], t_range, y_clean(overlap), title = ps, legend = false)
    end
    plot!(ps_overlap_plot, 
    suptitle = "Dots in Reservoir: $(length(qd_system.coordinates_reservoir)), Electrons in Reservoir: $(qd_system.qn_reservoir), \n Reservoir Eigenstate: $(ρ_res_labels[2]), Reservoir S^2: $(ρ_res_labels[1]) ϵb: $(ρ_res_labels[3])")

    display(ps_overlap_plot)
end

res_state_spin(qd_system, ρ_res) = y_clean(expectation_value(ρ_res, total_spin_op(qd_system.coordinates_reservoir, qd_system.f, qd_system.H_reservoir)))

## ====================================================================
#Def system
nbr_dots_res = 2
qn_reserovoir = 1
qd_system = def_system(nbr_dots_res, qn_reserovoir)

#Set parameters
ϵ_func() = 0
ϵb_func() = [0, 0, 0.01]
u_intra_func() = 10

t_func() = 1
t_so_func() = 0
u_inter_func() = 0

main_system_parameters = set_dot_params(ϵ_func, ϵb_func, u_intra_func, qd_system.coordinates_main)
reservoir_parameters = set_dot_params(ϵ_func, ϵb_func, u_intra_func, qd_system.coordinates_reservoir)
interaction_parameters = set_interaction_params(t_func, t_so_func, u_inter_func, qd_system.coordinates_total)
hams = hamiltonians(qd_system, main_system_parameters, reservoir_parameters, interaction_parameters)

n = 1
ρ_res = get_ρ_res(qd_system, hams, n)
ρ_res_labels = [res_state_spin(qd_system, ρ_res), n, ϵb_func()]

#Choose measurement operator
op_symbolic = p1((2,1), qd_system.f)

#Plot overlap
t_range = range(0.01,5*π, 50)
eff_op_t = get_eff_op_t(qd_system, hams, ρ_res, op_symbolic, t_range)
overlap_list = get_overlaps(qd_system, eff_op_t)
plot_overlap(qd_system, t_range, overlap_list, ρ_res_labels)


## Operator
#correlated_op_symbolic = correlated_measurements(qd_system)[5]
#measurement_comb = get_measurement_combinations(qd_system)[5]