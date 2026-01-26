function get_hams(nbr_dots_res)
    nbr_dots_main, nbr_dots_res, qn_reservoir = 2, nbr_dots_res, 0
    qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)

    #Set parameters
    ϵ_func() = 0
    ϵb_func() = [0, 0, 1]
    u_intra_func() = 10

    t_func() = 1
    t_so_func() = 0
    u_inter_func() = 0

    main_system_parameters = set_dot_params(ϵ_func, ϵb_func, u_intra_func, qd_system.coordinates_main)
    reservoir_parameters = set_dot_params(ϵ_func, ϵb_func, u_intra_func, qd_system.coordinates_reservoir)
    interaction_parameters = set_interaction_params(t_func, t_so_func, u_inter_func, qd_system.coordinates_total)
    hams = hamiltonians(qd_system, main_system_parameters, reservoir_parameters, interaction_parameters)
    
    return hams
end

function get_states(hams, nbr_dots_res, qn_res)
    nbr_dots_main, nbr_dots_res, qn_reservoir = 2, nbr_dots_res, qn_res
    qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
    
    ρ_main_list, state_labels = singlet_triplets(qd_system)
    if qn_res == 0 || qn_res == nbr_dots_res*2 
        ρ_reservoir = eig_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir, 1)
    else
        ρ_reservoir = eig_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir, 1)
    end
    ρ_total_list = [tensor_product((ρ_main, ρ_reservoir), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total)
                    for ρ_main in ρ_main_list]

    t_range = range(0, 1*2π, 300)
    
    ham_tot = matrix_representation(hams.hamiltonian_total, qd_system.H_total)
    ρt_range_list = [[state_time_evolution(ρ_total, t, ham_tot, qd_system.H_total, qn_res+2) 
                for t in t_range] 
                for ρ_total in ρ_total_list]
    return t_range, ρt_range_list, state_labels, qd_system
end

function singlet_triplets(qd_system)
    state_labels = ["S", "T0", "T+", "T-"]
    ρ_main_list = [def_state(singlet, qd_system.H_main, qd_system.f),
                  def_state(triplet_0, qd_system.H_main, qd_system.f),
                  def_state(triplet_plus, qd_system.H_main, qd_system.f),
                  def_state(triplet_minus, qd_system.H_main, qd_system.f)]
    return ρ_main_list, state_labels
end

function triplets(qd_system)
    state_labels = ["T0", "T+", "T-"]
    ρ_main_list = [def_state(triplet_0, qd_system.H_main, qd_system.f),
                  def_state(triplet_plus, qd_system.H_main, qd_system.f),
                  def_state(triplet_minus, qd_system.H_main, qd_system.f)]
    return ρ_main_list, state_labels
end

function single_spin_diff_state(qd_system)
    f = qd_system.f
    v_1 = matrix_representation(f[(1,1), :↑]'f[(1,2), :↓]', qd_system.H_main)*vac_state(qd_system.H_main)
    v_2 = matrix_representation(f[(1,1), :↓]'f[(1,2), :↑]', qd_system.H_main)*vac_state(qd_system.H_main)
    ρ1 = v_1*v_1'
    ρ2=  v_2*v_2'

    state_labels = ["|↑↓>", "|↓↑>"]
    return [ρ1, ρ2], state_labels
end
function prod_sep_states(qd_system)
    f = qd_system.f
    v_sep1 = matrix_representation(f[(1,1), :↑]'f[(1,2), :↓]', qd_system.H_main)*vac_state(qd_system.H_main)
    v_sep2 = matrix_representation(f[(1,1), :↓]'f[(1,2), :↑]', qd_system.H_main)*vac_state(qd_system.H_main)
    ρ_sep = v_sep1*v_sep1' + v_sep2*v_sep2'
    ρ_sep= ρ_sep/tr(ρ_sep)
    ρ_ent = def_state(singlet, qd_system.H_main, qd_system.f)
    ρ_states_list = [ρ_sep, ρ_ent]
    state_labels = ["Separable", "Entangled"]
    return ρ_states_list, state_labels
end

function plot_charge_exp(nbr_dots_res, qn_res)
    hams = get_hams(nbr_dots_res)
    t_range, ρt_range_list, state_labels, qd_system =  get_states(hams, nbr_dots_res, qn_res)
    
    coord = qd_system.coordinates_reservoir[end]
    p0_measurement = matrix_representation(p0(coord, qd_system.f), qd_system.H_total)
    p1_measurement = matrix_representation(p1(coord, qd_system.f), qd_system.H_total)
    p2_measurement = matrix_representation(p2(coord, qd_system.f), qd_system.H_total)
    plot_charges = plot(layout = (1, 3), size = (1000, 500))

    plot!(plot_charges[1,1], title = "0 electrons")
    plot!(plot_charges[1,1], ylabel = "Probability")

    for ρt_range in ρt_range_list
        exp_value_t = [expectation_value(ρ, p0_measurement) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(plot_charges[1,1], t_range, exp_value_t, legend = false)
    end
    plot!(plot_charges[1,2], title = "1 electron")
    for ρt_range in ρt_range_list
        exp_value_t = [expectation_value(ρ, p1_measurement) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(plot_charges[1,2], t_range, exp_value_t, legend = false)
    end

    plot!(plot_charges[1,3], title = "2 electrons")
    for (i, ρt_range) in enumerate(ρt_range_list)
        exp_value_t = [expectation_value(ρ, p2_measurement) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(plot_charges[1,3], t_range, exp_value_t, label = "State: $(state_labels[i])", legend = :topright)
    end
    display(plot_charges)
end


function plot_charge_exp(nbr_dots_res, qn_res, sub_plot)
    hams = get_hams(nbr_dots_res)
    t_range, ρt_range_list, state_labels, qd_system =  get_states(hams, nbr_dots_res, qn_res)

    coord = qd_system.coordinates_main[1]
    p0_measurement = matrix_representation(p0(coord, qd_system.f), qd_system.H_total)
    p1_measurement = matrix_representation(p1(coord, qd_system.f), qd_system.H_total)
    p2_measurement = matrix_representation(p2(coord, qd_system.f), qd_system.H_total)

    plot!(sub_plot[qn_res+1,1], title = "P(0) - $(qn_res) e in reservoir")

    for ρt_range in ρt_range_list
        exp_value_t = [expectation_value(ρ, p0_measurement) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(sub_plot[qn_res+1,1], t_range, exp_value_t, legend = false)
    end

    plot!(sub_plot[qn_res+1,2], title = "P(1) - $(qn_res) e in reservoir")
    for ρt_range in ρt_range_list
        exp_value_t = [expectation_value(ρ, p1_measurement) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(sub_plot[qn_res+1,2], t_range, exp_value_t, legend = false)
    end

    plot!(sub_plot[qn_res+1,3], title = "P(2) - $(qn_res) e in reservoir")
        for (i, ρt_range) in enumerate(ρt_range_list)
        exp_value_t = [expectation_value(ρ, p2_measurement) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(sub_plot[qn_res+1,3], t_range, exp_value_t, label = "State: $(state_labels[i])", legend = :topright)
    end
end

function plot_charge_exp(nbr_dots_res)
    plot_charges_vary_qn = plot(layout = (nbr_dots_res*2+1, 3), size = (1500, 1500))
    for qn_res in 0:nbr_dots_res*2
        plot_charge_exp(nbr_dots_res, qn_res, plot_charges_vary_qn)
    end
    plot!(plot_charges_vary_qn, suptitle = "$(nbr_dots_res) QD in reservoir")
    display(plot_charges_vary_qn)
end
#plot_charge_exp(2,2)
plot_charge_exp(2)