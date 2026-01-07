function get_hams(nbr_dots_res)
    nbr_dots_main, nbr_dots_res, qn_reservoir = 2, nbr_dots_res, 0
    qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
    return hamiltonians_equal_param(qd_system)
end

function get_states(hams, nbr_dots_res, qn_res)
    nbr_dots_main, nbr_dots_res, qn_reservoir = 2, nbr_dots_res, qn_res
    qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)

    ρ_main = def_state(triplet_0, qd_system.H_main, qd_system.f)
    ρ_reservoir = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)
    ρ_total = tensor_product((ρ_main, ρ_reservoir), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total)

    t_range = range(0, 3*2π, 1000)
    
    ham_tot = matrix_representation(hams.hamiltonian_total, qd_system.H_total)
    ρt_range = [state_time_evolution(ρ_total, t, ham_tot, qd_system.H_total, qn_res+2) for t in t_range]
    return t_range, ρt_range, qd_system
end

function plot_charge_exp(nbr_dots_res, qn_res)
    hams = get_hams(nbr_dots_res)
    t_range, ρt_range, qd_system =  get_states(hams, nbr_dots_res, qn_res)
    
    p0_measurement_set =[matrix_representation(p0(coordinate, qd_system.f), qd_system.H_total) 
                        for coordinate in qd_system.coordinates_total]
    p1_measurement_set =[matrix_representation(p1(coordinate, qd_system.f), qd_system.H_total) 
                        for coordinate in qd_system.coordinates_total]
    p2_measurement_set =[matrix_representation(p2(coordinate, qd_system.f), qd_system.H_total) 
                        for coordinate in qd_system.coordinates_total]

    plot_charges = plot(layout = (1, 3), size = (700, 500))

    plot!(plot_charges[1,1], title = "0 electrons")
    plot!(plot_charges[1,1], ylabel = "Probability")
    for (i,m) in enumerate(p0_measurement_set)
        exp_value_t = [expectation_value(ρ, m) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(plot_charges[1,1], t_range, exp_value_t, legend = false)
    end

    plot!(plot_charges[1,2], title = "1 electron")
    for (i,m) in enumerate(p1_measurement_set)
        exp_value_t = [expectation_value(ρ, m) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(plot_charges[1,2], t_range, exp_value_t, legend = false)
        
    end

    plot!(plot_charges[1,3], title = "2 electrons")
    for (i,m) in enumerate(p2_measurement_set)
        exp_value_t = [expectation_value(ρ, m) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(plot_charges[1,3], t_range, exp_value_t, label = "Coordinate: $(qd_system.coordinates_total[i])", legend = :topright)
    end
    display(plot_charges)
end


function plot_charge_exp(nbr_dots_res, qn_res, sub_plot)
    hams = get_hams(nbr_dots_res)
    t_range, ρt_range, qd_system =  get_states(hams, nbr_dots_res, qn_res)
    
    p0_measurement_set =[matrix_representation(p0(coordinate, qd_system.f), qd_system.H_total) 
                        for coordinate in qd_system.coordinates_total]
    p1_measurement_set =[matrix_representation(p1(coordinate, qd_system.f), qd_system.H_total) 
                        for coordinate in qd_system.coordinates_total]
    p2_measurement_set =[matrix_representation(p2(coordinate, qd_system.f), qd_system.H_total) 
                        for coordinate in qd_system.coordinates_total]

    plot!(sub_plot[qn_res+1,1], title = "P(0) - $(qn_res) e in reservoir")
    for (i,m) in enumerate(p0_measurement_set)
        exp_value_t = [expectation_value(ρ, m) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(sub_plot[qn_res+1,1], t_range, exp_value_t, legend = false)
    end

    plot!(sub_plot[qn_res+1,2], title = "P(1) - $(qn_res) e in reservoir")
    for (i,m) in enumerate(p1_measurement_set)
        exp_value_t = [expectation_value(ρ, m) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(sub_plot[qn_res+1,2], t_range, exp_value_t, legend = false)
        
    end

    plot!(sub_plot[qn_res+1,3], title = "P(2) - $(qn_res) e in reservoir")
    for (i,m) in enumerate(p2_measurement_set)
        exp_value_t = [expectation_value(ρ, m) for ρ in ρt_range]
        exp_value_t[abs.(exp_value_t) .< 1e-10] .= 0
        plot!(sub_plot[qn_res+1,3], t_range, exp_value_t, label = "Coordinate: $(qd_system.coordinates_total[i])", legend = :topright)
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
plot_charge_exp(2)