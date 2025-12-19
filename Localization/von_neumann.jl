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

function plot_entropy(hams, nbr_dots_res, qn_res, plot_S_sub)
    t_range, ρt_range, qd_system = get_states(hams, nbr_dots_res, qn_res)
    Sρ_list =[]
    # Plot entropy of each dot subsystem
    for coordinate in qd_system.coordinates_total
        Sρ = map(ρ -> von_neumann(partial_trace_dot(ρ, [coordinate], qd_system)), ρt_range)
        #show_legend = qn_res == nbr_dots_res*2 ? :outerbottom : false
        plot!(plot_S_sub,t_range, Sρ, label= "Coordinate: $(coordinate)",  linewidth= 2, legend=:outerright)
        push!(Sρ_list, Sρ)     
    end

    #Plot average of dot entropies
    Sρ_sum = [sum(Sρ[i] for Sρ in Sρ_list)/length( qd_system.coordinates_total) for i in 1:length(Sρ_list[1])]
    plot!(plot_S_sub, t_range, Sρ_sum, label = "Average dot entropy", color= :grey, linewidth= 2)

    # Plot entropy of main system
    Sρ_main = map(ρ -> von_neumann(partial_trace_dot(ρ, qd_system.coordinates_main, qd_system)), ρt_range)
    plot!(plot_S_sub, t_range, Sρ_main, label = "Main system", linewidth= 2, color = :black)
end

function plot_entropies(hams, nbr_dots_res)
    plot_S = plot(layout = (nbr_dots_res*2+1, 1), size = (700, 1000), xlabel= "t", ylabel = "S(ρ_sub)")
    for qn_res in 0:(nbr_dots_res*2)
        plot_S_sub = plot_S[qn_res+1, 1]
        plot_entropy(hams, nbr_dots_res, qn_res, plot_S_sub)
        title!(plot_S_sub, "electrons in reservoir: $(qn_res)")
    end
    plot!(plot_S, suptitle = "Von Neuman Entropies \n Dots in Reservoir: $(nbr_dots_res) \n")
    display(plot_S)
end

function partial_trace_dot(ρ, coordinates, qd_system)
    H_sub_dot = hilbert_space(labels(coordinates), NumberConservation())
    partial_trace(ρ, qd_system.H_total => H_sub_dot)
end

function von_neumann(ρ)
    λ = real(eigen(ρ).values)
    λ = λ[λ .> 0]
    S = -sum(λ .* log.(λ))
end

function mutual_information(coordinate_A, coordinate_B, ρ, qd_system)
    ρA = partial_trace_dot(ρ, [coordinate_A], qd_system)
    ρB = partial_trace_dot(ρ, [coordinate_B], qd_system)
    ρAB = partial_trace_dot(ρ, [coordinate_A, coordinate_B], qd_system)
    return von_neumann(ρA) + von_neumann(ρB) - von_neumann(ρAB)
end

function plot_mutual_information(hams, nbr_dots_res, qn_res, p_mi_sub)
    t_range, ρt_range, qd_system = get_states(hams, nbr_dots_res, qn_res)
    for (i, coordinate_i) in enumerate(qd_system.coordinates_main)
        for (j, coordinate_j) in enumerate(qd_system.coordinates_reservoir)
            mi = map(ρ -> mutual_information(coordinate_i, coordinate_j, ρ, qd_system), ρt_range)
            plot!(p_mi_sub,t_range, mi, label = "Dot A : $(coordinate_i), Dot B: $(coordinate_j)", legend=:outerright)
        end
    end
end

function plot_mutual_informations(hams, nbr_dots_res)
    p_mi = plot(layout=(nbr_dots_res*2+1, 1), size = (700, 1000))
    for qn_res in 0:nbr_dots_res*2
        p_mi_sub = p_mi[qn_res+1]
        plot_mutual_information(hams, nbr_dots_res, qn_res, p_mi_sub)
    end
    display(p_mi)
end

#S(rhoA) + S(rhoB) - S(rhaAB)
seed = 1
nbr_dots = 2
hams = get_hams(nbr_dots)
plot_entropies(hams, nbr_dots)

#
#nbr_dots = 2
#hams = get_hams(nbr_dots)
#plot_mutual_informations(hams, nbr_dots)
#

