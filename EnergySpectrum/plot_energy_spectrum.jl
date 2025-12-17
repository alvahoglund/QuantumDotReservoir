function get_H_Ham(nbr_dots_res)
    nbr_dots_main, nbr_dots_res, qn_reservoir = 2, nbr_dots_res, 0
    qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
    hams = hamiltonians(qd_system)
    ham_tot= matrix_representation(hams.hamiltonian_total, qd_system.H_total)
    return hams, qd_system
end

function get_energy_spectrum(qn_res,hams, qd_system)
    ham_tot = matrix_representation(hams.hamiltonian_total, qd_system.H_total)
    ind = FermionicHilbertSpaces.indices(2+qn_res, qd_system.H_total)
    ham_tot_qn = ham_tot[ind,ind]
    vals, vecs = eigen(Matrix(ham_tot_qn))
    return vals, vecs
end

function get_energy_spectrum(hams, qd_system)
    vals, vecs = eigen(Matrix(matrix_representation(hams.hamiltonian_total, qd_system.H_total)))
    return vals, vecs
end

function get_state_energy(qn_res, hams, qd_system)
    qd_system_qn = tight_binding_system(2, length(qd_system.coordinates_reservoir), qn_res)
    ρ_main = def_state(singlet, qd_system_qn.H_main, qd_system_qn.f)
    ρ_reservoir = ground_state(hams.hamiltonian_reservoir, qd_system_qn.H_reservoir, qn_res)
    ρ_total = tensor_product((ρ_main, ρ_reservoir), (qd_system_qn.H_main, qd_system_qn.H_reservoir) => qd_system_qn.H_total)
    ham_tot = matrix_representation(hams.hamiltonian_total, qd_system.H_total)

    ind_qn = FermionicHilbertSpaces.indices(qn_res+2, qd_system_qn.H_total)
    ρ_tot_qn = ρ_total[ind_qn, ind_qn]
    ham_tot_qn = ham_tot[ind_qn, ind_qn]

    E_exp = real(tr(ham_tot_qn*ρ_tot_qn))
    return ρ_tot_qn, E_exp
end

energy_eigenbasis(vecs, ρ_tot_qn) = [real(vecs[:, i]' * ρ_tot_qn * vecs[:, i]) for i in 1:size(vecs)[1]]

function get_plot_index(nbr_dots_res, qn_res)
    if qn_res ≤ nbr_dots_res
        col = 1
        row = qn_res + 1
    else
        col = 2
        row = 2*(nbr_dots_res + 1) - (qn_res+1)
    end
    return row, col
end

function plot_data!(p_es, subplot_idx, vals, E_exp, c_i2, qn_res)
        xvals = 1:length(vals)
        ymin, ymax = minimum(vals)-0.1, maximum(vals)+0.1
        background = repeat(reshape(c_i2, 1, :), 100, 1)

        # Background heatmap in the subplot
        heatmap!(p_es, xvals, LinRange(ymin, ymax, 100), background,
                 color=cgrad([:white, :purple]),
                 #color = 1.-cgrad(:acton),
                 legend=false,
                 ylims=(ymin, ymax),
                 subplot=subplot_idx)

        # Overlay scatter points
        scatter!(p_es, xvals, vals, color=:white, ms=5,
                 title="Electrons in reservoir: $(qn_res)",
                 subplot=subplot_idx)

        # Horizontal line
        hline!(p_es, [E_exp],
       subplot=subplot_idx,
       color=:grey,        
       linestyle=:dash,    
       linewidth=5)        
end

function plot_spectrum(nbr_dots_res)
    hams, qd_system = get_H_Ham(nbr_dots_res)
    p_es = plot(layout = (nbr_dots_res + 1, 2), size = (1500, 1500))

    for qn_res in 0:2*nbr_dots_res
        row, col = get_plot_index(nbr_dots_res, qn_res)
        subplot_idx = (row - 1) * 2 + col

        vals, vecs = get_energy_spectrum(qn_res, hams, qd_system)
        ρ_tot_qn, E_exp = get_state_energy(qn_res, hams, qd_system)
        c_i2 = energy_eigenbasis(vecs, ρ_tot_qn)

        plot_data!(p_es, subplot_idx, vals, E_exp, c_i2, qn_res)
    end

    plot!(p_es[nbr_dots_res+1, 2], get_energy_spectrum(hams, qd_system)[1],  title = "Total energy spectrum")

    plot!(p_es, suptitle  = " Energy spectrum of Hamiltonian block \n Dots in reservoir: $(nbr_dots_res)", ylabel   = "Eigenenergy")
    display(p_es)
end

plot_spectrum(3)