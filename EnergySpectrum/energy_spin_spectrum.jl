### ======== Find the total hamiltonian of a single QD system =============

function temp_dot_param(coordinates, ϵ_val, ϵb_val, u_intra_val)
    ϵ = Dict(coordinate => ϵ_val for coordinate in coordinates)
    ϵb = Dict(coordinate => ϵb_val for coordinate in coordinates)
    u_intra = Dict(coordinate => u_intra_val for coordinate in coordinates)
    return DotParams(ϵ, ϵb, u_intra)
end

function temp_interaction_param(coordinates, t_val, t_so_val, u_inter_val)
    coupled_coordinates = get_coupled_coordinates(coordinates)
    t = Dict(coupled_coordinate => t_val for coupled_coordinate in coupled_coordinates)
    t_so = Dict(coupled_coordinate => t_so_val for coupled_coordinate in coupled_coordinates)
    u_inter = Dict(coupled_coordinate => u_inter_val for coupled_coordinate in coupled_coordinates)
    InteractionParams(t,t_so,u_inter)
end

function get_eigenbasis(coordinates, qn, t_val, U_val, ϵb_val)
    ϵ_func = 0
    ϵb_func = ϵb_val
    u_intra_func = U_val

    t_func = t_val
    t_so_func = 0
    u_inter_func = 0

    dp = temp_dot_param(coordinates, ϵ_func, ϵb_func, u_intra_func)
    ip = temp_interaction_param(coordinates,t_func,  t_so_func, u_inter_func)

    @fermions f

    H = hilbert_space(labels(coordinates), NumberConservation(qn))

    sys_ham_dot = hamiltonian_dots(dp, coordinates,f) 
    sys_ham_int = hamiltonian_interactions(ip, coordinates, f)
    sys_ham = sys_ham_dot + sys_ham_int
    return eigen(Matrix(matrix_representation(sys_ham , H)))
end

# ================= Plot the energy spectrum and teh total spin of each state =============

function plot_energy_spin_spectrum(coordinates, qn, t_val, U_val, ϵb_val)
    vals, vecs = get_eigenbasis(coordinates, qn, t_val, U_val, ϵb_val)
    vals = real(vals)
    idx = FermionicHilbertSpaces.indices(qn, hilbert_space(labels(coordinates), NumberConservation()))
    s2_op = total_spin_op(coordinates, f, hilbert_space(labels(coordinates), NumberConservation()))

    s_vals = [s_from_s2(expectation_value(vecs[:,i]*vecs[:,i]', s2_op[idx, idx])) for i in 1:length(vals)]

    p_se = plot(xlim = (vals[1]-0.5, vals[end]+0.5), 
                ylim = (0, maximum(s_vals)+1),
                xlabel = "Energy",
                ylabel = "Total Spin", 
                title= "$(length(coordinates)) dots and $(qn) electrons")
    println(vals)
    scatter!(p_se, vals, s_vals, label = "Spin S")
    vline!(p_se, vals, label = "Energy")
    display(p_se)
    plot!(p_se, show_legend = false)
    return vals, vecs
end

## ============= Plot the energy differences in the energy spectrum =============

differences(vals) = [abs(vals[j] - vals[i]) for i in eachindex(vals), j in eachindex(vals) if i<j]

function plot_energy_differences(coordinates, t_val, U_val)
    pd_all = plot(layout = (length(coordinates)*2-1, 1), size = (800, 1100))
    for qn in 1:length(coordinates)*2-1
        plot_energy_difference!(pd_all, coordinates, qn, t_val, U_val)
    end
    plot!(pd_all, suptitle = "$(length(coordinates)) QDs in total")
    display(pd_all)
end

function plot_energy_difference!(pd_all, coordinates, qn, t_val, U_val)
    # Plot Hubbard Hamiltonian energy differences
    vals, vecs = plot_energy_spin_spectrum(coordinates, qn, t_val, U_val, ϵb_val)
    show_legend = qn == 1 ? :topright : false
    plot!(pd_all[qn, 1], legend = show_legend, xlim =(-1, 11), title= "$(qn) electrons",)
    if qn == length(coordinates)*2-1
        plot!(pd_all[qn, 1], xlabel = "Energy level difference")
    end
    vline!(pd_all[qn, 1], differences(vals), label = "Hubbard Hamiltonian Energy Differences", linewidth= 2, color = :black)

    # Plot energy differences of tunneling hamiltonian
    H = hilbert_space(labels(coordinates), NumberConservation(qn))
    ham_t = Matrix(matrix_representation(hamiltonian_t(default_equal_interaction_params(coordinates).t, coordinates, f), H))
    vline!(pd_all[qn, 1], differences(eigen(ham_t).values), label = "Tunneling Hamiltonian Energy Difference", linewidth =3, linestyle = :dash)
    vline!(pd_all[qn, 1], [U_val], label = "U", linewidth =3, linestyle = :dash)
    
    #vline!(pd, [4*t_val^2/U_val], label = "4t^2/U",linewidth=3, linestyle = :dash)
    vline!(pd_all[qn, 1], differences(eigen(Matrix(heisenberg_hamiltonian(coordinates, f, t_val, U_val))).values), label = "Heisenberg Hamiltonian Energy Difference", linewidth =3, linestyle = :dash)
end


function plot_energy_difference(coordinates, qn, t_val, U_val)
    # Plot Hubbard Hamiltonian energy differences
    vals, vecs = plot_energy_spin_spectrum(coordinates, qn, t_val, U_val, ϵb_val)
    pd = plot(xlabel = "Energy difference", xlim =(-1, 11), title= "$(length(coordinates)) dots and $(qn) electrons", size = (700, 400))
    vline!(pd, differences(vals), label = "Hubbard Hamiltonian Energy Differences", linewidth= 2, color = :black)
    
    print("Vals: \n $(vals)\n Differences: \n $(differences(vals))")
    
    # Plot energy differences of tunneling hamiltonian
    H = hilbert_space(labels(coordinates), NumberConservation(qn))
    ham_t = Matrix(matrix_representation(hamiltonian_t(default_equal_interaction_params(coordinates).t, coordinates, f), H))
    vline!(pd, differences(eigen(ham_t).values), label = "Tunneling Hamiltonian Energy Difference", linewidth =3, linestyle = :dash)
    vline!(pd, [U_val], label = "U", linewidth =3, linestyle = :dash)
    
    #vline!(pd, [4*t_val^2/U_val], label = "4t^2/U",linewidth=3, linestyle = :dash)
    vline!(differences(eigen(Matrix(heisenberg_hamiltonian(coordinates, f, t_val, U_val))).values), label = "Heisenberg Hamiltonian Energy Difference", linewidth =3, linestyle = :dash)
    display(pd)
end

function heisenberg_hamiltonian(coordinates, f, t_val, U_val)
    H_total = hilbert_space(labels(coordinates), NumberConservation())
    J = 4*t_val^2/U_val
    Sij_op = sum([-J* Sij(coordinate_i, coordinate_j, f, H_total) 
                for (coordinate_i, coordinate_j) in get_coupled_coordinates(coordinates)
                if coordinate_i ∈ coordinates && coordinate_j ∈ coordinates])
    
    # Reduce to the subspace where we have one electron per dot
    idx = FermionicHilbertSpaces.indices(localized_hilbert_space(coordinates), H_total)
    return Sij_op[idx, idx]
end

function localized_hilbert_space(coordinates)
    # Subspace with one electron per dot
    local_qd_hilbert_spaces = [hilbert_space(labels([coordinate]), NumberConservation(1)) for coordinate in coordinates]
    localized_hilbert_space_temp = local_qd_hilbert_spaces[1]
    for local_qd_hilbert_space in local_qd_hilbert_spaces[2:end]
        localized_hilbert_space_temp = tensor_product(localized_hilbert_space_temp, local_qd_hilbert_space)
    end
    return localized_hilbert_space_temp
end
# ============ Main =========================

@fermions f
coordinates = [(2,1), (2,2), (3,1),(3,2)]
qn = 2
t_val = 1
U_val = 10
ϵb_val = [0,0,0]

vals, vecs = plot_energy_spin_spectrum(coordinates, qn, t_val, U_val, ϵb_val)
#plot_energy_difference(coordinates, qn, t_val, U_val)
#plot_energy_differences(coordinates, t_val, U_val)