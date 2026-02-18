## ===================== Functions ==================================
function getS(qd_system, hams, ρ_res, t_list)
    measurements = map(op -> matrix_representation(op, qd_system.H_total), charge_probabilities(qd_system))
    S = vcat([scrambling_map(qd_system, measurements, ρ_res, matrix_representation(hams.hamiltonian_total, qd_system.H_total), t) for t in t_list]...)
    return S
end

function set_hams(qd_system, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    main_system_parameters = set_dot_params(ϵ_func, ϵb_func, u_intra_func, qd_system.coordinates_main)
    reservoir_parameters = set_dot_params(ϵ_func, ϵb_func, u_intra_func, qd_system.coordinates_reservoir)
    interaction_parameters = set_interaction_params(t_func, t_so_func, u_inter_func, qd_system.coordinates_total)
    hamiltonians(qd_system, main_system_parameters, reservoir_parameters, interaction_parameters)
end

function spec_hams(qd_system)
    ϵ_func() = 0
    ϵb_func() = [0, 1, 0]
    u_intra_func() = 5
    t_func() = 0.1
    t_so_func() = 0.5
    u_inter_func() = 0
    hams = set_hams(qd_system, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
end

## Measure of the robustness
function robustness_scores(S, λ_range)
    F = svd(S)
    return [[robustness_score(reshape(pauli_mat[i, :], 1, 16), λ,F) for λ in λ_range] for i in 1:16]
end

function robustness_score(P_ij, λ, F)
    b = 0.0147 #for HS-ensemble
    #b = 0.05 # For random pure states
    D = (b*λ^2)./(b.*(F.S.^2) .+λ^2)
    real(P_ij*F.V*Diagonal(D)*F.V'*P_ij')[1]
end

## Plot singular values over time
function svd_t(qd_system, hams, ρ_res, t_list)
    t_factor = exp10.(range(log10(10^-4), log10(10^3), length=20))
    #t_factor = range(0.01, 100, length = 20)
    S_list = [getS(qd_system, hams, ρ_res, t_list.*i) for i in t_factor]
    sv_list = [svd(S).S for S in S_list]
    sv_m = hcat(sv_list...)[2:end, :]
    plt_svd_t = plot()
    plot!(plt_svd_t, 
            t_factor, 
            sv_m', 
            label = false,
            xlabel = "Time factor",
            ylabel = "Singular values",
            xaxis = :log,)
    display(plt_svd_t)
end

# Print overlap with different pauli strings
function print_overlap(S, singular_value_index, qd_system)
    overlap_m = overlap_matrix(S, qd_system)
    println("Overlap between pauli strings and right singular vector with index $(singular_value_index) ")
    for i in 1:16
        println("$(pauli_string_labels()[i]): $(overlap_m[singular_value_index,i])")
    end 
end
function overlap_matrix(S, qd_system)
    F = svd(S)
    pauli_mat = pauli_string_matrix(qd_system)
    overlap_m = abs2.(real.(Matrix(F.V'*pauli_mat'))) 
    overlap_m
end

#Plot singular values of S with varying system size
function plot_sv()
    nbr_dots_res_list = [2,3,4,5,6]
    qn_reserovoir = 1
    t_list = [1, 2].*1000
    sv_list = []
    for nbr_dots_res in nbr_dots_res_list
        qd_system = tight_binding_system(2, nbr_dots_res, qn_reserovoir)
        hams = spec_hams(qd_system)
        ρ_res =  eig_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir, 1)
        S = getS(qd_system, hams, ρ_res, t_list)
        append!(sv_list, [svd(S).S])
    end
    sv_plot = plot(size = (600, 400))
    sv_mat = hcat(sv_list...)'[:, 2:end] # remove larges singular value
    plot!(sv_plot, 
        nbr_dots_res_list, 
        sv_mat, 
        marker = 'o',
        title = "Singular values of scrambling map \n $(qn_reserovoir) electron in reservoir, time: $(t_list)",
        xlabel = "Dots in reservoir",
        ylabel = "Singular values", 
        legend = false
    )
    display(sv_plot)
    return sv_list
end
## ===================== Set Parameters ===================================
nbr_dots_res = 3
qn_reserovoir = 1
qd_system = tight_binding_system(2, nbr_dots_res, qn_reserovoir)

#Set hamiltonian parameters
#ϵ_func() = 0
#ϵb_func() = [0, 1, 0]
#u_intra_func() = 5
#
#t_func() = 0.1
#t_so_func() = 0.5
#u_inter_func() = 0
#hams = set_hams(qd_system, ϵ_func, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
hams = hamiltonians(qd_system)

# Set reservoir state
n = 1
ρ_res =  eig_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir, n)

# Choose time steps
t_list = [1].*1000

# Find S
S = getS(qd_system, hams, ρ_res, t_list) 

## =========== Plots ========

#svd_t(qd_system, hams, ρ_res, t_list)
#print_overlap(S, 1, qd_system)
#overlap_matrix(S, qd_system)
plot_sv()