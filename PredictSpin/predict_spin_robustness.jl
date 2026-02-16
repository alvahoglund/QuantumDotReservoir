includet("predict_spin.jl")
function prediction_mse(states, qd_system, S, op, noise_std, regularization)
    seed = 123
    op_vec, spin_exp_val, charge_exp_val = measure_spin_charge(states, qd_system, S, op, noise_std, seed)
    X_train, X_test, Y_train, Y_test = transform_data(spin_exp_val, charge_exp_val, nbr_states_train)
    W = ridge_regression(X_train, Y_train, regularization)
    Y_test_pred = X_test*W
    mse = mean((Y_test -Y_test_pred).^2)
    return mse
end

function prediction_mse_norm(states, qd_system, S, op, noise_std, regularization)
    seed = 123
    op_vec, spin_exp_val, charge_exp_val = measure_spin_charge(states, qd_system, S, op, noise_std, seed)
    X_train, X_test, Y_train, Y_test = transform_data(spin_exp_val, charge_exp_val, nbr_states_train)
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    Y_train_norm, Y_test_norm = normalize_data(Y_train, Y_test)
    W_norm = ridge_regression(X_train_norm, Y_train_norm, regularization)
    Y_test_pred_norm = X_test_norm*W_norm
    mse = mean((Y_test_norm -Y_test_pred_norm).^2)
    return mse
end

function robustness_scores(S, λ_range)
    F = svd(S)
    pauli_mat = pauli_string_matrix(qd_system)
    return [[robustness_score(reshape(pauli_mat[i, :], 1, 16), λ,F) for λ in λ_range] for i in 1:16]
end

function robustness_score(P_ij, λ, F)
    b = 0.0147 #for HS-ensemble
    #b = 0.05 # For random pure states
    D = (b*λ^2)./(b.*(F.S.^2) .+λ^2)
    real(P_ij*F.V*Diagonal(D)*F.V'*P_ij')[1]
end

## ======= DEFINE PARAMETERS =========
# Training and test
nbr_states = 2*10000
regularization = 0 
nbr_states_train = nbr_states ÷ 2

#QD system
nbr_dots_res = 2
qn_reservoir =2
qd_system = tight_binding_system(2, nbr_dots_res, qn_reservoir)

# Hamiltonian
ϵ_func() = 0
ϵb_func() = [0, 0, 1]
u_intra_func() = 10

t_func() = 1
t_so_func() = 1
u_inter_func() = 0

main_system_parameters = set_dot_params(ϵ_func, ϵb_func, u_intra_func, qd_system.coordinates_main)
reservoir_parameters = set_dot_params(ϵ_func, ϵb_func, u_intra_func, qd_system.coordinates_reservoir)
interaction_parameters = set_interaction_params(t_func, t_so_func, u_inter_func, qd_system.coordinates_total)
hams = hamiltonians(qd_system, main_system_parameters, reservoir_parameters, interaction_parameters)
hamiltonian_total = matrix_representation(hams.hamiltonian_total, qd_system.H_total)

# Scrambling map
ρ_res = eig_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir, 4)
measurements = map(op -> matrix_representation(op, qd_system.H_total), charge_probabilities(qd_system))
t_list = [10, 20, 30, 40, 50].*10
S = vcat([scrambling_map(qd_system, measurements, ρ_res, hamiltonian_total, t) for t in t_list]...)

#Generate states
#states = get_states(nbr_states, random_pure_states) 
states = get_states(nbr_states, hilbert_schmidt_ensamble) 

## ======== Peform prediction =================
noise_list = exp10.(range(log10(10^-7), log10(1), length=40))
#noise_list = range(10^-7, 10^-5, length = 40)
op_list = [(σi, σj) for σi ∈ [σ0, σx, σy, σz] for σj ∈ [σ0, σx, σy, σz]]
pred_mse = [[prediction_mse(states, qd_system, S, op, noise, regularization) for noise in noise_list] for op in op_list]


plot_mse = plot(layout = (2,2), size = (1300, 1000))
colors  = palette(:auto) 
for (i, op) in enumerate(op_list)
    plot!(plot_mse[(i - 1) ÷ 8 + 1, ((i - 1) ÷ 4) % 2 + 1], 
        noise_list, 
        pred_mse[i].+10^(-12), 
        label = "MSE: $(pauli_string_labels()[i])", 
        legend = :topleft, 
        xaxis = :log,
        yaxis = :log,
        color = colors[(i - 1) % 4 + 1], 
        xlabel = "Standard Deviation of Noise"
    )
end

rs = robustness_scores(S, noise_list)
for (i, op) in enumerate(op_list)
    plot!(plot_mse[(i - 1) ÷ 8 + 1, ((i - 1) ÷ 4) % 2 + 1], 
        noise_list, 
        rs[i], 
        label = "Robustness: $(pauli_string_labels()[i])",
        legend = :topleft, 
        linestyle = :dash,  
        color = colors[(i - 1) % 4 + 1]
    )
end
display(plot_mse)

## ==================================