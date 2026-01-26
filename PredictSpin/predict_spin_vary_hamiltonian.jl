includet("predict_spin.jl")

## ======= DEFINE PARAMETERS =========
#Reservoir
nbr_dots_res = 2
qn_reservoir =2
seed = 2

#Measurements
noise_std = 0
op = (σy, σy)

#Training and test
nbr_states = 2*10000
regularization = 0
nbr_states_train = nbr_states ÷ 2
## ======== DEFINE SYSTEM AND CHOOSE HAMILTONIAN ==============
qd_system = tight_binding_system(2, nbr_dots_res, qn_reservoir)

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
hamiltonian_total = matrix_representation(hams.hamiltonian_total, qd_system.H_total)

ρ_res = eig_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir, 4)

measurements = map(op -> matrix_representation(op, qd_system.H_total), charge_probabilities(qd_system))

t = 100
S = scrambling_map(qd_system, measurements, ρ_res, hamiltonian_total, t)

rank(S)
## ==================================
states = get_states(nbr_states)

op_vec, spin_exp_val, charge_exp_val = measure_spin_charge(states, qd_system, S, op, noise_std)
X_train, X_test, Y_train, Y_test = transform_data(spin_exp_val, charge_exp_val, nbr_states_train)
#X_train_norm, X_test_norm = normalize(X_train, X_test)
#W_norm = ridge_regression(X_train_norm, Y_train, regularization)
W = ridge_regression(X_train, Y_train, regularization)

Y_test_pred = X_test*W

idx_sort = sortperm(Y_test)
Y_pred_sorted = Y_test_pred[idx_sort]
Y_true_sorted = Y_test[idx_sort]

Y_pred_sorted[1:10:end]
plot(Y_true_sorted[1:10:end], seriestype=:scatter)
plot!(Y_pred_sorted[1:10:end], seriestype=:scatter)
plot!(title = "Predicting $(join(string.(nameof.(op)), " ⊗"))")

 