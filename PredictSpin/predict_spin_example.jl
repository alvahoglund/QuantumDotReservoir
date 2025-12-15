includet("predict_spin.jl")

## ======= DEFINE PARAMETERS =========
#Reservoir
nbr_dots_res = 3
qn_reservoir =1
seed = 2

#Measurements
noise_std = 0.001
t_list = [10, 20]
measurement_type = correlated_measurements
op = (σx, σz)

#Training and test
nbr_states = 2*10000
regularization = 0
nbr_states_train = nbr_states ÷ 2
## ======================================

states = get_states(nbr_states)
qd_system, S = get_S(nbr_dots_res, qn_reservoir, t_list, seed, measurement_type)
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
