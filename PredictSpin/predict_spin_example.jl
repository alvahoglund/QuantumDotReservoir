includet("predict_spin.jl")

## ======= DEFINE PARAMETERS =========
#Reservoir
nbr_dots_res = 2
qn_reservoir = 1
seed = 1

#Measurements
noise_std = 0.000000001
t_list = [10, 20, 30]
measurement_type = charge_probabilities
op = (σx, σ0)

#Training and test
nbr_states = 2*10000
regularization = 0
nbr_states_train = nbr_states ÷ 2
## ======================================

states = get_states(nbr_states)
qd_system, S = get_S(nbr_dots_res, qn_reservoir, t_list, seed, charge_probabilities)
op_vec, spin_exp_val, charge_exp_val = measure_spin_charge(states, qd_system, S, op, noise_std)
X_train, X_test, Y_train, Y_test = transform_data(spin_exp_val, charge_exp_val, nbr_states_train)
W = ridge_regression(X_train, Y_train, regularization)
