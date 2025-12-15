includet("predict_spin.jl")

## ======= DEFINE PARAMETERS =========
#Reservoir
nbr_dots_res = 3
qn_reservoir =3

seed = 5

#Measurements
noise_std = 0.0001
t_list = [100, 200]
measurement_types = [charge_measurements, correlated_measurements]
op = (σx, σz)
noise_std_list_long = 10 .^ range(-7, -1, length=30)

#Training and test
nbr_states = 2*10000
regularization = 0
nbr_states_train = nbr_states ÷ 2
## ======================================

states = get_states(nbr_states)
p_1 = plot(layout = (1, 2), size = (700,300))
yaxis!("<σx ⊗ σy>")
for (i, measurement_type) in enumerate(measurement_types)
    qd_system, S = get_S(nbr_dots_res, qn_reservoir, t_list, seed, measurement_type)
    op_vec, spin_exp_val, charge_exp_val = measure_spin_charge(states, qd_system, S, op, noise_std)
    X_train, X_test, Y_train, Y_test = transform_data(spin_exp_val, charge_exp_val, nbr_states_train)
    W = ridge_regression(X_train, Y_train, regularization)

    Y_test_pred = X_test*W

    idx_sort = sortperm(Y_test)
    Y_pred_sorted = Y_test_pred[idx_sort]
    Y_true_sorted = Y_test[idx_sort]

    Y_pred_sorted[1:10:end]
    plot!(p_1[i], Y_true_sorted[1:10:end], seriestype=:scatter)
    plot!(p_1[i], Y_pred_sorted[1:10:end], seriestype=:scatter)
    title!(p_1[i], string(measurement_type))
end

display(p_1)


function prediction_mse(states, qd_system, S, op, noise_std)
    op_vec, spin_exp_val, charge_exp_val = measure_spin_charge(states, qd_system, S, op, noise_std)
    X_train, X_test, Y_train, Y_test = transform_data(spin_exp_val, charge_exp_val, nbr_states_train)
    W = ridge_regression(X_train, Y_train, regularization)
    Y_test_pred = X_test*W
    mse = mean((Y_test -Y_test_pred).^2)
    return mse
end

plt2 = plot(xlabel = "σₐ", ylabel = "MSE",  xaxis=:log, legend=:topleft)
for measurement_type in measurement_types
    qd_system_i, S_i = get_S(nbr_dots_res, qn_reservoir, t_list, seed, measurement_type)
    mse_list = [prediction_mse(states, qd_system_i, S_i, op, noise_std) for noise_std in noise_std_list_long] 
    plot!(plt2, noise_std_list_long, mse_list, label = "$(measurement_type), Cond. nbr.: $(round(cond(S_i), digits = 0))")
end
display(plt2)