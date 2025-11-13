includet("generate_data.jl")
function ridge_regression(X, Y, λ)
    _, d = size(X)
    k = size(Y)
    ey = Matrix(I, d, d)
    
    W = pinv(X' * X + λ * I) * X' * Y
    return W
end

function predict_spin(nbr_states, op)
    t_list = [10, 20, 30]
    nbr_dots_main, nbr_dots_res, qn_reservoir = (2,2,1)
    measurements_outcomes, spin_measurements, op_R, S = generate_data(nbr_dots_main, nbr_dots_res, qn_reservoir, nbr_states, t_list,op)
    
    nbr_train = Int64(nbr_states ÷ 2)

    X_train = measurements_outcomes[:, 1:nbr_train]'
    Y_train = spin_measurements[:, 1:nbr_train]'
    λ = 0
    W = ridge_regression(X_train, Y_train, λ)'

    X_test = measurements_outcomes[:, nbr_train+1:end]'
    Y_test = spin_measurements[:, nbr_train+1:end]'

    predictions_train = real(W*X_train')
    mse_train = mean((real(predictions_train - Y_train')).^2)
    predictions_test = real(W*X_test')
    mse_test = mean((real(predictions_test - Y_test')).^2)
    op_diff =mean((real(W-op_R)).^2)
    return mse_train, mse_test, op_diff
end

nbrs = range(10, 200, 50)
op = (σx, σ0)
predictions = [predict_spin(nbr, op) for nbr in nbrs]
mse_test_list = getindex.(predictions, 2)
op_diff_list = getindex.(predictions, 3)
plot(nbrs .÷ 2, mse_test_list, ls=:dot, yaxis=:log, xlabel = "Number states in training data", ylabel = "MSE", label = "1/Ntest ∑(<o> - o_pred)^2",legendfont=font(10))
plot!(nbrs .÷ 2, op_diff_list, label = "1/d ∑((o'|R - W)^2")