get_states(nbr_states) = hcat([reshape(hilbert_schmidt_ensamble(4), 16,1) for i in 1:nbr_states]...)

function vectorize_operator(op_symbolic, qd_system)
    ind_qn = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)
    op_matrix = pauli_string(op_symbolic[1], op_symbolic[2], qd_system)[ind_qn,ind_qn]
    return reshape(op_matrix', 1, 16)
end

function measure_spin(op_symbolic, states, qd_system)
    # Measure spin of the initial quantum state
    op_vec = vectorize_operator(op_symbolic, qd_system)
    spin_measurements = op_vec*states
    return op_vec, spin_measurements
end

function measure_charge(S, states, noise_std)
    # Measure charge of the evolved quantum states
    measurement_outcomes = S*states
    noise = [rand(Normal(0, noise_std)) for m in measurement_outcomes]
    measurement_outcomes += noise
    return measurement_outcomes
end

function get_S(nbr_dots_res, qn_reserovoir, t_list, seed, measurement_type)
    # Define the quantum dot system and find the scrambling map S
    qd_system = tight_binding_system(2, nbr_dots_res, qn_reserovoir)
    S = scrambling_map(qd_system, t_list, measurement_type, seed)
    return qd_system, S
end

ridge_regression(X,Y, λ) = pinv(X' * X + λ * I) * X' * Y

function measure_spin_charge(states, qd_system, S, op_symbolic, noise_std)
    op_vec, spin_exp_val = measure_spin(op_symbolic, states, qd_system)
    charge_exp_val = measure_charge(S, states, noise_std)
    return op_vec, spin_exp_val, charge_exp_val
end

operator_recovery_map(op_symbolic, S, qd_system) = vectorize_operator(op_symbolic, qd_system)*pinv(S) 
    
function transform_data(spin_exp_val, charge_exp_val, training_size)
    # Transform from shape (features x data_points) ie (charge_measurements x nbr states) 
    # to (data_points x features) as expected of a feature matrix
    X = real(charge_exp_val)'
    Y = real(spin_exp_val)'
    X_train, X_test =  X[1:training_size, :], X[training_size+1:end, :]
    Y_train, Y_test =  Y[1:training_size], Y[training_size+1:end]
    return X_train, X_test, Y_train, Y_test
end

function normalize(X_train, X_test)
    μ = mean(X_train, dims=1)           
    σ = std(X_train, dims=1, corrected=true) 
    σ[σ .== 0] .= 1

    X_train_norm = (X_train .- μ) ./ σ
    X_test_norm  = (X_test  .- μ) ./ σ
    
    return X_train_norm, X_test_norm

end