get_states(nbr_states) = hcat([reshape(hilbert_schmidt_ensamble(4), 16,1) for i in 1:nbr_states]...)

function measure_spin(op, states, qd_system)
    op = matrix_representation(op((1,1),qd_system.f), qd_system.H_main_qn)
    op = reshape(op', 1, 16)
    spin_measurements = op*states
    return op, spin_measurements
end
function measure_charge(states, qd_system, t_list)
    seed =1 
    S = scrambling_map(qd_system, t_list, charge_measurements, seed)
    measurements_outcomes = S*states
    return measurements_outcomes, S
end
function generate_data(nbr_dots_main, nbr_dots_res, qn_reservoir, nbr_states, t_list, op)
    qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
    states = get_states(nbr_states)
    op, spin_measurements = measure_spin(op, states, qd_system)
    measurements_outcomes, S = measure_charge(states, qd_system, t_list)
    op_R = op*pinv(S)
    return measurements_outcomes, spin_measurements, op_R, S
end
