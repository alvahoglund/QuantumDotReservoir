get_states(nbr_states) = hcat([reshape(hilbert_schmidt_ensamble(4), 16,1) for i in 1:nbr_states]...)

function measure_spin(op_symbolic, states, qd_system)
    ind_qn = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)
    op_matrix = pauli_string(op_symbolic[1], op_symbolic[2], qd_system)[ind_qn,ind_qn]
    op_vec = reshape(op_matrix', 1, 16)
    spin_measurements = op_vec*states
    return op_vec, spin_measurements
end
function measure_charge(states, qd_system, t_list, measurement_type)
    seed =1 
    S = scrambling_map(qd_system, t_list, measurement_type, seed)
    measurements_outcomes = S*states
    return measurements_outcomes, S
end
function generate_data(nbr_dots_main, nbr_dots_res, qn_reservoir, nbr_states, t_list, op_symbolic)
    qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
    states = get_states(nbr_states)
    op_vec, spin_measurements = measure_spin(op_symbolic, states, qd_system)
    measurement_type = charge_measurements
    measurements_outcomes, S = measure_charge(states, qd_system, t_list, measurement_type)
    op_R = op_vec*pinv(S)
    return measurements_outcomes, spin_measurements, op_R, S
end
