function generate_separable_states(qd_system, dataset_size)
    sizes = dataset_size ÷ 3
    sep_1 = [random_separable_state(1, qd_system) for i in 1:sizes]
    sep_2 = [random_separable_state(2, qd_system) for i in 1:sizes]
    sep_3 = [random_separable_state(3, qd_system) for i in 1:(dataset_size-sizes*2)]
    return vcat(sep_1, sep_2, sep_3)
end

function generate_entangled_states(state_name, qd_system, dataset_size)
    p_min, p_max = 0, 0.5
    p_range = range(p_min, p_max, dataset_size)
    return [werner_state(state_name, p, qd_system.H_main, qd_system.f) for p in p_range]
end

pauli_string(σi, σj, qd_system) = tensor_product((matrix_representation(σi(qd_system.coordinates_main[1], qd_system.f), qd_system.H_main_a), 
                matrix_representation(σj(qd_system.coordinates_main[2], qd_system.f), qd_system.H_main_b)),
                (qd_system.H_main_a, qd_system.H_main_b) => qd_system.H_main)

pauli_strings_short(qd_system) = [pauli_string(σj, σj, qd_system) for σj in [σx, σy, σz]]
pauli_strings_long(qd_system) = [pauli_string(σi, σj, qd_system) for σi in [σ0, σx, σy, σz] for σj in [σ0, σx, σy, σz]]

function generate_dataset(qd_system, nbr_sep_states :: Integer, nbr_ent_states :: Integer, entangled_state)
    measurements = pauli_strings_short(qd_system)

    separable_states = generate_separable_states(qd_system, nbr_sep_states)
    entangled_states = generate_entangled_states(entangled_state, qd_system, nbr_ent_states)

    measurements_entangled = [expectation_value(state, measurement) for state in entangled_states, measurement in measurements]
    measurements_separable = [expectation_value(state, measurement) for state in separable_states, measurement in measurements]
    measurement_values = vcat(measurements_entangled, measurements_separable)
    labels = vcat([-1 for i in 1:nbr_ent_states], [1 for i in 1:nbr_ent_states])
    return measurement_values, labels
end
