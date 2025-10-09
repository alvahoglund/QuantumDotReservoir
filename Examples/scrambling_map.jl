##############
nbr_dots_main = 2
nbr_dots_res = 6
qn_reservoir = 0
qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)

quantum_dot_system = tight_binding_system(2,6,0)
seed = 2
hams = hamiltonians(quantum_dot_system, seed)
reservoir_state = ground_state(hams.hamiltonian_reservoir, quantum_dot_system.H_reservoir, quantum_dot_system.qn_reservoir)

initial_states = [def_state(triplet_plus, quantum_dot_system.H_main, quantum_dot_system.f),
                    def_state(singlet, quantum_dot_system.H_main, quantum_dot_system.f),
                    random_product_state(quantum_dot_system),
                    random_separable_state(3, quantum_dot_system)]

total_states = map(initial_state -> tensor_product((initial_state, reservoir_state), (quantum_dot_system.H_main, quantum_dot_system.H_reservoir)=> quantum_dot_system.H_total), initial_states)
measurements = map(op -> matrix_representation(op, quantum_dot_system.H_total), charge_measurements(quantum_dot_system))

t = 10
ham_total = matrix_representation(hams.hamiltonian_total,quantum_dot_system.H_total)

time_evolved_states = map(total_state -> state_time_evolution(total_state, t, ham_total, quantum_dot_system.H_total, quantum_dot_system.qn_total), total_states)
time_evolved_measurements = map(measurement -> operator_time_evolution(measurement, t, ham_total, quantum_dot_system.qn_total, quantum_dot_system.H_total), measurements)
effective_measurements = map(measurement -> effective_measurement(measurement, reservoir_state, quantum_dot_system), time_evolved_measurements)
sm = scrambling_map(qd_system, measurements, ρ_res, matrix_representation(hams.hamiltonian_total, qd_system.H_total), t)

Matrix(effective_measurements[1])
sm


scrambling_map_temp = vcat([vec(m) for m in effective_measurements]...)

expectation_value(initial_states[3][ind, ind], effective_measurements[1]) 
transpose(sm[1, :]) * vec(initial_states[3][ind,ind])


(sm * vec(initial_states[3][ind,ind]))[1]

###############################################

prop = propagator(t, ham_tot, qd_system.qn_total, qd_system.H_total)

process_measurements = op -> effective_measurement(
    operator_time_evolution(sparse(prop), sparse(op)), ρ_res, qd_system
) 
eff_measurements = map(process_measurements, measurements)
scrambling_map_temp2 = vcat([vec(m) for m in eff_measurements]...)