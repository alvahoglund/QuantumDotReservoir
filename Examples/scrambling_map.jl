##############
using QuantumDotReservoir
nbr_dots_main = 2
nbr_dots_res = 6
qn_reservoir = 0
quantum_dot_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
seed = 2
hams = hamiltonians(quantum_dot_system, seed)
reservoir_state = ground_state(hams.hamiltonian_reservoir, quantum_dot_system.H_reservoir, quantum_dot_system.qn_reservoir)
ind = indices(quantum_dot_system.H_main_qn, quantum_dot_system.H_main)

initial_states = [def_state(triplet_plus, quantum_dot_system.H_main, quantum_dot_system.f),
                    def_state(singlet, quantum_dot_system.H_main, quantum_dot_system.f),
                    random_product_state(quantum_dot_system),
                    random_separable_state(3, quantum_dot_system)]

total_states = map(initial_state -> tensor_product((initial_state, reservoir_state), (quantum_dot_system.H_main, quantum_dot_system.H_reservoir)=> quantum_dot_system.H_total; physical_algebra = true), initial_states);
measurements = map(op -> matrix_representation(op, quantum_dot_system.H_total), charge_measurements(quantum_dot_system))

t = 10
ham_total = matrix_representation(hams.hamiltonian_total,quantum_dot_system.H_total)

time_evolved_states = map(total_state -> state_time_evolution(total_state, t, ham_total, quantum_dot_system.H_total, quantum_dot_system.qn_total), total_states)
time_evolved_measurements = map(measurement -> operator_time_evolution(measurement, t, ham_total, quantum_dot_system.qn_total, quantum_dot_system.H_total), measurements)
effective_measurements = map(measurement -> effective_measurement(measurement, reservoir_state, quantum_dot_system), time_evolved_measurements)
@time sm = scrambling_map(quantum_dot_system, measurements, reservoir_state, ham_total, t);

measured_values = map(m -> expectation_value(time_evolved_states[3], m), measurements)
if nbr_dots_res≥6
    reshape(inv(sm)*measured_values, 4,4) ≈ initial_states[3][ind,ind]
end
###############