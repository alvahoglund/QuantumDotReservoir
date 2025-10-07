## Define the system
nbr_dots_main = 2
nbr_dots_res = 6
qn_reservoir = 0
qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)

hams = hamiltonians(qd_system)
ρ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)

# Define some different initial states 
ρ1 = def_state(triplet_minus, qd_system.H_main, qd_system.f)
ρ2 = random_product_state(qd_system)

## Time evolove total state
ρ1_tot = tensor_product((ρ1, ρ_res), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total)
ρ2_tot = tensor_product((ρ2, ρ_res), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total)

t=1
ρ1_tot_t = state_time_evolution(ρ1_tot, t, matrix_representation(hams.hamiltonian_total, qd_system.H_total), 
                                qd_system.H_total, qd_system.qn_total)
ρ2_tot_t = state_time_evolution(ρ2_tot, t, matrix_representation(hams.hamiltonian_total, qd_system.H_total), 
                                qd_system.H_total, qd_system.qn_total)

#Define the measurement and measure
measurements =  map(op -> matrix_representation(op, qd_system.H_total), charge_measurements(qd_system))

measurement_values_1 = map(m->expectation_value(m, ρ1_tot_t), measurements)
measurement_values_2 = map(m->expectation_value(m, ρ2_tot_t), measurements)

## Scrambling map
sm = scrambling_map(qd_system, measurements, ρ_res, matrix_representation(hams.hamiltonian_total, qd_system.H_total), t)

## Compare measurement on time evolved states and with scrambling map
scrambling_map_values1 = Vector(real(sm*vec(ρ1[ind, ind])))
measurement_values_1 ≈ scrambling_map_values1 # True

scrambling_map_values2 = Vector(real(sm*vec(ρ2[ind, ind])))
measurement_values_2 ≈ scrambling_map_values2 ## true(?)
measurement_values_2 .-scrambling_map_values2

## Find recovery map
rm = inv(sm)
rm*sm ≈ Matrix(I,16,16)

ρ1_recovered = reshape(rm*measurement_values_1, 4, 4)
ρ1_recovered ≈ ρ1[ind,ind] # True

ρ2_recovered = reshape(rm*measurement_values_2, 4,4)
ρ2_recovered ≈ ρ2[ind,ind] # True
