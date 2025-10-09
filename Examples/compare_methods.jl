## Define the system
nbr_dots_main = 2
nbr_dots_res = 3
qn_reservoir = 1
qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)

hams = hamiltonians(qd_system)
ρ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)

# Define some different initial states 
ρ1 = def_state(triplet_plus, qd_system.H_main, qd_system.f)
ρ2 = random_product_state(qd_system)

ind = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)
ρ1[ind,ind]

## Time evolove total state
ρ1_tot = tensor_product((ρ1, ρ_res), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total)
ρ2_tot = tensor_product((ρ2, ρ_res), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total)

t=1
ρ1_tot_t = state_time_evolution(ρ1_tot, t, matrix_representation(hams.hamiltonian_total, qd_system.H_total), 
                                qd_system.H_total, qd_system.qn_total)
ρ2_tot_t = state_time_evolution(ρ2_tot, t, matrix_representation(hams.hamiltonian_total, qd_system.H_total), 
                                qd_system.H_total, qd_system.qn_total)

#Define some measurement
op = matrix_representation(nbr_op((2,1), qd_system.f), qd_system.H_total)
op_eff = effective_measurement(op, ρ_res, qd_system.H_main_qn, qd_system.H_reservoir, qd_system.H_total)

# Time evolve measurement
op_t = operator_time_evolution(op, t, matrix_representation(hams.hamiltonian_total, qd_system.H_total))
op_t_eff = effective_measurement(op_t, ρ_res, qd_system.H_main_qn, qd_system.H_reservoir, qd_system.H_total)


# Calculate expectation value without time evolution

testd1 = expectation_value(ρ1_tot, op)
teste1 = expectation_value(ρ1[ind,ind], op_eff)

testd2 = expectation_value(ρ2_tot, op)
teste2 = expectation_value(ρ2[ind,ind], op_eff)

testd1 ≈ teste1 # True
testd2 ≈ teste2 # True

# Calculate expectation value of measurement with time evolution
testa1 = expectation_value(ρ1[ind,ind],op_t_eff)
testb1 = expectation_value(ρ1_tot_t, op)
testc1 = expectation_value(ρ1_tot, op_t)

testa2 = expectation_value(ρ2[ind,ind],op_t_eff)
testb2 = expectation_value(ρ2_tot_t, op)
testc2 = expectation_value(ρ2_tot, op_t)

testa1 ≈ testb1 ≈ testc1 # True
testa2 ≈ testb2 ≈ testc2 # True