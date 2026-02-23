
@testset "State and operator evolution" begin
    using FermionicHilbertSpaces, LinearAlgebra, Random

    ρ = rand(ComplexF64, 4, 4) + hc
    ρ = ρ ./ tr(ρ)
    ham = (rand(ComplexF64, 4, 4) + hc) / 2
    op = (rand(ComplexF64, 4, 4) + hc) / 2
    t = 1.0

    ρt = state_time_evolution(t, ρ, ham)
    exp_value_ρ = expectation_value(ρt, op)

    op_t = operator_time_evolution(t, op, ham)
    exp_value_op = expectation_value(ρ, op_t)

    @test exp_value_op ≈ exp_value_ρ
end
@testset "Effective measurements & time evolution" begin
    using SparseArrays
    ## Initialize system
    qd_system = tight_binding_system(2, 3, 1)
    seed = 2
    hams = hamiltonians(qd_system, seed)
    reservoir_state = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir_qn)

    ind = indices(qd_system.H_reservoir_qn, qd_system.H_reservoir)
    ρ_res = spzeros(ComplexF64, dim(qd_system.H_reservoir), dim(qd_system.H_reservoir))
    ρ_res[ind, ind] = reservoir_state*reservoir_state'
    
    initial_states = [def_state(triplet_plus, qd_system.H_main, qd_system.f),
        def_state(singlet, qd_system.H_main, qd_system.f),
        random_product_state(qd_system),
        random_separable_state(3, qd_system)]
    
    t = 10
    #Time evolve the states in the full hilbert space
    ham_total = matrix_representation(hams.hamiltonian_total, qd_system.H_total)
    total_states = map(initial_state -> tensor_product((initial_state, ρ_res), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total), initial_states)
    time_evolved_states = map(total_state -> state_time_evolution(t, total_state, ham_total), total_states)
    measurements_total = map(op -> matrix_representation(op, qd_system.H_total), charge_measurements(qd_system))

    # Construct effective measurements
    time_evolved_measurements = map(op -> operator_time_evolution(t, op, hams.hamiltonian_total, qd_system.H_total_qn), charge_measurements(qd_system))
    effective_measurements = map(measurement -> effective_measurement(measurement, reservoir_state, qd_system), time_evolved_measurements)

    scrambling_block = scrambling_map(qd_system, charge_measurements(qd_system), reservoir_state, hams.hamiltonian_total, t, QRC.BlockPropagatorAlg())

    ham_total_qn = matrix_representation(hams.hamiltonian_total, qd_system.H_total_qn)
    scrambling_pure =  scrambling_map(qd_system, charge_measurements(qd_system), reservoir_state, hams.hamiltonian_total, t, QRC.PureStatePropagatorAlg())
    
    ind_tot_qn = FermionicHilbertSpaces.indices(qd_system.H_total_qn, qd_system.H_total)
    ind_main_qn = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)

    for i in eachindex(initial_states), j in eachindex(measurements_total)
        exp_val_1 = expectation_value(time_evolved_states[i], measurements_total[j])
        exp_val_2 = expectation_value(total_states[i][ind_tot_qn, ind_tot_qn], time_evolved_measurements[j])
        exp_val_3 = expectation_value(initial_states[i][ind_main_qn, ind_main_qn], effective_measurements[j])
        exp_val_4 = (scrambling_block*vec(initial_states[i][ind_main_qn, ind_main_qn]))[j]
        exp_val_5 = (scrambling_pure*vec(initial_states[i][ind_main_qn, ind_main_qn]))[j]
        @test exp_val_1 ≈ exp_val_2 ≈ exp_val_3 ≈ exp_val_4 ≈ exp_val_5
    end
end

