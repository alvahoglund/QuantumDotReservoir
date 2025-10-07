function scrambling_map(qd_system :: QuantumDotSystem, measurements, ρ_res, hamiltonian_total, t)
    prop = propagator(t, hamiltonian_total, qd_system.qn_total, qd_system.H_total)

    process_measurements = op -> effective_measurement(
        operator_time_evolution(sparse(prop), sparse(op)), ρ_res, qd_system
    ) 
    eff_measurements = map(process_measurements, measurements)
    scrambling_map = vcat([vec(m)' for m in eff_measurements]...)
    return scrambling_map
end

function scrambling_map(qd_system, t)    
    hams = hamiltonians(qd_system)
    hamiltonian_total = matrix_representation(hams.hamiltonian_total, qd_system. H_total)
    ρ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)
    measurements = map(op -> matrix_representation(op, qd_system.H_total), charge_measurements(qd_system))
    return scrambling_map(qd_system, measurements, ρ_res, hamiltonian_total, t)
end


function pauli_string_map(qd_system)
    measurements = pauli_strings(qd_system)
    ind = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)
    measurements = map(op -> op[ind,ind], measurements) 
    pauli_string_map = vcat([vec(m)' for m in measurements]...)
    return pauli_string_map
end