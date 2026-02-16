function scrambling_map(qd_system :: QuantumDotSystem, measurements, ρ_res, hamiltonian_total, t)
    prop = propagator(t, hamiltonian_total, qd_system.qn_total, qd_system.H_total)

    process_measurements = op -> effective_measurement(
        operator_time_evolution(sparse(prop), sparse(op)), ρ_res, qd_system
    ) 
    eff_measurements = map(process_measurements, measurements)
    scrambling_map = vcat([vec(m)' for m in eff_measurements]...)
    return scrambling_map
end

scrambling_map(qd_system, t::Number, seed = nothing) = scrambling_map(qd_system, t, charge_measurements, seed) 
scrambling_map(qd_system, t_list::AbstractArray, seed = nothing) = scrambling_map(qd_system, t_list, charge_measurements, seed) 

function scrambling_map(qd_system, t::Number, measurement_types, seed = nothing)    
    isnothing(seed) ? hams = hamiltonians(qd_system) : hams = hamiltonians(qd_system, seed)
    hamiltonian_total = matrix_representation(hams.hamiltonian_total, qd_system. H_total)
    ρ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)
    measurements = map(op -> matrix_representation(op, qd_system.H_total), measurement_types(qd_system))
    return scrambling_map(qd_system, measurements, ρ_res, hamiltonian_total, t)
end

function scrambling_map(qd_system, t_list :: AbstractArray, measurement_types, seed = nothing)
    isnothing(seed) ? hams = hamiltonians(qd_system) : hams = hamiltonians(qd_system, seed)
    hamiltonian_total = matrix_representation(hams.hamiltonian_total, qd_system. H_total)
    ρ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)
    measurements = map(op -> matrix_representation(op, qd_system.H_total), measurement_types(qd_system))
    scrambling_maps = [scrambling_map(qd_system, measurements, ρ_res, hamiltonian_total, t) for t in t_list]
    return vcat(scrambling_maps...)
end