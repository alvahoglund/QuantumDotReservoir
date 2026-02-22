function effective_measurement(op, ρ_reservoir, H_main_qn, H_reservoir, H_total)
    H_main = hilbert_space(keys(H_main_qn), NumberConservation()) 
    ρ_reservoir_tot = embed(ρ_reservoir, H_reservoir => H_total; complement = H_main)
    op_rho = partial_trace(op*ρ_reservoir_tot, H_total => H_main; complement = H_reservoir, alg = FermionicHilbertSpaces.FullPartialTraceAlg())
    ind = FermionicHilbertSpaces.indices(H_main_qn, H_main)
    return op_rho[ind, ind]
end

effective_measurement(op, ρ_reservoir, qd_system ::QuantumDotSystem) = 
    effective_measurement(op, ρ_reservoir, qd_system.H_main_qn, qd_system.H_reservoir,qd_system.H_total)