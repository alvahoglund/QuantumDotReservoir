function effective_measurement(op_qn :: AbstractMatrix, ψ_res_qn, qd_system)
    ρ_res_qn = if ψ_res_qn isa AbstractVector
        ψ_res_qn * ψ_res_qn'
    else
        ψ_res_qn
    end
    ρ_reservoir = embed(ρ_res_qn, qd_system.H_reservoir_qn, qd_system.H_reservoir)
    op = embed(op_qn, qd_system.H_total_qn, qd_system.H_total)

    ρ_reservoir_tot = embed(ρ_reservoir, qd_system.H_reservoir => qd_system.H_total; complement = qd_system.H_main)
    op_rho = partial_trace(op*ρ_reservoir_tot, qd_system.H_total => qd_system.H_main; complement = qd_system.H_reservoir, alg = FermionicHilbertSpaces.FullPartialTraceAlg())
    ind = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)
    return op_rho[ind, ind]
end