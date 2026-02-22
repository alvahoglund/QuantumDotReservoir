struct ExtendedResState{T} ρ::T end

effective_measurement(op_qn, Ψ_res :: AbstractArray, qd_system) = 
    effective_measurement(op_qn, extend_res_state(Ψ_res, qd_system), qd_system)

function effective_measurement(op_qn, ρ_reservoir_tot ::ExtendedResState, qd_system)
    ind = indices(qd_system.H_total_qn, qd_system.H_total)
    ρ_reservoir_tot_qn = ρ_reservoir_tot.ρ[ind, ind]
    op_rho_qn = ρ_reservoir_tot_qn*op_qn

    op_rho = spzeros(Complex{Float64}, dim(qd_system.H_total), dim(qd_system.H_total))
    op_rho[ind, ind] = op_rho_qn

    op_rho_tr = partial_trace(op_rho, qd_system.H_total => qd_system.H_main; complement = qd_system.H_reservoir, alg = FermionicHilbertSpaces.FullPartialTraceAlg())
    ind = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)
    return op_rho_tr[ind, ind]
end


function extend_res_state(ψ_res_qn, qd_system)
    ρ_res_qn = if ψ_res_qn isa AbstractVector
        ψ_res_qn * ψ_res_qn'
    else
        ψ_res_qn
    end

    ind_res = indices(qd_system.H_reservoir_qn, qd_system.H_reservoir)
    ρ_reservoir = spzeros(Complex{Float64}, dim(qd_system.H_reservoir), dim(qd_system.H_reservoir))
    ρ_reservoir[ind_res, ind_res] = ρ_res_qn
    ρ_reservoir_tot = embed(ρ_reservoir, qd_system.H_reservoir => qd_system.H_total; complement = qd_system.H_main)
    return ExtendedResState(ρ_reservoir_tot)
end