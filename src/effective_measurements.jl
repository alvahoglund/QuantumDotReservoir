function effective_measurement(op, ρ_reservoir, H_main_qn, H_reservoir_qn, H_total_qn)
    H_total = pad(H_total_qn)
    H_main = pad(H_main_qn)
    H_reservoir = pad(H_reservoir_qn)

    function exp_value(ρ_m) 
        ρ_tot = tensor_product((ρ_m, ρ_reservoir), (H_main, H_reservoir) => H_total)
        ind = sector_index(H_total_qn, H_total)
        ρ_tot_qn = ρ_tot[ind,ind]
        return tr(ρ_tot_qn*op)
    end

    function func(ρ_main_qn_vec)
        ρ_main_qn = reshape(ρ_main_qn_vec, dim(H_main_qn), dim(H_main_qn))
        
        #Pad matrix
        ρ_main = zeros(Complex{Float64}, dim(H_main), dim(H_main))
        index = sector_index(H_main_qn, H_main)
        ρ_main[index, index] = ρ_main_qn
        
        return exp_value(ρ_main)
    end

    lmap = LinearMaps.LinearMap(func, 1, dim(H_main_qn)^2)
    n_eff = reshape(Matrix{Complex{Float64}}(lmap), dim(H_main_qn), dim(H_main_qn))
    return n_eff
end

effective_measurement(op, ρ_reservoir, qd_system ::QuantumDotSystem) = 
    effective_measurement(op, ρ_reservoir, qd_system.H_main_qn, qd_system.H_reservoir_qn, qd_system.H_total_qn)