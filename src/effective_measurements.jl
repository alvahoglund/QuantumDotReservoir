function effective_meaurement(op, ρ_reservoir, H_main_qn, H_reservoir, H_total)
    H_main = hilbert_space(keys(H_main_qn), NumberConservation()) 
    # Is this the best way to extend the Hilbert space? 
    #Could also input the total main Hilbert space and run sector(qn, H) to get the subsystem, but then I loose the option of having ssub H_main with one particle in each dot.
    
    exp_value(ρ_m) = tr((tensor_product((ρ_m, ρ_reservoir), (H_main, H_reservoir) => H_total))*op)
    
    function func(ρ_main_qn_vec)
        ρ_main_qn = reshape(ρ_main_qn_vec, dim(H_main_qn), dim(H_main_qn))
        
        #Pad matrix
        ρ_main = zeros(Complex{Float64}, dim(H_main), dim(H_main))
        index = sector_index(H_main_qn, H_main)
        ρ_main[index, index] = ρ_main_qn
        
        return exp_value(ρ_main)
    end

    lmap = LinearMaps.LinearMap(func, 1, dim(H_main_qn)^2)
    n_eff = reshape(Matrix{Complex{Float64}}(lmap), dim(H_main_qn),  dim(H_main_qn))
    return n_eff
end