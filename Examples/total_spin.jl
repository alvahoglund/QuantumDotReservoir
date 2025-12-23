function singlet_triplets_spin(qd_system)

    ρ_singlet = def_state(singlet, qd_system.H_main, f)
    ρ_triplet_minus = def_state(triplet_minus, qd_system.H_main, f)
    ρ_triplet = def_state(triplet_0, qd_system.H_main, f)
    ρ_triplet_plus = def_state(triplet_plus, qd_system.H_main, f)

    s2_op = total_spin_op(qd_system.coordinates_main, qd_system.f, qd_system.H_main)

    println("Total spin (S) of S: $(s_from_s2(expectation_value(ρ_singlet, s2_op)))")
    println("Total spin (S) of T+: $(s_from_s2(expectation_value(ρ_triplet_minus, s2_op)))")
    println("Total spin (S) of T0: $(s_from_s2(expectation_value(ρ_triplet, s2_op)))")
    println("Total spin (S) of T-: $(s_from_s2(expectation_value(ρ_triplet_plus, s2_op)))")
end

function init_state_spin(qd_system)
    seed = 5
    hams = hamiltonians(qd_system, seed)
    
    ρ0 = def_state(triplet_0, qd_system.H_main, qd_system.f)
    ρ_res =  ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir, qd_system.qn_reservoir)
    ρ_total = tensor_product((ρ0, ρ_res), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total)

    s2_op = total_spin_op(qd_system.coordinates_total, qd_system.f, qd_system.H_total)
    s2_val = expectation_value(ρ_total, s2_op)
    println("Total spin (S^2) of initial state: $(s2_val)")
end

nbr_dots_main = 2
nbr_dots_res = 2
qn_res = 2
qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_res)

singlet_triplets_spin(qd_system)
init_state_spin(qd_system)