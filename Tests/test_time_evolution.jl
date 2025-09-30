
@testitem "State and operator evolution" begin 
    using FermionicHilbertSpaces, LinearAlgebra,Random
    
    ρ = rand(ComplexF64, 4,4) + hc
    ρ = ρ./tr(ρ)
    ham = (rand(ComplexF64,4,4) +hc)/2
    op = (rand(ComplexF64, 4,4) +hc)/2
    t = 1.0

    ρt = state_time_evolution(ρ, t, ham)
    exp_value_ρ = expectation_value(ρt,op)

    op_t = operator_time_evolution(op, t, ham)
    exp_value_op = expectation_value(ρ,op_t)

    @test exp_value_op≈ exp_value_ρ
end


@testitem "Effective measurements" begin
    @fermions f
    Random.seed!(2)
    quantum_dot_system = tight_binding_system(2,3,1)
    hams = hamiltonians(quantum_dot_system)

    ham1 = matrix_representation(hams[1], quantum_dot_system.H_main)
    ρ1 = ground_state(ham1)
    ham2 = matrix_representation(hams[2], quantum_dot_system.H_reservoir)
    ρ2 = ground_state(ham2)
    
    op = matrix_representation(nbr_op((1,1),quantum_dot_system.f), quantum_dot_system.H_total)
    
    ρ12 = tensor_product((ρ1, ρ2), (quantum_dot_system.H_main, quantum_dot_system.H_reservoir)=> quantum_dot_system.H_total)
    exp_value = expectation_value(ρ12, op)

    op_eff = effective_measurement(op, ρ2, quantum_dot_system.H_main, quantum_dot_system.H_reservoir, quantum_dot_system.H_total)
    exp_value_eff = expectation_value(ρ1, op_eff)
    @test exp_value ≈ exp_value_eff
end

