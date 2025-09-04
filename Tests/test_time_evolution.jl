
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

    @test exp_value_op≈ exp_value_op
end


@testitem "Effective measurements" begin
    @fermions f
    Random.seed!(2)
    H1 = hilbert_space(labels([1,2]))
    H2 = hilbert_space(labels([2,3]))
    H12 = hilbert_space(labels([1,2,3]))
    ham = matrix_representation(hamiltonian(H12, f), H12)

    ham1 = partial_trace(ham, H12 => H1)
    ρ1 = ground_state(ham1)
    ham2 = partial_trace(ham, H12 => H2)
    ρ2 = ground_state(ham2)
    
    op = matrix_representation(nbr_op(1,f), H12)
    
    ρ12 = tensor_product((ρ1, ρ2), (H1, H2)=> H12)
    exp_value = expectation_value(ρ12, op)

    op_eff = effective_meaurement(op, ρ2, H1, H2, H12)
    exp_value_eff = expectation_value(ρ1, op_eff)
    @test exp_value ≈ exp_value_eff
end
