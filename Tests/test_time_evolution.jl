
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
    H_main = hilbert_space(labels([1,2]))
    H_res = hilbert_space(labels([2,3]))
    H_tot = hilbert_space(labels([1,2,3]))
    hams = hamiltonians(hamiltonian_so_b, H_main, H_res, f)

    ham1 = matrix_representation(hams.hamiltonian_main, H_main)
    ρ1 = ground_state(ham1)
    ham2 = matrix_representation(hams.hamiltonian_reservoir, H_res)
    ρ2 = ground_state(ham2)
    
    op = matrix_representation(nbr_op(1,f), H_tot)
    
    ρ12 = tensor_product((ρ1, ρ2), (H_main, H_res)=> H_tot)
    exp_value = expectation_value(ρ12, op)

    op_eff = effective_measurement(op, ρ2, H_main, H_res, H_tot)
    exp_value_eff = expectation_value(ρ1, op_eff)
    @test exp_value ≈ exp_value_eff
end
