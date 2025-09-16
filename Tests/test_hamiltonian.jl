
@testitem "Hamlitonian Hermiticity of fully conneted system" begin
    @fermions f
    # Single fully connected system 
    H1 = hilbert_space(labels([1,2,3]), NumberConservation())
    ham1 = matrix_representation(hamiltonian(hamiltonian_so_b, H1, f), H1)
    @test ham1 ≈ ham1'

    H2 = hilbert_space(labels([2, 5, 3]), NumberConservation())
    ham2 = matrix_representation(hamiltonian(hamiltonian_so_b, H2, f), H2)
    @test ham2 ≈ ham2'

    H3 = hilbert_space(labels([1,2,3]), NumberConservation(2))
    ham3 = matrix_representation(hamiltonian(hamiltonian_so_b, H3, f), H3)
    @test ham3 ≈ ham3'

    H4 = hilbert_space(labels([1,2,3]))
    ham4 = matrix_representation(hamiltonian(hamiltonian_simple, H4,f), H4)
    @test ham4 ≈ ham4
end

@testitem "Hamiltonian of two connected subsystem" begin
    @fermions f
    H_main = hilbert_space(labels([1,2]), NumberConservation())
    H_res = hilbert_space(labels([3, 4]), NumberConservation())
    H_total = tensor_product(H_main, H_res) 
    hams = hamiltonians(hamiltonian_so_b, H_main, H_res, f)
    
    ham_main = matrix_representation(hams.hamiltonian_main, H_main)
    ham_res = matrix_representation(hams.hamiltonian_reservoir, H_res)
    ham_interactions = matrix_representation(hams.hamiltonian_interaction, H_total)
    ham_total = matrix_representation(hams.hamiltonian_total, H_total)

    @test ham_main ≈ ham_main'
    @test ham_res ≈ ham_res'
    @test ham_interactions ≈ ham_interactions'
    @test ham_total ≈ ham_total'

    I_main = Matrix(I, dim(H_main), dim(H_main))
    I_res = Matrix(I, dim(H_res), dim(H_res))
    @test ham_total ≈ tensor_product((ham_main, I_main), (H_main, H_res) => H_total) +
                    tensor_product((I_main, ham_res), (H_main, H_res) => H_total) +
                    ham_interactions
end
#Add some more testing to actually test the hamiltonian structure, not just Hermiticity
