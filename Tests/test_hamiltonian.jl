
@testitem "Hamlitonian Hermiticity" begin
    @fermions f

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

#Add some more testing to actually test the hamiltonian structure, not just Hermiticity
