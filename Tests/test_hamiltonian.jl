
@testitem "Hamiltonian of tight binding model" begin
    qd_system = tight_binding_system(2,3,1)
    hams = hamiltonians(qd_system)
    h_main = matrix_representation(hams.hamiltonian_main, qd_system.H_total)
    h_res = matrix_representation(hams.hamiltonian_reservoir, qd_system.H_total)
    h_int = matrix_representation(hams.hamiltonian_interaction, qd_system.H_total)
    h_tot = matrix_representation(hams.hamiltonian_total, qd_system.H_total)
    @test  h_main' ≈ h_main
    @test  h_res' ≈ h_res
    @test  h_int' ≈ h_int
    @test  h_tot' ≈ h_tot
end