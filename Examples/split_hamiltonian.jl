nbr_dots_main = 2
nbr_dots_res = 1
qn_reservoir = 2
qd_system = quantum_dot_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
hams = hamiltonians(hamiltonian_so_b, H_main, H_reservoir, f)

ham_main = matrix_representation(hams.hamiltonian_main, qd_system.H_main)
ham_res = matrix_representation(hams.hamiltonian_reservoir, qd_system.H_reservoir)
ham_interactions = matrix_representation(hams.hamiltonian_interaction, qd_system.H_total)
ham_tot = matrix_representation(hams.hamiltonian_total, qd_system.H_total)

# H_tot = hamiltonian_main ⊗ I + I⊗ hamiltonian_reservoir + hamiltonian_interactions
I_main = Matrix(I, dim(qd_system.H_main), dim(qd_system.H_main))
I_res = Matrix(I, dim(qd_system.H_reservoir), dim(qd_system.H_reservoir))
ham_tot ≈ tensor_product((ham_main, I_res), (qd_system.H_main, qd_system.H_reservoir) => qd_system. H_total) +
        tensor_product((I_main, ham_res), (qd_system.H_main, qd_system.H_reservoir) => qd_system. H_total) + 
        ham_interactions

open("output.txt", "w") do io
     Base.show(io, hams.hamiltonian_reservoir; max_terms = 999)
end
