singlet(f) = 1/ √2 * (f[(1,1), :↑]' * f[(2,1), :↓]' - f[(1,1), :↓]' * f[(2,1), :↑]')
triplet_0(f) = 1/ √2 * (f[(1,1), :↑]' * f[(2,1), :↓]' + f[(1,1), :↓]' * f[(2,1), :↑]')
triplet_plus(f) = 1/√2  *((f[(1,1), :↑]' * f[(2,1), :↑]' + f[(1,1), :↓]' * f[(2,1), :↓]'))
triplet_minus(f) = 1/√2 *((f[(1,1), :↑]' * f[(2,1), :↑]' - f[(1,1), :↓]' * f[(2,1), :↓]'))


function vac_state(H)
    v0 = spzeros(dim(H))
    v0[FermionicHilbertSpaces.state_index(FockNumber(0), H)] = 1.0
    return v0
end

function def_state(f, state_name, H)
    v0 = vac_state(H)
    v = matrix_representation(state_name(f), H) * v0
    ρ = v*v'
    ρ = ρ / norm(ρ)
    return ρ
end

function ground_state(hamiltonian :: AbstractMatrix)
    eigenvalues, eigenvectors = eigen(Matrix(hamiltonian))
    ρ = eigenvectors[:, 1] * eigenvectors[:, 1]'
    return ρ
end

function ground_state(ham :: AbstractMatrix{T}, H ::FermionicHilbertSpaces.AbstractHilbertSpace, qn :: Int) where T
    index_qn = FermionicHilbertSpaces.FermionicHilbertSpaces.indices(qn, H)
    ham_qn = ham[index_qn, index_qn]
    state = spzeros(T, dim(H), dim(H))
    state[index_qn, index_qn] = ground_state(ham_qn)
    return state
end

ground_state(ham:: NonCommutativeProducts.NCAdd, H::FermionicHilbertSpaces.AbstractHilbertSpace, qn :: Int) = 
    ground_state(matrix_representation(ham, H), H, qn)
