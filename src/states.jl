@fermions f
singlet = 1/ √2 * (f[1, :↑]' * f[2, :↓]' - f[1, :↓]' * f[2, :↑]')
triplet_0 = 1/ √2 * (f[1, :↑]' * f[2, :↓]' + f[1, :↓]' * f[2, :↑]')
triplet_plus = 1/√2  *((f[1, :↑]' * f[2, :↑]' + f[1, :↓]' * f[2, :↓]'))
triplet_minus = 1/√2 *((f[1, :↑]' * f[2, :↑]' - f[1, :↓]' * f[2, :↓]'))


function vac_state(H)
    v0 = zeros(dim(H))
    v0[FermionicHilbertSpaces.state_index(FockNumber(0), H)] = 1.0
    return v0
end

function def_state(state_name, H)
    v0 = vac_state(H)
    v = matrix_representation(state_name, H) * v0
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
    index_qn = sector_index(qn, H)
    ham_qn = ham[index_qn, index_qn]
    state = zeros(T, dim(H), dim(H))
    state[index_qn, index_qn] = ground_state(ham_qn)
    return state
end