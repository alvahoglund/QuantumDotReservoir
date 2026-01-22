## =================== Vaccum state ================ 
function vac_state(H)
    v0 = spzeros(dim(H))
    v0[FermionicHilbertSpaces.state_index(FockNumber(0), H)] = 1.0
    return v0
end

## =================== 2 dot main system states =============
singlet(f) = 1/ √2 * (f[(1,1), :↑]' * f[(1,2), :↓]' - f[(1,1), :↓]' * f[(1,2), :↑]')
triplet_0(f) = 1/ √2 * (f[(1,1), :↑]' * f[(1,2), :↓]' + f[(1,1), :↓]' * f[(1,2), :↑]')
triplet_plus(f) = 1/√2  *((f[(1,1), :↑]' * f[(1,2), :↑]' + f[(1,1), :↓]' * f[(1,2), :↓]'))
triplet_minus(f) = 1/√2 *((f[(1,1), :↑]' * f[(1,2), :↑]' - f[(1,1), :↓]' * f[(1,2), :↓]'))

function def_state(state_name, H, f)
    v0 = vac_state(H)
    v = matrix_representation(state_name(f), H) * v0
    ρ = v*v'
    ρ = ρ / norm(ρ)
    return ρ
end

function max_mixed_state(H,f)
    v0 = vac_state(H)
    states = [matrix_representation(f[(1,1), σ1]'f[(1,2), σ2]', H)*v0 for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓]]
    ρ_mixed = 1/2*sum(state*state' for state in states)
    return ρ_mixed
end

werner_state(state_name, p, H,f) = (1-p)*def_state(state_name, H, f) + p*max_mixed_state(H,f)

function random_qubit_state(coordinate, f) 
    θ = acos(2 * rand() - 1) 
    ϕ = rand()*π*2
    return cos(θ/2)*f[coordinate, :↑]'+exp(im*ϕ)*sin(θ/2)*f[coordinate, :↓]'
end

function random_product_state(coordinate_a, coordinate_b, Ha, Hb, Hab, f)
    va = matrix_representation(random_qubit_state(coordinate_a, f), Ha)*vac_state(Ha)
    ρa = va*va'
    vb = matrix_representation(random_qubit_state(coordinate_b, f), Hb)*vac_state(Hb)
    ρb = vb*vb'
    return tensor_product((ρa, ρb), (Ha, Hb) => Hab)
end

random_product_state(qd_system) = 
    random_product_state(qd_system.coordinates_main[1], qd_system.coordinates_main[2], qd_system.H_main_a, qd_system.H_main_b, qd_system.H_main, qd_system.f)

function random_separable_state(nbr_states, qd_system)
    p = rand(nbr_states)
    p = p ./ sum(p)
    ρ_sep = sum(p[i]*random_product_state(qd_system) for i ∈ 1:nbr_states)
    return ρ_sep
end

function hilbert_schmidt_ensamble(dim)
    X = (randn(dim, dim) .+ 1im * randn(dim, dim))./sqrt(2)
    ρ = X'X/tr(X'X)
    return ρ
end
## =============== Ground States ====================
function eig_state(hamiltonian :: AbstractMatrix, n)
    eigenvalues, eigenvectors = eigen(Matrix(hamiltonian))
    ρ = eigenvectors[:, n] * eigenvectors[:, n]'
    return ρ
end

ground_state(ham :: AbstractMatrix{T}, H ::FermionicHilbertSpaces.AbstractHilbertSpace, qn :: Int) where T= eig_state(ham, H, qn, 1)

function eig_state(ham :: AbstractMatrix{T}, H ::FermionicHilbertSpaces.AbstractHilbertSpace, qn :: Int, n) where T
    index_qn = FermionicHilbertSpaces.FermionicHilbertSpaces.indices(qn, H)
    ham_qn = ham[index_qn, index_qn]
    state = spzeros(T, dim(H), dim(H))
    state[index_qn, index_qn] = eig_state(ham_qn, n)
    return state
end

eig_state(ham:: NonCommutativeProducts.NCAdd, H::FermionicHilbertSpaces.AbstractHilbertSpace, qn :: Int, n) = 
    eig_state(matrix_representation(ham, H), H, qn, n)
ground_state(ham:: NonCommutativeProducts.NCAdd, H::FermionicHilbertSpaces.AbstractHilbertSpace, qn :: Int) = 
    ground_state(matrix_representation(ham, H), H, qn)

