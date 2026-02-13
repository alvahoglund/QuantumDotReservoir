#### ===== FUNCTIONS ================
function random_pure_states(dim)
    Ψ = (randn(dim) .+ 1im*randn(dim))'
    ρ = Ψ'Ψ /tr(Ψ'Ψ)
    return ρ
end

qd_system = tight_binding_system(2,2,1)
get_states(nbr_states) = hcat([reshape(hilbert_schmidt_ensamble(4), 16,1) for i in 1:nbr_states]...)
get_states_list(nbr_states) = [hilbert_schmidt_ensamble(4) for i in 1:nbr_states]

clean_val(y) = map(x -> abs(x) < 1e-2 ? 0.0 : x, y)
ind = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)

i_op_vec(qd_system) = reshape(pauli_string(σ0, σ0, qd_system)[ind,ind]', 16,1)
i_op(qd_system) = pauli_string(σ0, σ0, qd_system)[ind,ind]'

flip_op(qd_system) = (1/2).*(pauli_string(σ0, σ0, qd_system)[ind,ind] +pauli_string(σx, σx, qd_system)[ind,ind]+pauli_string(σy, σy, qd_system)[ind,ind]+pauli_string(σz, σz, qd_system)[ind,ind])
flip_op_vec(qd_system) = reshape(flip_op_vec(qd_system)', 16, 1)

## ======= Constants ===========
N = 4
purity = 0.47
a = (N-purity)/(N*(N^2-1))
b = (N*purity-1)/(N*(N^2-1))

## ======= E[ρ ⊗ ρ] ============
nbr_states = 10^4
state_list = get_states_list(nbr_states)
E_hs = clean_val.(sum([Matrix(state ⊗ state) for state in state_list])./nbr_states)

ps_2qb = [(1/2).*Matrix(ps[ind,ind]) for ps in pauli_strings(qd_system)]
F_o = clean_val(sum([Matrix(ps2 ⊗ ps2) for ps2 in ps_2qb]))
I_o = Matrix(I, 16,16)

Eρρ = a*I_o+b*F_o
sum(E_hs - Eρρ)

### ====== E[RR'] ================
nbr_states = 10^5
R_m = get_states(nbr_states)
clean_val.(real.(R_m*R_m'.*1/nbr_states))

ps_2qb_vec = [(1/2).*reshape(Matrix(ps[ind,ind]), 16,1)  for ps in pauli_strings(qd_system)]
F_temp = clean_val(real.(sum([ps2*ps2' for ps2 in ps_2qb_vec])))
I_temp = reshape(Matrix(I, 4,4), 16,1)*reshape(Matrix(I, 4,4), 16,1)'
Eρρ_vec = a*I_temp+b*F_temp
sum(Eρρ_vec-smm)

## ======= Pauli basis ===================
R_m = get_states(nbr_states)
pauli_mat = pauli_string_matrix(qd_system)
R_m_ps = (1/2).*pauli_mat*R_m
smm_ps = R_m_ps*R_m_ps'.*1/nbr_states
clean_val.(real.(smm_ps))

F_p = (1/4).*pauli_mat*F_temp*pauli_mat'
I_p = (1/4).*pauli_mat*I_temp*pauli_mat'

Eρρ_p = a*I_p + b*F_p

sum(Eρρ_p-smm_ps)