abstract type AbstractPropagatorAlg end
struct BlockPropagatorAlg <: AbstractPropagatorAlg end

function scrambling_map(qd_system, measurements :: AbstractArray{<:AbstractMatrix}, ρ_reservoir_tot ::ExtendedResState, 
    hamiltonian_total_qn :: AbstractMatrix, t, ::BlockPropagatorAlg)
    prop = propagator(t, hamiltonian_total_qn)
    process_measurements = op -> effective_measurement(
        operator_time_evolution(prop, op), ρ_reservoir_tot, qd_system
    )
    eff_measurements = map(process_measurements, measurements)
    scrambling_map = vcat([vec(m)' for m in eff_measurements]...)
    return scrambling_map
end

function scrambling_map(qd_system, measurements  :: AbstractArray{<:FermionicHilbertSpaces.NonCommutativeProducts.NCAdd}, Ψ_res, 
    hamiltonian_total :: FermionicHilbertSpaces.NonCommutativeProducts.NCAdd, t_list, alg::BlockPropagatorAlg)
    ρ_res = if Ψ_res isa AbstractVector
        Ψ_res * Ψ_res'
    else
        Ψ_res
    end
    if isa(t_list, Number)
        t_list = [t_list]
    end
    measurements_m = map(m -> Diagonal(diag(matrix_representation(m ,qd_system.H_total_qn))), measurements)
    hamiltonian_total_qn = matrix_representation(hamiltonian_total, qd_system.H_total_qn)

    ρ_reservoir_tot = extend_res_state(ρ_res, qd_system)

    scrambling_maps = [scrambling_map(qd_system, measurements_m, ρ_reservoir_tot, hamiltonian_total_qn, t, alg) for t in t_list]
    return vcat(scrambling_maps...)
end

struct PureStatePropagatorAlg <: AbstractPropagatorAlg
    krylov_dim::Int
    tol::Float64
end
PureStatePropagatorAlg(; krylov_dim=200, tol=1e-6) = PureStatePropagatorAlg(krylov_dim, tol)

function scrambling_map(qd_system, measurements:: AbstractArray{<:FermionicHilbertSpaces.NonCommutativeProducts.NCAdd}, Ψ_res :: AbstractVector, 
    hamiltonian_total :: FermionicHilbertSpaces.NonCommutativeProducts.NCAdd, t_list, alg::PureStatePropagatorAlg)
    if isa(t_list, Number)
        t_list = [t_list]
    end
    measurements_m = map(m -> Diagonal(diag(matrix_representation(m ,qd_system.H_total_qn))), measurements)
    hamiltonian_total_qn = matrix_representation(hamiltonian_total, qd_system.H_total_qn)
    scrambling_maps = [scrambling_map(qd_system, measurements_m, Ψ_res, hamiltonian_total_qn, t, alg) for t in t_list]
    return vcat(scrambling_maps...)
end

function scrambling_map(sys::QuantumDotSystem, measurements :: AbstractArray{<:AbstractMatrix}, ψres::AbstractVector, hamiltonian ::AbstractMatrix, t, alg::PureStatePropagatorAlg)
    iH = -im .* hamiltonian
    N = dim(sys.H_total_qn) 
    N_main = dim(sys.H_main_qn)
    Ks = KrylovSubspace{ComplexF64}(N, alg.krylov_dim)
    e_j = zeros(ComplexF64, N_main)
    U = zeros(ComplexF64, N, N_main)
    U = stack(1:N_main) do n
        fill!(e_j, 0)
        e_j[n] = 1.0
        ψtot = generalized_kron((e_j, ψres), (sys.H_main_qn, sys.H_reservoir_qn) => sys.H_total_qn)
        arnoldi!(Ks, iH, ψtot; tol=alg.tol)
        expv(t, Ks)
    end
    stack(op -> vec(U' * Diagonal(op) * U), measurements)'
end

