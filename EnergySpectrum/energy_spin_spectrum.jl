function temp_dot_param(coordinates, ϵ_val, ϵb_val, u_intra_val)
    ϵ = Dict(coordinate => ϵ_val for coordinate in coordinates)
    ϵb = Dict(coordinate => ϵb_val for coordinate in coordinates)
    u_intra = Dict(coordinate => u_intra_val for coordinate in coordinates)
    return DotParams(ϵ, ϵb, u_intra)
end

function temp_interaction_param(coordinates, t_val, t_so_val, u_inter_val)
    coupled_coordinates = get_coupled_coordinates(coordinates)
    t = Dict(coupled_coordinate => t_val for coupled_coordinate in coupled_coordinates)
    t_so = Dict(coupled_coordinate => t_so_val for coupled_coordinate in coupled_coordinates)
    u_inter = Dict(coupled_coordinate => u_inter_val for coupled_coordinate in coupled_coordinates)
    InteractionParams(t,t_so,u_inter)
end

function get_eigenbasis(coordinates, qn, t_val, U_val)
    ϵ_val = 0
    ϵb_val = 0
    u_intra_val = U_val

    t_val = t_val
    t_so_val = 0
    u_inter_val = 0

    dp = temp_dot_param(coordinates, ϵ_val, ϵb_val, u_intra_val)
    ip = temp_interaction_param(coordinates,t_val,  t_so_val, u_inter_val)

    @fermions f

    H = hilbert_space(labels(coordinates), NumberConservation(qn))

    sys_ham_dot = hamiltonian_dots(dp, coordinates,f) 
    sys_ham_int = hamiltonian_interactions(ip, coordinates, f)
    sys_ham = sys_ham_dot + sys_ham_int
    return eigen(Matrix(matrix_representation(sys_ham , H)))
end

Si2(coordinate_i, f, H_i) = matrix_representation(3/4*p1(coordinate_i, f), H_i)

function Sij(coordinate_i, coordinate_j, f, H)
    Hi = hilbert_space(labels([coordinate_i]), NumberConservation())
    Hj = hilbert_space(labels([coordinate_j]), NumberConservation())
    Hij = hilbert_space(labels([coordinate_i, coordinate_j]), NumberConservation())

    σxi = matrix_representation(σx(coordinate_i, f), Hi)
    σxj = matrix_representation(σx(coordinate_j, f), Hj)
    σxij = tensor_product((σxi, σxj), (Hi, Hj) => Hij)

    σyi = matrix_representation(σy(coordinate_i, f), Hi)
    σyj = matrix_representation(σy(coordinate_j, f), Hj)
    σyij = tensor_product((σyi, σyj), (Hi, Hj) => Hij)

    σzi = matrix_representation(σz(coordinate_i, f), Hi)
    σzj = matrix_representation(σz(coordinate_j, f), Hj)
    σzij = tensor_product((σzi, σzj), (Hi, Hj) => Hij)

    S_ij = 1/4*(σxij + σyij + σzij)

    return embed(S_ij, Hij=> H)
end

function total_spin_op(coordinates, f)
    H_total = hilbert_space(labels(coordinates), NumberConservation())
    S2_op = sum([Si2(coordinate, f, H_total) for coordinate in coordinates])
    Sij_op = sum([Sij(coordinate_i, coordinate_j, f, H_total) 
                for (i, coordinate_i) in enumerate(coordinates)
                for (j, coordinate_j) in enumerate(coordinates)
                if i<j])
    return S2_op + 2*Sij_op
end
s_from_s2(s2_val) = -1/2 + √(s2_val+1/4)

function test_singlet_triplets()
    coordinates = [(1,1), (1,2)]
    s2_op = total_spin_op(coordinates, f)
    
    H_main = hilbert_space(labels(coordinates), NumberConservation())
    ρ_singlet = def_state(singlet, H_main, f)
    ρ_triplet_minus = def_state(triplet_minus, H_main, f)
    ρ_triplet = def_state(triplet_0, H_main, f)
    ρ_triplet_plus = def_state(triplet_plus, H_main, f)
    
    println("Total spin of S: $(s_from_s2(expectation_value(ρ_singlet, s2_op)))")
    println("Total spin of T+: $(s_from_s2(expectation_value(ρ_triplet_minus, s2_op)))")
    println("Total spin of T0: $(s_from_s2(expectation_value(ρ_triplet, s2_op)))")
    println("Total spin of T-: $(s_from_s2(expectation_value(ρ_triplet_plus, s2_op)))")
end


function plot_energy_spin_spectrum(coordinates, qn, t_val, U_val)
    vals, vecs = get_eigenbasis(coordinates, qn, t_val, U_val)
    vals = real(vals)
    idx = FermionicHilbertSpaces.indices(qn, hilbert_space(labels(coordinates), NumberConservation()))
    s2_op = total_spin_op(coordinates, f)

    s_vals = [s_from_s2(expectation_value(vecs[:,i]*vecs[:,i]', s2_op[idx, idx])) for i in 1:length(vals)]

    p_se = plot(xlim = (vals[1]-0.5, vals[end]+0.5), 
                ylim = (0, maximum(s_vals)+1),
                xlabel = "Energy",
                ylabel = "Total Spin", 
                title= "$(length(coordinates)) dots and $(qn) electrons")
    println(vals)
    scatter!(p_se, vals, s_vals)
    vline!(p_se, vals)
    display(p_se)
    return vals, vecs
end

differences(vals) = [abs(vals[j] - vals[i]) for i in eachindex(vals), j in eachindex(vals) if i<j]

function plot_energy_difference(vals, t_val, U_val)
    pd = plot(xlabel = "Energy level difference", xlim =(-1, 11), title= "$(length(coordinates)) dots and $(qn) electrons", size = (700, 400))
    vline!(pd, differences(vals), label = "Hubbard Hamiltonian Energy Differences", linewidth= 2, color = :black)
    vline!(pd, [2*t_val], label = "t", linewidth =3, linestyle = :dash)
    vline!(pd, [U_val], label = "U", linewidth =3, linestyle = :dash)
    #vline!(pd, [4*t_val^2/U_val], label = "4t^2/U",linewidth=3, linestyle = :dash)
    vline!(differences(eigen(Matrix(heisenberg_hamiltonian(coordinates, f, t_val, U_val))).values), label = "Heisenberg Hamiltonian Energy Difference", linewidth =3, linestyle = :dash)
    vline!(pd, )
    display(pd)
end

function heisenberg_hamiltonian(coordinates, f, t_val, U_val)
    J = 4*t_val^2/U_val
    H_total = hilbert_space(labels(coordinates), NumberConservation())
    Sij_op = sum([-J* Sij(coordinate_i, coordinate_j, f, H_total) 
                for (coordinate_i, coordinate_j) in get_coupled_coordinates(coordinates)
                if coordinate_i ∈ coordinates && coordinate_j ∈ coordinates])
    idx = FermionicHilbertSpaces.indices(length(coordinates), H_total)
    return Sij_op[idx, idx]
end


#test_singlet_triplets()

@fermions f
coordinates = [(1,1), (1,2), (2,1)]
qn = length(coordinates)
t_val = 1
U_val = 10
vals, vecs = plot_energy_spin_spectrum(coordinates, qn, t_val, U_val)
plot_energy_difference(vals, t_val, U_val)
