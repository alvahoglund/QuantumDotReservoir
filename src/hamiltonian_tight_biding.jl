struct DotParams
    ϵ::Vector{Float64}
    ϵb::Matrix{Float64}
    u_intra::Vector{Float64}
end

struct InteractionParams
    t::Matrix{ComplexF64}
    tso::Matrix{ComplexF64}
    u_inter::Matrix{Float64}
end

struct Hamiltonians
    hamiltonian_main
    hamiltonian_reservoir
    hamiltonian_interaction
    hamiltonian_total
end

## ============ Singel dot ================
hamiltonian_ϵ(ϵ, coordinate_labels, f) = sum(
    ϵ[label]*f[label, σ]' * f[label, σ] 
    for σ ∈ [:↑, :↓], (i,label) ∈ enumerate(coordinate_labels))

hamiltonian_b(ϵb, coordinate_labels, f) = sum(
    ϵb[label]*(-1)^(n+1)*f[label,σ]'f[label,σ]
    for (n,σ) ∈ enumerate([:↑, :↓]), (i,label) ∈ enumerate(coordinate_labels))

hamiltonian_c_intra(u_intra, coordinate_labels, f) = sum(
    u_intra[i]*f[label,:↑]'f[label, :↓]'f[label, :↓]f[label, :↑]
    for (i, label) ∈ enumerate(coordinate_labels))

## ============ Interactions ================

hamiltonian_c_inter(u_inter, coordinate_labels, f) = hamiltonian_c_inter_x(u_inter, coordinate_labels, f)+ hamiltonian_c_inter_y(u_inter, coordinate_labels, f)

hamiltonian_c_inter_x(u_inter, coordinate_labels, f) = sum(
    u_inter[(i,j), (i+1,j)]*f[(i,j), σ1]'f[(i+1,j),σ2]'f[(i+1,j),σ2]f[(i,j),σ1]
    for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓], (i,j) ∈ coordinate_labels if (i+1, j) ∈ coordinate_labels
)
hamiltonian_c_inter_y(u_inter, coordinate_labels, f) = sum(
    u_inter[(i,j), (i,j+1)]*f[(i,j), σ1]'f[(i,j+1),σ2]'f[(i,j+1),σ2]f[(i,j),σ1]
    for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓], (i,j) ∈ coordinate_labels if (i, j+1) ∈ coordinate_labels
)

hamiltonian_t(t, coordinate_labels, f) = hamiltonian_t_x(t, coordinate_labels, f) + hamiltonian_t_y(t, coordinate_labels, f)

hamiltonian_t_x(t, coordinate_labels, f) = sum(
    t[(i,j), (i+1, j)]f[(i+1,j), σ]'f[(i, j),σ] + hc
    for σ ∈ [:↑, :↓], (i,j) ∈ coordinate_labels if (i+1, j) ∈ coordinate_labels
)
hamiltonian_t_y(t, coordinate_labels, f) = sum(
    t[(i,j), (i, j+1)]f[(i,j+1), σ]'f[(i, j),σ] + hc
    for σ ∈ [:↑, :↓], (i,j) ∈ coordinate_labels if (i, j+1) ∈ coordinate_labels
)

hamiltonian_so(t_so, coordinate_labels, f) = hamiltonian_so_x(t_so, coordinate_labels, f) + hamiltonian_so_y(t_so, coordinate_labels, f)

hamiltonian_so_x(t_so, coordinate_labels, f) = sum(
    t_so[(i,j), (i+1, j)]*(-f[(i+1,j), ↑]'f[(i,j),↓] + f[(i+1,j), ↑]'f[(i,j),↓]) + hc
    for (i,j) ∈ coordinate_labels if (i+1, j) ∈ coordinate_labels
)
hamiltonian_so_y(t_so, coordinate_labels, f) = sum(
    t_so[(i,j), (i, j+j)]*(im*f[(i,j+1), ↑]'f[(i,j),↓] + im*f[(i,j+j), ↑]'f[(i,j),↓]) + hc
    for (i,j) ∈ coordinate_labels if (i, j+1) ∈ coordinate_labels
)

## ========= Set Dot Parameters =============
function main_system_dot_param(coordinates)
    Random.seed!(1)
    ϵ = Dict(coordinate => 1 for coordinate in coordinates)
    ϵb = Dict(coordinate => 1 for coordinate in coordinates)
    u_intra = Dict(coordinate => (rand()+1)*10 for coordinate in coordinates)
    return DotParams(ϵ, ϵb, u_intra)
end
function randomize_dot_param(coordinates)
    Random.seed!(1)
    ϵ = Dict(coordinate => rand() for coordinate in coordinates)
    ϵb = Dict(coordinate => 1 for coordinate in coordinates)
    u_intra = Dict(coordinate => (rand()+1)*10 for coordinate in coordinates)
    return DotParams(ϵ, ϵb, u_intra)
end
function randomize_interaction_param(coordinates)
    coupled_coordinates = get_coupled_coordinates(coordinates)
    t = (coupled_coordinate => rand() for coupled_coordinate in coupled_coordinates)
    t_so = (coupled_coordinate => rand() for coupled_coordinate in coupled_coordinates)
    u_inter = (coupled_coordinate => rand() for coupled_coordinate in coupled_coordinates)
    InteractionParams(t,t_so,u_inter)
end

function get_coupled_coordinates(coordinates)
    coupled_coordinates_x = [((i,j),(i+1,j)) for (i,j) in coordinates]
    coupled_coordinates_y = [((i,j),(i,j+1)) for (i,j) in coordinates]
    return vcat(coupled_coordinates_x, coupled_coordinates_y)
end

## ========= System Hamiltonians =============

hamiltonian_dots(dot_params, coordinates, f) = 
    hamiltonian_ϵ(dot_params.ϵ, coordinates, f) +  hamiltonian_b(dot_params.ϵb, coordinates, f) + hamiltonian_c_intra(dot_params.u_intra, coordinates, f)
    
hamiltonian_interactions(interaction_params, coordinates, f) = 
    hamiltonian_t(interaction_params.t, coordinates, f) + hamiltonian_so(interaction_params.t_so, coordinates, f) + hamiltonian_c_inter(interaction_params.u_inter, coordinates, f) 

hamiltonian_intersection(interaction_params, coordinates, f) = 
    hamiltonian_t_x(interaction_params.t, coordinates, f) + hamiltonian_so_x(interaction_params.t_so, coordinates, f) + hamiltonian_c_inter_x(interaction_params.u_inter, coordinates, f) 

function hamiltonian(quantum_dot_system, f)
    dot_params = merge(main_system_dot_param(quantum_dot_system.coordinates_main), randomize_dot_param(quantum_dot_system.sites_reservoir))
    interaction_params = randomize_interaction_param(quantum_dot_system.sites_total)

    hamiltonian_main = hamiltonian_dots(dot_params, quantum_dot_system.sites_main, f) + hamiltonian_interactions(interaction_params, quantum_dot_system.sites_main, f)
    hamiltonian_reservoir = hamiltonian_dots(dot_params, quantum_dot_system.sites_reservoir, f) + hamiltonian_interactions(interaction_params, quantum_dot_system.sites_reservoir, f)
    hamiltonian_intersection = hamiltonian_interactions(interaction_params, quantum_dot_system.sites_intersection, f)
    hamiltonian_total = hamiltonian_main + hamiltonian_reservoir + hamiltonian_intersection 
    return hamiltonian_main, hamiltonian_reservoir, hamiltonian_intersection, hamiltonian_total
end