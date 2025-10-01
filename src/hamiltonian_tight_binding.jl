struct DotParams
    ϵ
    ϵb
    u_intra
end

struct InteractionParams
    t
    t_so
    u_inter
end

struct Hamiltonians
    hamiltonian_main
    hamiltonian_reservoir
    hamiltonian_intersection
    hamiltonian_total
    dot_params_main
    dot_params_reservoir
    interaction_params
end

## ============ Singel dot ================

hamiltonian_ϵ(ϵ, coordinate_labels, f) = sum(
    ϵ[label]*f[label, σ]' * f[label, σ] 
    for σ ∈ [:↑, :↓], label ∈ coordinate_labels;
    init=0
)

hamiltonian_b(ϵb, coordinate_labels, f) = sum(
    ϵb[label]*(-1)^(n+1)*f[label,σ]'f[label,σ]
    for (n,σ) ∈ enumerate([:↑, :↓]), label ∈ coordinate_labels;
    init=0
)

hamiltonian_c_intra(u_intra, coordinate_labels, f) = sum(
    u_intra[label]*f[label,:↑]'f[label, :↓]'f[label, :↓]f[label, :↑]
    for label ∈ coordinate_labels;
    init=0
)

## ============ Interactions ================

hamiltonian_c_inter(u_inter, coordinate_labels, f) = hamiltonian_c_inter_x(u_inter, coordinate_labels, f)+ hamiltonian_c_inter_y(u_inter, coordinate_labels, f)

hamiltonian_c_inter_x(u_inter, coordinate_labels, f) = sum(
    u_inter[(i,j), (i+1,j)]*f[(i,j), σ1]'f[(i+1,j),σ2]'f[(i+1,j),σ2]f[(i,j),σ1]
    for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓], (i,j) ∈ coordinate_labels if (i+1, j) ∈ coordinate_labels;
    init=0
)
hamiltonian_c_inter_y(u_inter, coordinate_labels, f) = sum(
    u_inter[(i,j), (i,j+1)]*f[(i,j), σ1]'f[(i,j+1),σ2]'f[(i,j+1),σ2]f[(i,j),σ1]
    for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓], (i,j) ∈ coordinate_labels if (i, j+1) ∈ coordinate_labels;
    init=0
)

hamiltonian_t(t, coordinate_labels, f) = hamiltonian_t_x(t, coordinate_labels, f) + hamiltonian_t_y(t, coordinate_labels, f)

hamiltonian_t_x(t, coordinate_labels, f) = sum(
    t[(i,j), (i+1, j)]f[(i+1,j), σ]'f[(i, j),σ] + hc
    for σ ∈ [:↑, :↓], (i,j) ∈ coordinate_labels if (i+1, j) ∈ coordinate_labels;
        init=0
)
hamiltonian_t_y(t, coordinate_labels, f) = sum(
    t[(i,j), (i, j+1)]f[(i,j+1), σ]'f[(i, j),σ] + hc
    for σ ∈ [:↑, :↓], (i,j) ∈ coordinate_labels if (i, j+1) ∈ coordinate_labels;
    init=0
)

hamiltonian_so(t_so, coordinate_labels, f) = hamiltonian_so_x(t_so, coordinate_labels, f) + hamiltonian_so_y(t_so, coordinate_labels, f)

hamiltonian_so_x(t_so, coordinate_labels, f) = sum(
    t_so[(i,j), (i+1, j)]*(-f[(i+1,j), :↑]'f[(i,j),:↓] + f[(i+1,j), :↓]'f[(i,j),:↑]) + hc
    for (i,j) ∈ coordinate_labels if (i+1, j) ∈ coordinate_labels;
    init=0
)
hamiltonian_so_y(t_so, coordinate_labels, f) = sum(
    t_so[(i,j), (i, j+1)]*(im*f[(i,j+1), :↑]'f[(i,j),:↓] + im*f[(i,j+1), :↓]'f[(i,j),:↑]) + hc
    for (i,j) ∈ coordinate_labels if (i, j+1) ∈ coordinate_labels;
    init=0
)

## ========= Set Dot Parameters =============
function main_system_dot_param(coordinates)
    ϵ = Dict(coordinate => 0.5 for coordinate in coordinates)
    ϵb = Dict(coordinate => 1 for coordinate in coordinates)
    u_intra = Dict(coordinate => (rand()+10) for coordinate in coordinates)
    return DotParams(ϵ, ϵb, u_intra)
end

function randomize_dot_param(coordinates)
    ϵ = Dict(coordinate => rand() for coordinate in coordinates)
    ϵb = Dict(coordinate => 1 for coordinate in coordinates)
    u_intra = Dict(coordinate => (rand()+10) for coordinate in coordinates)
    return DotParams(ϵ, ϵb, u_intra)
end

function randomize_interaction_param(coordinates)
    coupled_coordinates = get_coupled_coordinates(coordinates)
    t = Dict(coupled_coordinate => rand() for coupled_coordinate in coupled_coordinates)
    t_so = Dict(coupled_coordinate => rand()*0.1 for coupled_coordinate in coupled_coordinates)
    u_inter = Dict(coupled_coordinate => rand() for coupled_coordinate in coupled_coordinates)
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

hamiltonian_interactions_x(interaction_params, coordinates, f) = 
    hamiltonian_t_x(interaction_params.t, coordinates, f) + hamiltonian_so_x(interaction_params.t_so, coordinates, f) + hamiltonian_c_inter_x(interaction_params.u_inter, coordinates, f) 

function hamiltonians(quantum_dot_system, seed)
    Random.seed!(seed)
    return hamiltonians(quantum_dot_system)
end 
function hamiltonians(quantum_dot_system)
    dot_params_main = main_system_dot_param(quantum_dot_system.coordinates_main)
    dot_params_reservoir = randomize_dot_param(quantum_dot_system.coordinates_reservoir)
    interaction_params = randomize_interaction_param(quantum_dot_system.coordinates_total)

    hamiltonian_main = hamiltonian_dots(dot_params_main, quantum_dot_system.coordinates_main, quantum_dot_system.f) + hamiltonian_interactions(interaction_params, quantum_dot_system.coordinates_main, quantum_dot_system.f)
    hamiltonian_reservoir = hamiltonian_dots(dot_params_reservoir, quantum_dot_system.coordinates_reservoir, quantum_dot_system.f) + hamiltonian_interactions(interaction_params, quantum_dot_system.coordinates_reservoir, quantum_dot_system.f)
    hamiltonian_intersection = hamiltonian_interactions_x(interaction_params, quantum_dot_system.coordinates_intersection, quantum_dot_system.f)
    hamiltonian_total = hamiltonian_main + hamiltonian_reservoir + hamiltonian_intersection 
    return Hamiltonians(hamiltonian_main, hamiltonian_reservoir, hamiltonian_intersection, hamiltonian_total, 
                        dot_params_main, dot_params_reservoir, interaction_params)
end