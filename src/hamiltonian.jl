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

# ============= Single dot ==================
hamiltonian_ϵ(ϵ, sites, f) = sum(
    ϵ[i]*f[label, σ]' * f[label, σ] 
    for σ ∈ [:↑, :↓], (i,label) ∈ enumerate(sites))

hamiltonian_b(ϵb,sites, f) = sum(
    ϵb[i]*(-1)^(n+1)*f[label,σ]'f[label,σ]
    for (n,σ) ∈ enumerate([:↑, :↓]), (i,label) ∈ enumerate(sites))

hamiltonian_c_intra(u_intra, sites, f) = sum(
    u_intra[i]*f[label,:↑]'f[label, :↓]'f[label, :↓]f[label, :↑]
    for (i, label) ∈ enumerate(sites))

# =============== Interactions - all connected =================
hamiltonian_t(t, sites, f) = length(sites) >1 ? sum(
    t[i,j]*f[label_i, σ]'f[label_j, σ] + hc
    for σ ∈ [:↑, :↓], (i,label_i) ∈ enumerate(sites), (j,label_j) ∈ enumerate(sites) if i < j) : 0

hamiltonian_so(tso, sites, f) = length(sites) >1 ? sum(
    tso[i,j]*f[label_i,:↑]'f[label_j,:↓] + tso[i,j]*f[label_i,:↓]'f[label_j,:↑] +hc
    for (i,label_i) ∈ enumerate(sites), (j, label_j) ∈ enumerate(sites) if i<j) : 0

hamiltonian_c_inter(u_inter, sites, f) = length(sites) >1 ? sum(
    u_inter[i,j]*f[label_i, σ1]'f[label_j, σ2]'f[label_j, σ2]f[label_i, σ1]
    for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓], (i, label_i) ∈ enumerate(sites), (j,label_j) ∈ enumerate(sites) if i < j) : 0

# =============  Interactions - connect two subsystems ================
hamiltonian_t(t, sites_i, sites_j, f) = sum(
    t[i,j]*f[label_i, σ]'f[label_j, σ] + hc
    for σ ∈ [:↑, :↓], (i,label_i) ∈ enumerate(sites_i), (j,label_j) ∈ enumerate(sites_j))
  
hamiltonian_so(tso, sites_i, sites_j, f) = sum(
    tso[i,j]*f[label_i,:↑]'f[label_j,:↓] + tso[i,j]*f[label_i,:↓]'f[label_j,:↑] +hc
    for (i,label_i) ∈ enumerate(sites_i), (j, label_j) ∈ enumerate(sites_j))

hamiltonian_c_inter(u_inter, sites_i, sites_j, f) = sum(
    u_inter[i,j]*f[label_i, σ1]'f[label_j, σ2]'f[label_j, σ2]f[label_i, σ1]
    for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓], (i, label_i) ∈ enumerate(sites_i), (j,label_j) ∈ enumerate(sites_j))


# =============== Randomizing Parameters ====================
function randomize_dot_param(nbr_sites)
    Random.seed!(3)
    ϵ = rand(nbr_sites)
    ϵb = rand(nbr_sites, nbr_sites)
    u_intra = rand(nbr_sites)
    return DotParams(ϵ, ϵb, u_intra)
end

randomize_interaction_param(nbr_sites) = randomize_interaction_param(nbr_sites, nbr_sites)
function randomize_interaction_param(nbr_sites_i, nbr_sites_j) 
    Random.seed!(1)
    t = rand(ComplexF64, nbr_sites_i, nbr_sites_j)
    tso = rand(ComplexF64, nbr_sites_i, nbr_sites_j)
    u_inter = rand(nbr_sites_i, nbr_sites_j) 
    return InteractionParams(t, tso, u_inter)
end

# =================== Hamiltonian with all dots connected =============
hamiltonian_simple(dot_param :: DotParams, interaction_param ::InteractionParams, sites, f) = hamiltonian_ϵ(dot_param.ϵ, sites, f) + hamiltonian_t(interaction_param.t, sites, f)

hamiltonian_so_b(dot_param :: DotParams, interaction_param :: InteractionParams, sites, f) = hamiltonian_ϵ(dot_param.ϵ, sites, f) + hamiltonian_b(dot_param.ϵb, sites, f) + hamiltonian_t(interaction_param.t, sites,f) +
    hamiltonian_so(interaction_param.tso, sites, f) + hamiltonian_c_intra(dot_param.u_intra, sites, f) +hamiltonian_c_inter(interaction_param.u_inter, sites, f) 

hamiltonian_temp(dot_param :: DotParams, interaction_param ::InteractionParams, sites, f) = hamiltonian_t(interaction_param.t, sites, f)

function hamiltonian(hamiltonian_type, H, f)
    site_labels = sites(H)
    nbr_sites = length(site_labels)
    dot_param = randomize_dot_param(nbr_sites)
    interaction_param = randomize_interaction_param(nbr_sites) 
    return hamiltonian_type(dot_param, interaction_param, site_labels, f)
end

hamiltonian(H, f) = hamiltonian(hamiltonian_so_b, H, f)

# =================== Hamiltonian for two connected subsystems ==============
hamiltonian_so_b(interaction_param :: InteractionParams, sites_i, sites_j, f) =
    hamiltonian_t(interaction_param.t, sites_i, sites_j, f) + hamiltonian_so(interaction_param.tso, sites_i, sites_j, f) + hamiltonian_c_inter(interaction_param.u_inter, sites_i, sites_j, f) 

function hamiltonian_interaction(hamiltonian_type, H_i, H_j, f)
    site_labels_i = sites(H_i)
    site_labels_j = sites(H_j)
    nbr_sites_i = length(site_labels_i)
    nbr_sites_j = length(site_labels_j)
    interaction_param = randomize_interaction_param(nbr_sites_i, nbr_sites_j)
    hamiltonian_type(interaction_param, site_labels_i, site_labels_j, f)
end

function hamiltonians(hamiltonian_type, H_i, H_j, f)
    ham_i = hamiltonian(hamiltonian_type, H_i, f)
    ham_j = hamiltonian(hamiltonian_type, H_j, f)
    ham_interactions = hamiltonian_interaction(hamiltonian_type, H_i, H_j, f)
    ham_tot = ham_i + ham_j + ham_interactions 
    return Hamiltonians(ham_i, ham_j, ham_interactions, ham_tot)
end