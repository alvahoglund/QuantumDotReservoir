struct HamiltonianParams
    ϵ::Vector{Float64}
    ϵb::Matrix{Float64}
    t::Matrix{ComplexF64}
    tso::Matrix{ComplexF64}
    u_intra::Vector{Float64}
    u_inter::Matrix{Float64}
end

hamiltonian_ϵ(ϵ, sites, f) = sum(
    ϵ[i]*f[label, σ]' * f[label, σ] 
    for σ ∈ [:↑, :↓], (i,label) ∈ enumerate(sites))

hamiltonian_b(ϵb,sites, f) = sum(
    ϵb[i]*(-1)^(n+1)*f[label,σ]'f[label,σ]
    for (n,σ) ∈ enumerate([:↑, :↓]), (i,label) ∈ enumerate(sites))

hamiltonian_t(t, sites, f) = sum(
    t[i,j]*f[label_i, σ]'f[label_j, σ] + hc
    for σ ∈ [:↑, :↓], (i,label_i) ∈ enumerate(sites), (j,label_j) ∈ enumerate(sites) if i < j)

hamiltonian_so(tso, sites, f) = sum(
    tso[i,j]*f[label_i,:↑]'f[label_j,:↓] + tso[i,j]*f[label_i,:↓]'f[label_j,:↑] +hc
    for (i,label_i) ∈ enumerate(sites), (j, label_j) ∈ enumerate(sites) if i<j)

hamiltonian_c_intra(u_intra, sites, f) = sum(
    u_intra[i]*f[label,:↑]'f[label, :↓]'f[label, :↓]f[label, :↑]
    for (i, label) ∈ enumerate(sites))

hamiltonian_c_inter(u_inter, sites, f) = sum(
    u_inter[i,j]*f[label_i, σ1]'f[label_j, σ2]'f[label_j, σ2]f[label_i, σ1]
    for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓], (i, label_i) ∈ enumerate(sites), (j,label_j) ∈ enumerate(sites) if i < j)

hamiltonian_simple(param :: HamiltonianParams, sites, f) = hamiltonian_ϵ(param.ϵ, sites, f) + hamiltonian_t(param.t, sites, f)

hamiltonian_so_b(param :: HamiltonianParams, sites, f) = hamiltonian_ϵ(param.ϵ, sites, f) + hamiltonian_b(param.ϵb, sites, f) + hamiltonian_t(param.t, sites,f) +
    hamiltonian_so(param.tso, sites, f) + hamiltonian_c_intra(param.u_intra, sites, f) +hamiltonian_c_inter(param.u_inter, sites, f) 

function randomize_param(nbr_sites)
    Random.seed!(1)
    ϵ = rand(nbr_sites)
    ϵb = rand(nbr_sites, nbr_sites)
    t = rand(ComplexF64, nbr_sites, nbr_sites)
    tso = rand(ComplexF64, nbr_sites, nbr_sites)
    u_intra = rand(nbr_sites)
    u_inter = rand(nbr_sites, nbr_sites)
    return HamiltonianParams(ϵ, ϵb, t, tso, u_intra, u_inter)
end 

function hamiltonian(hamiltonian_type, H, f)
    site_labels = sites(H)
    nbr_sites = length(site_labels)
    param = randomize_param(nbr_sites)
    return hamiltonian_type(param, site_labels, f)
end

hamiltonian(H, f) = hamiltonian(hamiltonian_so_b, H, f)

