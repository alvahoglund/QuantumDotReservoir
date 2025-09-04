hamiltonian_ϵ(ϵ, sites, f) = sum(ϵ[i]*f[i, σ]' * f[i, σ] 
    for σ ∈ [:↑, :↓], i ∈ sites)

hamiltonian_t(t, sites, f) = sum(
    t[i,j]*f[i, σ]'f[j, σ] + hc
    for σ ∈ [:↑, :↓], i ∈ sites, j ∈ sites if i != j)

hamiltonian(ϵ, t, sites, f) = hamiltonian_ϵ(ϵ, sites, f) + hamiltonian_t(t, sites, f)

function hamiltonian(H, f)
    spatial_sites = sites(H)
    Random.seed!(1)
    nbr_sites = length(spatial_sites)
    ϵ = rand(nbr_sites)
    t = rand(nbr_sites,nbr_sites)  + im*rand(nbr_sites,nbr_sites)
    t = 0.5*(t + t')
    hamiltonian(ϵ, t, spatial_sites,f)
end