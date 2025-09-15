struct QuantumDotSystem
    sites_main :: Vector{Int}
    sites_reservoir :: Vector{Int}
    sites_total :: Vector{Int}
    qn_reservoir :: Int
    qn_total :: Int

    H_main_qn:: FermionicHilbertSpaces.AbstractHilbertSpace
    H_reservoir_qn:: FermionicHilbertSpaces.AbstractHilbertSpace
    H_total_qn :: FermionicHilbertSpaces.AbstractHilbertSpace
    f
end

function quantum_dot_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
    sites_main = collect(1:nbr_dots_main)
    sites_reservoir = collect(nbr_dots_main+1:nbr_dots_main+nbr_dots_res)
    sites_total = vcat(sites_main, sites_reservoir)
    qn_total = qn_reservoir + nbr_dots_main

    H_main_qn = tensor_product(hilbert_space(labels(sites_main[1]), NumberConservation(1)), hilbert_space(labels(sites_main[2]), NumberConservation(1)))
    H_reservoir_qn = hilbert_space(labels(sites_reservoir), NumberConservation(qn_reservoir))
    H_total_qn = hilbert_space(labels(vcat(sites_main, sites_reservoir)), NumberConservation(qn_total))

    @fermions f

    return QuantumDotSystem(sites_main, sites_reservoir, sites_total, qn_reservoir,qn_total,
                            H_main_qn, H_reservoir_qn, H_total_qn, f)
end

methods(quantum_dot_system)