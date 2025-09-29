struct QuantumDotSystem
    coordinates_main
    coordinates_reservoir
    coordinates_total
    coordinates_intersection

    qn_main :: Int
    qn_reservoir :: Int
    qn_total :: Int

    H_main_qn:: FermionicHilbertSpaces.AbstractHilbertSpace
    H_main:: FermionicHilbertSpaces.AbstractHilbertSpace
    H_reservoir:: FermionicHilbertSpaces.AbstractHilbertSpace
    H_total:: FermionicHilbertSpaces.AbstractHilbertSpace

    f
end

labels(coordinates) = [(coordinate, spin) for coordinate in coordinates for spin in (:↑, :↓)]
sites(H) = unique(first.(keys(H)))

function tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
    coordinates_main, coordinates_reservoir, coordinates_total, coordinates_intersection = generate_grid(nbr_dots_main, nbr_dots_res)
    qn_total = qn_reservoir + nbr_dots_main

    H_main_qn = tensor_product(hilbert_space(labels([coordinates_main[1]]), NumberConservation(1)), hilbert_space(labels([coordinates_main[2]]), NumberConservation(1)))
    H_main = hilbert_space(keys(H_main_qn), NumberConservation())
    H_reservoir = hilbert_space(labels(coordinates_reservoir), NumberConservation())
    H_total = hilbert_space(labels(coordinates_total), NumberConservation())

    @fermions f

    QuantumDotSystem(coordinates_main, coordinates_reservoir, coordinates_total, coordinates_intersection,
                    nbr_dots_main, qn_reservoir, qn_total, H_main_qn, H_main, H_reservoir, H_total, f)
end

function generate_grid(nbr_dots_main::Int, nbr_dots_reservoir::Int)
    coordinates_main = [(i, 1) for i in 1:nbr_dots_main]
    coordinates_reservoir = [
        (mod1(i, nbr_dots_main), div(i - 1, nbr_dots_main) + 2)
        for i in 1:nbr_dots_reservoir 
    ]
    coordinates_total = vcat(coordinates_main, coordinates_reservoir)
    coordinates_intersection = vcat(coordinates_main, [coordinate for coordinate in coordinates_reservoir if coordinate[2] ==2])
    return coordinates_main, coordinates_reservoir, coordinates_total, coordinates_intersection
end

function fully_connected_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
    sites_main = collect(1:nbr_dots_main)
    sites_reservoir = collect(nbr_dots_main+1:nbr_dots_main+nbr_dots_res)
    sites_total = vcat(sites_main, sites_reservoir)
    qn_total = qn_reservoir + nbr_dots_main

    H_main_qn = tensor_product(hilbert_space(labels(sites_main[1]), NumberConservation(1)), hilbert_space(labels(sites_main[2]), NumberConservation(1)))
    H_main = hilbert_space(keys(H_main_qn), NumberConservation())
    H_reservoir = hilbert_space(labels(sites_reservoir), NumberConservation())
    H_total = hilbert_space(labels(vcat(sites_main, sites_reservoir)), NumberConservation())

    @fermions f

    return QuantumDotSystem(sites_main, sites_reservoir, sites_total, sites_total, nbr_dots_main, qn_reservoir, qn_total,
                            H_main_qn, H_main, H_reservoir, H_total, f)
end