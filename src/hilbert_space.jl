labels(sites) = [(site, spin) for site in sites for spin in (:↑, :↓)]
sites(H) = unique(first.(keys(H)))

sector_index(Hsub::FermionicHilbertSpaces.AbstractHilbertSpace, H::FermionicHilbertSpaces.AbstractHilbertSpace) = map(Base.Fix2(FermionicHilbertSpaces.state_index, H), basisstates(Hsub))
sector_index(qn:: Int, H::FermionicHilbertSpaces.AbstractHilbertSpace) = map(Base.Fix2(FermionicHilbertSpaces.state_index, H), H.symmetry.qntofockstates[qn])

pad(H::FermionicHilbertSpaces.AbstractHilbertSpace) = hilbert_space(keys(H), NumberConservation())