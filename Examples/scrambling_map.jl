##############
using QuantumDotReservoir, LinearAlgebra
import QuantumDotReservoir as QRC
nbr_dots_main = 2
nbr_dots_res = 4
qn_reservoir = 3
sys = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
seed = 2
hams = hamiltonians(sys, seed)
ψres = ground_state(hams.hamiltonian_reservoir, sys.H_reservoir_qn, QRC.ArnoldiAlg())
measurements =charge_measurements(sys)
t = 10
@time sm_pure = scrambling_map(sys, measurements, ψres, hams.hamiltonian_total, t, QRC.PureStatePropagatorAlg(; krylov_dim=200, tol=1e-6));
## Check convergence of the scrambling map with respect to the Krylov dimension!
sm_pures = [scrambling_map(sys, measurements, ψres, hams.hamiltonian_total, t, QRC.PureStatePropagatorAlg(; krylov_dim, tol=1e-6)) for krylov_dim in [100, 200, 300]]
norm.(diff(sm_pures))
##
@profview sm_pure = scrambling_map(sys,measurements, ψres, hams.hamiltonian_total, t, QRC.PureStatePropagatorAlg(; krylov_dim=200, tol=1e-6));
##
@time reservoir_state = ground_state(hams.hamiltonian_reservoir, sys.H_reservoir_qn, QRC.ExactDiagonalizationAlg())
ρres = ψres * ψres'
@time sm_block = scrambling_map(sys, measurements, ρres, hams.hamiltonian_total, t, QRC.BlockPropagatorAlg());
sm_block ≈ sm_pure

##
initial_states = [def_state(triplet_plus, sys.H_main, sys.f),
    def_state(singlet, sys.H_main, sys.f),
    random_product_state(sys),
    random_separable_state(3, sys)]

@profview sm = scrambling_map(sys, measurements, reservoir_state, hams.hamiltonian_total, t, QuantumDotReservoir.BlockPropagatorAlg());
@profview sm = scrambling_map(sys, measurements, reservoir_state, hams.hamiltonian_total, t, QuantumDotReservoir.PureStatePropagatorAlg());
