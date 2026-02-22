using QuantumDotReservoir, LinearAlgebra
import QuantumDotReservoir as QRC

## ===== DEFINING A SYSTEM FOR ALL EXAMPLES ==========
nbr_dots_main = 2
nbr_dots_res = 5
qn_reservoir = 4
qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)

hams = hamiltonians(qd_system)


## == CONSTRUCT SCRAMBLING MAP =======
t = 1
Ψ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir_qn)
@profview S = scrambling_map(qd_system, QRC.charge_probabilities(qd_system), Ψ_res, hams.hamiltonian_total, t, QRC.BlockPropagatorAlg())

qn_ind = indices(qd_system.H_main_qn, qd_system.H_main)
ρ_main_vec = reshape(def_state(singlet, qd_system.H_main, qd_system.f)[qn_ind, qn_ind], 16, 1)
S*ρ_main_vec

## ======= CONSTRUCTING EFFECTIVE MEASUREMENT ====

Ψ_res = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir_qn)

op = QRC.p1(qd_system.coordinates_main[1], qd_system.f)
t = 1 
op_t = operator_time_evolution(t, op, hams.hamiltonian_total, qd_system.H_total_qn) 
eff_m = QRC.effective_measurement(op_t, Ψ_res, qd_system)

qn_ind = indices(qd_system.H_main_qn, qd_system.H_main)
ρ_main = def_state(singlet, qd_system.H_main, qd_system.f)[qn_ind, qn_ind]
tr(eff_m*ρ_main)

## ===== EVOLVING STATES ==========================

ρ_main = def_state(singlet, qd_system.H_main, qd_system.f) 

Ψ_res_qn = ground_state(hams.hamiltonian_reservoir, qd_system.H_reservoir_qn)
ρ_res_qn = Ψ_res_qn*Ψ_res_qn'
ρ_res = embed(ρ_res_qn, qd_system.H_reservoir_qn => qd_system.H_reservoir) 

qn_ind = indices(qd_system.H_total_qn, qd_system.H_total)
ρ_tot_qn = tensor_product((ρ_main, ρ_res), (qd_system.H_main, qd_system.H_reservoir) => qd_system.H_total)[qn_ind, qn_ind]

t = 1
ρ_t = state_time_evolution(t, ρ_tot_qn, hams.hamiltonian_total, qd_system.H_total_qn)

op = matrix_representation(QRC.p1(qd_system.coordinates_main[1], qd_system.f), qd_system.H_total_qn)
tr(op*ρ_t)

##


