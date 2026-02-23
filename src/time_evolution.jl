propagator(t, hamiltonian::AbstractMatrix) = cis(-t * Matrix(hamiltonian))
propagator(t, hamiltonian :: FermionicHilbertSpaces.NonCommutativeProducts.NCAdd, H ::FermionicHilbertSpaces.AbstractHilbertSpace) = 
    propagator(t, matrix_representation(hamiltonian, H))

operator_time_evolution(propagator ::AbstractMatrix, operator :: AbstractMatrix) = 
    propagator' * operator * propagator
operator_time_evolution(t, operator :: AbstractMatrix, hamiltonian :: AbstractMatrix) = 
    operator_time_evolution(propagator(t, hamiltonian), operator)
operator_time_evolution(t, operator :: FermionicHilbertSpaces.NonCommutativeProducts.NCAdd, hamiltonian :: FermionicHilbertSpaces.NonCommutativeProducts.NCAdd, H) = 
    operator_time_evolution(t, matrix_representation(operator, H), matrix_representation(hamiltonian, H))

state_time_evolution(propagator ::AbstractMatrix, ρ ::AbstractMatrix) = propagator * ρ * propagator'
state_time_evolution(t, ρ :: AbstractMatrix, hamiltonian :: AbstractMatrix) = state_time_evolution(propagator(t, hamiltonian), ρ)
state_time_evolution(t, ρ :: AbstractMatrix, hamiltonian :: FermionicHilbertSpaces.NonCommutativeProducts.NCAdd, H ::FermionicHilbertSpaces.AbstractHilbertSpace) = 
    state_time_evolution(propagator(t, hamiltonian, H), ρ)
