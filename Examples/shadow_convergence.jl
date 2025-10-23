## ============  Define system ============ 
function def_system(t, nbr_dots_res, measurement)
    nbr_dots_main = 2
    #nbr_dots_res = 2
    qn_reservoir = 0
    qd_system = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_reservoir)
    seed = 1
    #t = [10,20,30, 40]

    S = scrambling_map(qd_system, t, measurement, seed)
    R = pinv(S)
    n = outcomes(length(qd_system.coordinates_total)*1*length(t))
    return qd_system, S, R
end

function outcomes(len)
    ranges = ntuple(_ -> 0:1, len)
    tuples = Iterators.product(ranges...)
    vectors = [collect(t) for t in tuples]
    return reshape(vectors, 1, :)
end

outcome_probabilites(ρ, S, n) = [real(prod(n[i] .* (S*ρ) + (1 .-n[i]).* (1 .-(S*ρ)))) for i in eachindex(n)]
function test(P_n, R, n)
    sum([P_n[i]*R*n[i] for i in eachindex(n)]) ≈ ρ
end

## =============== Shadows =========
using StatsBase
function state_shadow(R, n, P_n, iterations)
    shadows = Vector{Vector{ComplexF64}}()
    current_sum = zeros(ComplexF64, 16)
    for i in 1:iterations
        snapshot = R * sample(n, Weights(P_n))
        current_sum += snapshot
        push!(shadows, copy(current_sum / i))
    end
    return shadows
end
function operator_shadow(O, R, n, P_n, iterations)
    shadows = Vector{ComplexF64}()
    current_sum = 0.0 + im*0.0
    for i in 1:iterations
        snapshot = O * R * sample(n, Weights(P_n))
        current_sum += snapshot
        push!(shadows, copy(current_sum / i))
    end
    return shadows
end
state_shadow_variance(ρ, R, n, P_n) = sum(norm(P_n[i]*(R*n[i])-ρ)^2 for i in eachindex(n))
operator_shadow_variance(O, ρ, R, n, P_n) = sum(norm(P_n[i]*(op*R*n[i])-O*ρ)^2 for i in eachindex(n))
function plot_state_shadow(ρ, state_shadow_list)
    plot(
        map(shadow -> norm(shadow .-ρ), state_shadow_list),
        ylim= (0,1)
    )
end
function plot_operator_shadow(expectation_value, operator_shadow_list)
    plot(
        map(shadow -> norm(shadow -expectation_value), operator_shadow_list),
        ylim= (0, 1)
    )
end
## ===== Plot shadows after a number of measurements ======
t = [10, 20, 30, 40]
nbr_dots_res =3 
measurement = single_charge_probabilities
qd_system, S, R= def_system(t, nbr_dots_res, measurement)
n = outcomes((nbr_dots_res+2)*length(t))

ind = FermionicHilbertSpaces.indices(qd_system.H_main_qn, qd_system.H_main)
ρ = vec(random_product_state(qd_system)[ind, ind])
P_n = outcome_probabilites(ρ, S, n)
test(P_n, R, n)
cond(S), rank(S)
iterations = 10^5
state_shadow_list = state_shadow(R, n, P_n, iterations)
plot_state_shadow(ρ, state_shadow_list)
state_shadow_list[end] .- ρ

op = transpose(vec(pauli_string(σz, σx, qd_system)[ind,ind]'))
operator_shadow_list = operator_shadow(op, R, n, P_n, iterations)
plot_operator_shadow(tr(op*ρ), operator_shadow_list)
operator_shadow_list[end] - op*ρ
