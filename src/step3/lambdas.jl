# The units λ_u of [AKN24, Lemma 3.24, p. 21], shared by both prismatic envelopes of Step 3.
#
# [AKN24, §7.1, Step 3, p. 55]: "find elements λ_u ∈ W(F_q)[[z_0]] as in Lemma 3.24". With d = E(z_0):
#     λ_0 = -1/δ(d),     λ_{u+1} = λ_u^p / (1 - δ(d^{p^{u+1}} λ_u)).
# Each step applies δ once and loses one digit of p-adic precision ([AKN24, Lemma 7.5(a), p. 59]).

"""
    LambdaData

The units `λ_0, …, λ_M` of [AKN24, Lemma 3.24, p. 21] modulo `z^zprec`, together with the
denominators `D_u = 1 - δ(E^{p^{u+1}} λ_u)` (`0 ≤ u < M`), so that `λ_{u+1} = λ_u^p / D_u`.
Also stores the powers `Epow[u+1] = E^{p^u}` (`0 ≤ u ≤ M + 1`).
"""
struct LambdaData
    zprec::Int
    lambda::Vector{WSeries}
    D::Vector{WSeries}
    Epow::Vector{WSeries}
end

"""
    lambda_zero(ed::EisensteinData) -> WSeries

`λ_0 = -1/δ(E)` ([AKN24, Lemma 3.24, p. 21]).
"""
lambda_zero(ed::EisensteinData) = ws_neg(ws_inv(ed.deltaE))

"""
    lambda_denominator(Epow_next::WSeries, lambda_u::WSeries) -> WSeries

`D_u = 1 - δ(E^{p^{u+1}} λ_u)`, where `Epow_next = E^{p^{u+1}}`
([AKN24, Lemma 3.24, p. 21]).
"""
function lambda_denominator(Epow_next::WSeries, lambda_u::WSeries)
    zp = lambda_u.zprec
    x = ws_mul(Epow_next, lambda_u; zprec=zp)
    return ws_sub(ws_one(x.R, zp), ws_delta(x; zprec=zp))
end

"""
    lambda_next(lambda_u::WSeries, D_u::WSeries) -> WSeries

`λ_{u+1} = λ_u^p / D_u` ([AKN24, Lemma 3.24, p. 21]).
"""
function lambda_next(lambda_u::WSeries, D_u::WSeries)
    zp = lambda_u.zprec
    return ws_mul(ws_pow(lambda_u, lambda_u.R.p; zprec=zp), ws_inv(D_u; zprec=zp); zprec=zp)
end

"""
    LambdaData(ed::EisensteinData, M) -> LambdaData

`λ_0, …, λ_M` and `D_0, …, D_{M-1}` modulo `z^{ed.zprec}` ([AKN24, Lemma 3.24, p. 21]).
"""
function LambdaData(ed::EisensteinData, M::Integer)
    zp = ed.zprec
    p = ed.R.p
    Epow = [ed.E]
    for _ in 1:(M + 1)
        push!(Epow, ws_pow(Epow[end], p; zprec=zp))
    end
    M < 0 && return LambdaData(zp, WSeries[], WSeries[], Epow)
    lambda = [lambda_zero(ed)]
    D = WSeries[]
    for u in 0:(M - 1)
        push!(D, lambda_denominator(Epow[u + 2], lambda[u + 1]))
        push!(lambda, lambda_next(lambda[u + 1], D[u + 1]))
    end
    return LambdaData(zp, lambda, D, Epow)
end
