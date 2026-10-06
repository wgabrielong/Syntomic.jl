# Step 6: the reduction maps red and red^∇ induced by O_K → O_K/ϖ^n = R.
#
# [AKN24, §7.1, Step 6, p. 56]: "Reduce z_0^k for 1 ≤ k ≤ in-1 with respect to the relations in
# Δ^(1)_{R/W(F_q)[[z_0]]} and express as a linear combination of the basis elements z_0^k ∏ f_u^{e_u}
# to obtain a matrix red … Analogously (taking the indexing shift into account), compute a matrix
# red^∇."
#
# Indexing: red has rows and columns indexed by F-weights 1, …, L-1 (the bases z_0^k w(s^i) of
# Δ^(1)_{O_K/A_0}{i} and b_w w(s^i) of Δ^(1)_{R/A_0}{i}); red^∇ has rows and columns indexed by
# F-weights 0, …, L-2 (the ∇-terms F^{[0,in-2]}Δ^(1){i-1}, [AKN24, §7.1, Step 2, p. 54]).

"""
    f_column_matrix(env, columns, rows) -> WMatrix

The W-matrix whose `j`-th column holds the coefficients of `columns[j]` (an `FElem`) on the basis
monomials of the F-weights in `rows`.
"""
function f_column_matrix(
    env::FEnvelope, columns::Vector{FElem}, rows::AbstractVector{<:Integer}
)
    A = [f_coefficient(columns[j], w) for w in rows, j in eachindex(columns)]
    prec = minimum((x.prec for x in columns); init=env.R.N)
    return WMatrix(env.R, A, prec)
end

"""
    reduction_matrix(env; threaded=true) -> WMatrix

The matrix `red` of `F^{[1,in-1]}Δ^(1)_{O_K/A_0}{i} → F^{[1,in-1]}Δ^(1)_{R/A_0}{i}`: column `k` is the
normal form of `z_0^k`, `1 ≤ k ≤ L-1` ([AKN24, §7.1, Step 6, p. 56]).
"""
function reduction_matrix(env::FEnvelope; threaded::Bool=true)
    L = env.prm.L
    cols = tmap(k -> f_z_power(env, k), 1:(L - 1); threaded=threaded)
    return f_column_matrix(env, collect(FElem, cols), 1:(L - 1))
end

"""
    reduction_matrix_nabla(env; threaded=true) -> WMatrix

The matrix `red^∇` on the `∇`-terms: column `m` is the normal form of `z_0^m`, `0 ≤ m ≤ L-2`
([AKN24, §7.1, Step 6, p. 56]).
"""
function reduction_matrix_nabla(env::FEnvelope; threaded::Bool=true)
    L = env.prm.L
    cols = tmap(m -> f_z_power(env, m), 0:(L - 2); threaded=threaded)
    return f_column_matrix(env, collect(FElem, cols), 0:(L - 2))
end
