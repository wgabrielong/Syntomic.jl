# Regression tests against independent references:
#   * test/fixtures/akn_tables.json — [AKN, Appendix A, pp. 61–64] and [AKN, Fig. 1, p. 4];
#   * test/fixtures/sage_f1.json    — the original SageMath code (f = 1);
#   * test/fixtures/sage_fq.json    — a σ-patched copy of it (f > 1; lower trust);
#   * test/oracles/*.jl             — brute-force principal units (i = 1) and the
#                                     Hesselholt–Madsen formula (n ≤ e).
# All p-adic numbers in the fixtures are decimal strings.
using JSON3

include(joinpath(@__DIR__, "oracles", "brute_force_units.jl"))
include(joinpath(@__DIR__, "oracles", "hesselholt_madsen.jl"))

const FIXTURES = joinpath(@__DIR__, "fixtures")
const MAX_L_TIER1 = 24
const MAX_L_TIER2 = 64

read_fixture(name) = JSON3.read(read(joinpath(FIXTURES, name), String))
parse_w(v) = BigInt[parse(BigInt, String(x)) for x in v]

function fixture_ring(c)
    E = haskey(c, :E_exact_signed) ? c.E_exact_signed : c.E
    return finite_chain_ring(
        Int(c.p),
        BigInt[parse(BigInt, String(x)) for x in c.h_tilde],
        [parse_w(x) for x in E],
        Int(c.n),
    )
end

# Compare our W-matrix with a fixture matrix entrywise modulo the smaller precision.
function wmatrix_agrees(ours::S.WMatrix, fx)
    size(ours.A) == (fx.rows, fx.cols) || return false
    p = ours.R.p
    for r in 1:(fx.rows), c in 1:(fx.cols)
        prec = min(Int(fx.prec[r][c]), ours.prec)
        m = big(p)^prec
        theirs = parse_w(fx.entries[r][c])
        mine = BigInt.(ours.A[r, c])
        all(mod(mine[d] - theirs[d], m) == 0 for d in eachindex(mine)) || return false
    end
    return true
end

tier_allows(L) = L <= MAX_L_TIER1 || (LONG_TESTS && L <= MAX_L_TIER2)

@testset "fixtures and oracles" begin
    @testset "Sage (f = 1): cohomology and W-matrices" begin
        fx = read_fixture("sage_f1.json")
        for c in fx.cases
            c.error === nothing || continue
            tier_allows(c.L) || continue
            X = fixture_ring(c)
            i = Int(c.i)
            H = syntomic_cohomology(X, i; precision_policy=:exponent)
            @test (H.h1, H.h2) == (Vector{Int}(c.h1), Vector{Int}(c.h2))
            X.n * i > 1 || continue
            N = S.apriori_working_precision(X.p, X.n, i)
            W = S.syntomic_wmatrices(X, i, N)
            for (name, fname) in (
                (:can0, :can0), (:Phi, :phi0), (:can1, :can1), (:red, :red), (:redN, :redN)
            )
                @test wmatrix_agrees(getfield(W, name), c.matrices[fname])
            end
            # ∇_OK agrees with the fixture for odd p (the sign conventions differ by units for p = 2)
            isodd(X.p) && @test wmatrix_agrees(W.nablaOK, c.matrices.nablaOK)
        end
    end

    @testset "σ-patched Sage (f > 1): cohomology and W-matrices" begin
        fx = read_fixture("sage_fq.json")
        for c in fx.cases
            tier_allows(c.L) || continue
            X = fixture_ring(c)
            i = Int(c.i)
            H = syntomic_cohomology(X, i)
            @test (H.h1, H.h2) == (Vector{Int}(c.h1), Vector{Int}(c.h2))
            X.n * i > 1 || continue
            N = S.apriori_working_precision(X.p, X.n, i)
            W = S.syntomic_wmatrices(X, i, N)
            for (name, fname) in (
                (:can0, :can0), (:Phi, :phi0), (:can1, :can1), (:red, :red), (:redN, :redN)
            )
                @test wmatrix_agrees(getfield(W, name), c.matrices[fname])
            end
            isodd(X.p) && @test wmatrix_agrees(W.nablaOK, c.matrices.nablaOK)
        end
    end

    @testset "[AKN, Appendix A] and [AKN, Fig. 1]" begin
        fx = read_fixture("akn_tables.json")
        # group the table by (p, n, i)
        wanted = Dict{Tuple{Int,Int,Int},Dict{Symbol,Vector{Int}}}()
        for row in fx.appendix_A
            p, n, r = Int(row.p), Int(row.n), Int(row.r)
            i = isodd(r) ? div(r + 1, 2) : div(r + 2, 2)
            isodd(r) || i >= 2 || continue
            d = get!(wanted, (p, n, i), Dict{Symbol,Vector{Int}}())
            d[isodd(r) ? :h1 : :h2] = Vector{Int}(row.exponents)
        end
        for ((p, n, i), d) in sort(collect(wanted); by=first)
            tier_allows(n * i) || continue
            H = syntomic_cohomology(zpn(p, n), i; precision_policy=:exponent)
            haskey(d, :h1) && @test H.h1 == d[:h1]
            haskey(d, :h2) && @test H.h2 == d[:h2]
        end
        rings = Dict(
            "F2" => zpn(2, 1),
            "Z/4" => zpn(2, 2),
            "Z/8" => zpn(2, 3),
            "F2[z]/z^3" => finite_chain_ring(2, [0, 1], [[2], [0], [0], [1]], 3),
        )
        for row in fx.figure_1
            X = rings[String(row.ring)]
            r = Int(row.r)
            i = isodd(r) ? div(r + 1, 2) : div(r + 2, 2)
            H = syntomic_cohomology(X, i)
            if isodd(r)
                @test H.h1 == Vector{Int}(row.p_exponents)
                @test parse(BigInt, String(row.prime_to_p_order)) == big(2)^i - 1
            else
                @test H.h2 == Vector{Int}(row.p_exponents)
            end
        end
    end

    @testset "brute-force principal units (i = 1)" begin
        for (p, h, E, n, expected) in BRUTE_FORCE_UNITS_CASES
            X = finite_chain_ring(p, h, E, n)
            @test syntomic_cohomology(X, 1).h1 == expected
            @test principal_units_exponents(p, h, E, n) == expected
        end
    end

    @testset "Hesselholt–Madsen (n ≤ e: O_K/ϖ^n = F_q[z]/z^n)" begin
        for (p, f, e, n, i) in (
            (2, 1, 3, 3, 2),
            (2, 1, 3, 3, 3),
            (3, 1, 2, 2, 2),
            (3, 1, 3, 3, 2),
            (5, 1, 2, 2, 3),
            (2, 2, 2, 2, 3),
            (2, 2, 3, 3, 2),
            (3, 2, 2, 2, 2),
            (2, 3, 2, 2, 2),
            (5, 2, 3, 2, 2),
            (3, 3, 2, 2, 2),
        )
            Zz, t = polynomial_ring(ZZ, :z)
            X = finite_chain_ring(t^e + p, n; p=p, f=f)
            H = syntomic_cohomology(X, i)
            @test H.h1 == hm_exponents(p, f, n, i)
            @test H.h2 == Int[]
        end
    end
end
