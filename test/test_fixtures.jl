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
            Ha = syntomic_cohomology(X, i; precision_policy=:adaptive)
            @test (Ha.h1, Ha.h2) == (Vector{Int}(c.h1), Vector{Int}(c.h2))
            X.n * i > 1 || continue
            N = S.apriori_working_precision(X.p, X.n, i)
            W = S.syntomic_wmatrices(X, i, N)
            for (name, fname) in (
                (:can0, :can0), (:Phi, :phi0), (:can1, :can1), (:red, :red), (:redN, :redN)
            )
                @test wmatrix_agrees(getfield(W, name), c.matrices[fname])
            end
            # ∇_OK is compared for odd p (for p = 2 the fixture uses generators g_u that differ by
            # units, see the fixture's provenance)
            isodd(X.p) && @test wmatrix_agrees(W.nablaOK, c.matrices.nablaOK)
        end
    end

    @testset "σ-patched Sage (f > 1): cohomology and W-matrices" begin
        fx = read_fixture("sage_fq.json")
        for c in fx.cases
            tier_allows(c.L) || continue
            X = fixture_ring(c)
            i = Int(c.i)
            for policy in (:apriori, :adaptive)
                H = syntomic_cohomology(X, i; precision_policy=policy)
                @test (H.h1, H.h2) == (Vector{Int}(c.h1), Vector{Int}(c.h2))
            end
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

    @testset "[AKN, Appendix A]: the orders of all transcribed groups" begin
        # |K_{2i-1}(Z/p^n)| / |K_{2i-2}(Z/p^n)| = p^{i(n-1)} ([AKN, Prop. 2.18, p. 9]) for
        # every weight i of every table. K_0(Z/p^n; Z_p) = Z_p is not tabulated, and for
        # i = 1 the identity reads |K_1(Z/p^n; Z_p)| = p^{n-1}.
        fx = read_fixture("akn_tables.json")
        cells = Dict(
            (Int(row.p), Int(row.n), Int(row.r)) => Vector{Int}(row.exponents) for
            row in fx.appendix_A
        )
        @test length(cells) == length(fx.appendix_A) > 700    # no duplicated cells
        # every printed cell K_r, 1 ≤ r ≤ r_max, is tabulated or listed as not computed
        missing_cells = Set(
            (Int(x.p), Int(x.n), Int(r)) for x in fx.provenance.appendix_A_not_computed for
            r in x.r
        )
        printed = Set(
            (parse(Int, String(p)), Int(n), r) for
            (p, t) in pairs(fx.provenance.appendix_A_rows_printed) for n in t.n for
            r in 1:Int(t.r_max)
        )
        @test isempty(intersect(missing_cells, keys(cells)))
        @test printed == union(missing_cells, keys(cells))
        @test all(all(>(0), ex) && issorted(ex) for ex in values(cells))
        # the degrees 2i-1 and 2i-2 of a weight i ≥ 2 are both tabulated or both missing
        unpaired = [
            (p, n, r) for (p, n, r) in keys(cells) if
            r >= 2 && !haskey(cells, (p, n, isodd(r) ? r - 1 : r + 1))
        ]
        @test isempty(unpaired)
        # every (p, n) has a tabulated K_1
        @test all(haskey(cells, (p, n, 1)) for (p, n, _) in keys(cells))
        failures = Tuple{Int,Int,Int}[]
        weights = 0
        for ((p, n, r), ex) in cells
            isodd(r) || continue
            i = div(r + 1, 2)
            even = i == 1 ? Int[] : get(cells, (p, n, r - 1), nothing)
            even === nothing && continue
            weights += 1
            sum(ex) - sum(even; init=0) == i * (n - 1) || push!(failures, (p, n, i))
        end
        @test isempty(failures)
        @test weights == count(isodd(r) for (_, _, r) in keys(cells))
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
