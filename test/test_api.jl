# The user-facing API: residue fields, destructuring of the syntomic complex, K-theory tables
# and the export of results.
using JSON3

@testset "API" begin
    @testset "residue_field" begin
        # Z/p^n: h̃ = x, so the generator of F_p is the image 0 of x
        F, red = residue_field(zpn(5, 2))
        @test order(F) == 5 && degree(F) == 1
        @test iszero(gen(F))
        Zx, x = polynomial_ring(ZZ, :x)
        @test domain(red) === Zx && codomain(red) === F
        @test red(x + 7) == F(2)
        # W(F_9) presented by the Conway polynomial, as in qadic_field(3, 2)
        X = truncated_witt_vectors(3, 2, 4)
        F, red = residue_field(X)
        @test order(F) == 9 && characteristic(F) == 3 && degree(F) == 2
        @test red(x) == gen(F)
        @test iszero(red(Zx(X.htilde)))
        coefficients_mod_p(F) = [lift(ZZ, c) for c in coefficients(defining_polynomial(F))]
        @test coefficients_mod_p(F) == [mod(c, 3) for c in X.htilde]
        k, _ = residue_field(qadic_field(3, 2; precision=5, cached=false)[1])
        @test coefficients_mod_p(k) == coefficients_mod_p(F)
        # every coefficient of E reduces into F; the leading one is a unit
        Y = finite_chain_ring([[0, 3], [3], [1, 1]]; p=3, f=2, n=3)
        F, red = residue_field(Y)
        @test red(Zx(Y.E[end])) == gen(F) + 1
        @test all(iszero(red(Zx(c))) for c in Y.E[1:(end - 1)])
        a = gen(F)^5
        @test red(preimage(red, a)) == a && degree(preimage(red, a)) < 2
        # a custom presentation h̃ = x^2 + x + 1 of W(F_4)
        Zz, t = polynomial_ring(ZZ, :z)
        F, red = residue_field(finite_chain_ring(t + 2, 2; p=2, htilde=[1, 1, 1]))
        @test order(F) == 4 && iszero(gen(F)^2 + gen(F) + 1)
    end

    @testset "destructuring the syntomic complex" begin
        C = syntomic_complex(zpn(3, 2), 2)
        syn0, syn1 = C
        @test syn0 === C.syn0 && syn1 === C.syn1
        @test length(C) == 2 && eltype(C) == S.PMatrix
        @test collect(C) == [C.syn0, C.syn1]
        s0, s1 = syntomic_complex(zpn(2, 2), 2; threaded=false)
        @test size(s0.M) == (6, 3) && size(s1.M) == (3, 6)
    end

    @testset "K-theory tables" begin
        tab = k_theory_table(zpn(2, 2), 1:4; integral=true)
        @test tab isa KTheoryTable && tab isa AbstractVector
        @test length(tab) == 8 && size(tab) == (8,)
        @test [r.degree for r in tab] == collect(0:7)
        @test tab[1].exponents === nothing
        @test tab[4] == (degree=3, exponents=[3], prime_to_p_order=big(3))
        @test tab[8].exponents == [1, 3] && tab[8].prime_to_p_order == 15
        @test [r.exponents for r in tab[2:3]] == [[1], [1]]
        @test collect(tab) == tab.rows
        @test sprint(show, MIME("text/plain"), tab) == """
            K-groups of Z/2^2:
              K_0(Z/2^2) ≅ Z
              K_1(Z/2^2) ≅ Z/2
              K_2(Z/2^2) ≅ Z/2
              K_3(Z/2^2) ≅ Z/8 ⊕ Z/3
              K_4(Z/2^2) ≅ 0
              K_5(Z/2^2) ≅ Z/8 ⊕ Z/7
              K_6(Z/2^2) ≅ 0
              K_7(Z/2^2) ≅ Z/2 ⊕ Z/8 ⊕ Z/3 ⊕ Z/5"""
        padic = k_theory_table(zpn(2, 2), 1:2)
        @test !padic.integral && all(isone(r.prime_to_p_order) for r in padic)
        @test sprint(show, MIME("text/plain"), padic) == """
            2-adic K-groups of Z/2^2:
              K_0(Z/2^2; Z_2) ≅ Z_2
              K_1(Z/2^2; Z_2) ≅ Z/2
              K_2(Z/2^2; Z_2) ≅ Z/2
              K_3(Z/2^2; Z_2) ≅ Z/8"""
        # an unlabelled ring with repeated summands and two-digit degrees
        K, a = qadic_field(2, 2; precision=10, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        X = finite_chain_ring(z^2 + 2a * z + 2a, 3)
        text = sprint(show, MIME("text/plain"), k_theory_table(X, 2:2; integral=true))
        @test startswith(text, "K-groups of R = O_K/ϖ^3")
        @test occursin("K_3(R) ≅ (Z/2)^2 ⊕ (Z/8)^2 ⊕ Z/3 ⊕ Z/5", text)
        text = sprint(show, MIME("text/plain"), k_theory_table(zpn(3, 3), 5:6))
        @test occursin("\n  K_9(Z/3^3; Z_3)  ≅ (Z/3)^2 ⊕ Z/3^9", text)
        @test occursin("\n  K_10(Z/3^3; Z_3) ≅ 0", text)
        @test sprint(show, MIME("text/plain"), k_theory_table(zpn(2, 2), Int[])) ==
            "2-adic K-groups of Z/2^2: none"
        # primary decompositions
        @test S.cyclic_primary_decomposition(1) == Tuple{BigInt,Int}[]
        @test S.cyclic_primary_decomposition(big(2)^30 - 1) ==
            [(3, 2), (7, 1), (11, 1), (31, 1), (151, 1), (331, 1)]
        @test S.cyclic_primary_decomposition(big(2)^61 - 1) == [(big(2)^61 - 1, 1)]
        @test_throws ArgumentError S.cyclic_primary_decomposition(0)
        # a cofactor with more than max_digits digits that is not prime is kept whole
        l1, l2 = next_prime(ZZ(10)^12), next_prime(ZZ(10)^13)
        m = BigInt(6 * l1 * l2)
        @test S.cyclic_primary_decomposition(m) ==
            [(2, 1), (3, 1), (BigInt(l1), 1), (BigInt(l2), 1)]
        @test S.cyclic_primary_decomposition(m; max_digits=20) ==
            [(2, 1), (3, 1), (div(m, 6), 1)]
        row = S.KTheoryRow((3, [1, 1, 10], big(9)))
        @test S.group_summands(row, 2) == [(2, 1, 2), (2, 10, 1), (3, 2, 1)]
        @test S.group_string(row, 2) == "(Z/2)^2 ⊕ Z/2^10 ⊕ Z/9"
        @test S.group_string(S.KTheoryRow((0, nothing, big(1))), 3; integral=false) == "Z_3"
        @test_throws ArgumentError S.group_summands(S.KTheoryRow((0, nothing, big(1))), 3)
    end

    @testset "export" begin
        results = syntomic_sweep(zpn(2, 2), 2:3, 1:2)
        # JSON: parsed with JSON3; big integers are strings
        json = sprint(io -> save_results(io, results; format=:json))
        doc = JSON3.read(json)
        @test doc.format == "syntomic-results-v1" && doc.kind == "syntomic_cohomology"
        @test length(doc.rows) == length(results) == 4
        for (row, H) in zip(doc.rows, results)
            @test (row.p, row.f, row.e, row.n, row.i) == (2, 1, 1, H.ring.n, H.i)
            @test row.label == H.ring.label
            @test Vector{Int}(row.h1) == H.h1 && Vector{Int}(row.h2) == H.h2
            @test (row.N, row.prec0, row.prec1) == (H.N, H.prec0, H.prec1)
            @test row.policy == String(H.policy) && row.restarts == H.restarts
            @test row.htilde == ["0", "1"] && row.E == [["2"], ["1"]]
            @test row.time_s ≈ sum(values(H.timings))
            @test Set(keys(row.timings)) == Set(keys(H.timings))
        end
        # CSV: one header line and one line per result; no field contains a comma
        csv = sprint(io -> save_results(io, results; format=:csv))
        lines = split(chomp(csv), '\n')
        @test length(lines) == 5
        header = split(lines[1], ',')
        @test header == String.(S.COHOMOLOGY_COLUMNS)
        for (line, H) in zip(lines[2:end], results)
            cells = Dict(zip(header, split(line, ',')))
            @test parse(Int, cells["n"]) == H.ring.n && parse(Int, cells["i"]) == H.i
            @test cells["h1"] == join(H.h1, ';') && cells["h2"] == join(H.h2, ';')
            @test cells["E"] == "2;1" && cells["htilde"] == "0;1"
            @test parse(Int, cells["N"]) == H.N
        end
        # files: the format follows the extension
        mktempdir() do dir
            path = joinpath(dir, "results.json")
            @test save_results(path, results) == path
            @test JSON3.read(read(path, String)).rows[2].h1 == [3]
            path = joinpath(dir, "results.CSV")
            save_results(path, results)
            @test startswith(read(path, String), "p,f,e,n,i,label,")
            @test_throws ArgumentError save_results(joinpath(dir, "results.txt"), results)
            save_results(joinpath(dir, "results.txt"), results; format=:csv)
            @test isfile(joinpath(dir, "results.txt"))
        end
        @test_throws ArgumentError save_results(IOBuffer(), results; format=:xml)
        @test_throws ArgumentError save_results(IOBuffer(), [1, 2]; format=:json)
        # a single result, results of several rings, and W(F_q) coordinates with f = 2
        H = syntomic_cohomology(truncated_witt_vectors(3, 2, 2), 1)
        doc = JSON3.read(sprint(io -> save_results(io, H; format=:json)))
        @test length(doc.rows) == 1 && doc.rows[1].E == [["3", "0"], ["1", "0"]]
        csv = sprint(io -> save_results(io, [[H], results]; format=:csv))
        @test length(split(chomp(csv), '\n')) == 6
        @test split(split(csv, '\n')[2], ',')[8] == "3 0;1 0"
        @test chomp(sprint(io -> save_results(io, SyntomicCohomology[]; format=:csv))) ==
            join(String.(S.COHOMOLOGY_COLUMNS), ',')
        # labels are quoted when needed
        Y = finite_chain_ring(3, [0, 1], [[3], [1]], 2; label="Z/9, \"the\" ring")
        csv = sprint(io -> save_results(io, syntomic_cohomology(Y, 2); format=:csv))
        @test occursin(",\"Z/9, \"\"the\"\" ring\",", csv)
        doc = JSON3.read(
            sprint(io -> save_results(io, syntomic_cohomology(Y, 2); format=:json))
        )
        @test doc.rows[1].label == "Z/9, \"the\" ring"

        # K-theory tables
        tab = k_theory_table(zpn(3, 3), 1:4; integral=true)
        doc = JSON3.read(sprint(io -> save_results(io, tab; format=:json)))
        @test doc.kind == "k_theory_table" && doc.integral == true
        @test doc.ring.label == "Z/3^3" && doc.ring.E == [["3"], ["1"]]
        @test [r.degree for r in doc.rows] == collect(0:7)
        @test doc.rows[1].exponents === nothing && doc.rows[1].group == "Z"
        @test doc.rows[2].exponents == [2] && doc.rows[2].prime_to_p_order == "2"
        @test doc.rows[4].group == S.group_string(tab[4], 3)
        csv = sprint(io -> save_results(io, tab; format=:csv))
        lines = split(chomp(csv), '\n')
        @test lines[1] == join(String.(S.KTHEORY_COLUMNS), ',')
        @test length(lines) == 9
        cells = split(lines[5], ',')             # K_3(Z/27) = Z/3 ⊕ Z/27 ⊕ Z/8
        @test cells[7:10] == ["3", "1;3", "8", "Z/3 ⊕ Z/27 ⊕ Z/8"]
        @test split(lines[2], ',')[7:10] == ["0", "", "1", "Z"]
    end
end
