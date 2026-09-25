@testset "input: finite chain rings" begin
    @testset "constructors agree" begin
        K, a = qadic_field(2, 2; precision=20, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        X1 = finite_chain_ring(z^2 + 2a * z + 2a, 5)
        @test (X1.p, X1.f, X1.n, X1.e) == (2, 2, 5, 2)
        # the same polynomial as coefficient lists in the Conway generator
        X2 = finite_chain_ring([[0, 2], [0, 2], [1]]; p=2, f=2, n=5)
        @test X1 == X2
        # via an Eisenstein extension
        Lext, _ = eisenstein_extension(z^2 + 2a * z + 2a, :b)
        @test finite_chain_ring(Lext, 5) == X1
        # Teichmüller basis: for F_4 the Conway generator is already a root of unity
        X3 = finite_chain_ring([[0, 2], [0, 2], [1]]; p=2, f=2, n=5, basis=:teichmuller)
        @test X3 == X1
        # ZZ polynomial with f > 1 means E ∈ Z_p[z]
        Zz, t = polynomial_ring(ZZ, :z)
        @test finite_chain_ring(t + 3, 2; p=3, f=2) == truncated_witt_vectors(3, 2, 2)
        @test finite_chain_ring(t + 5, 3; p=5) == zpn(5, 3)
        @test finite_chain_ring(residue_ring(ZZ, 81)[1]) == zpn(3, 4)
        Qp = padic_field(3; precision=10)
        Qz, y = polynomial_ring(Qp, :y)
        @test finite_chain_ring(y + 3, 4) == zpn(3, 4)
        @test prime(X1) == 2 && inertia_degree(X1) == 2 && ramification_index(X1) == 2
    end

    @testset "Teichmüller basis for F_9" begin
        K, a = qadic_field(3, 2; precision=20, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        zeta = teichmuller(a)
        # E = ζ z + 3, entered both as a q-adic polynomial and in the Teichmüller basis
        X1 = finite_chain_ring(zeta * z + 3, 3)
        X2 = finite_chain_ring([[3], [0, 1]]; p=3, f=2, n=3, basis=:teichmuller)
        @test X1.htilde == X2.htilde
        # ζ is computed to the precision that n requires, or to a given precision
        @test X2.Eprec == S.required_input_precision(3, 1)
        X3 = finite_chain_ring(
            [[3], [0, 1]]; p=3, f=2, n=3, basis=:teichmuller, precision=10
        )
        @test X3.Eprec == 10
        @test all(mod.(X1.E[k] .- X3.E[k], big(3)^10) == [0, 0] for k in 1:2)
        @test S.with_power(X3, 10).n == 10
        @test_throws ArgumentError S.with_power(X2, 4)
    end

    @testset "precision of E" begin
        # exact input stays exact (18 is not reduced to 2)
        X = finite_chain_ring([18, 0, 0, 0, 0, 0, 0, 0, 1]; p=2, f=1, n=2)
        @test X.E[1] == [18] && X.Eprec == typemax(Int)
        @test S.with_power(X, 100).n == 100
        # q-adic input records its precision; larger powers n are rejected
        K, a = qadic_field(2, 2; precision=4, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        Y = finite_chain_ring(z^8 + 2a, 2)
        @test Y.Eprec == 4
        @test S.with_power(Y, 32).n == 32
        @test_throws ArgumentError S.with_power(Y, 33)
        @test_throws ArgumentError syntomic_sweep(Y, [2, 33], 1:2)
        @test_throws ArgumentError finite_chain_ring(2, [0, 1], [[2], [1]], 5; Eprec=4)
        # Σ a_j x^j is reduced exactly modulo h̃
        W = finite_chain_ring([[2], [0, 0, 0, 2], [1]]; p=2, f=2, n=3)
        @test W.E[2] == S._reduce_mod_htilde(BigInt[0, 0, 0, 2], W.htilde)
        @test W.E[2] == [2, 0]                               # x³ = 1 for h̃ = x² + x + 1
    end

    @testset "validation" begin
        K, a = qadic_field(2, 2; precision=20, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        @test_throws ArgumentError finite_chain_ring(z^2 + 4, 3)          # E(0) has valuation 2
        @test_throws ArgumentError finite_chain_ring(z^2 + z + 2, 3)      # middle coefficient a unit
        @test_throws ArgumentError finite_chain_ring(2 * z + 2, 3)        # leading coefficient not a unit
        @test_throws ArgumentError finite_chain_ring(z + 2, 0)
        Klow, b = qadic_field(2, 2; precision=2, cached=false)
        Klz, w = polynomial_ring(Klow, :z; cached=false)
        @test_throws ArgumentError finite_chain_ring(w + 2, 5)            # precision 2 < ⌈5/1⌉
        @test_throws ArgumentError zpn(4, 2)
        @test_throws ArgumentError finite_chain_ring(
            [[2], [1]]; p=2, f=2, n=2, htilde=[1, 0, 1]
        )
        @test_throws ArgumentError finite_chain_ring(
            [[2], [1]]; p=2, f=2, n=2, basis=:other
        )
        Zz, t = polynomial_ring(ZZ, :z)
        @test_throws ArgumentError finite_chain_ring(t + 2, 3; p=2, f=3, htilde=[1, 1, 1])
        @test finite_chain_ring(t + 2, 3; p=2, htilde=[1, 1, 1]).f == 2
    end

    @testset "every input error is an ArgumentError" begin
        K, a = qadic_field(2, 2; precision=20, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        Q3 = padic_field(3; precision=10)
        Q3y, y = polynomial_ring(Q3, :y; cached=false)
        Zz, t = polynomial_ring(ZZ, :z)
        h4 = S.default_htilde(2, 2)                                     # x² + x + 1
        # E of degree 0, in every input form
        @test_throws ArgumentError finite_chain_ring(2, h4, [[2, 0]], 3)
        @test_throws ArgumentError finite_chain_ring(2, h4, Vector{Int}[], 3)
        @test_throws ArgumentError finite_chain_ring(Kz(2), 3)
        @test_throws ArgumentError finite_chain_ring(Q3y(3), 3)
        @test_throws ArgumentError finite_chain_ring(Zz(2), 3; p=2)
        @test_throws ArgumentError finite_chain_ring([[2]]; p=2, f=2, n=3)
        @test_throws ArgumentError finite_chain_ring(
            [[2]]; p=2, f=2, n=3, basis=:teichmuller
        )
        # the wrong number of coordinates for a coefficient of E
        @test_throws ArgumentError finite_chain_ring(2, h4, [[2], [1, 0]], 3)
        @test_throws ArgumentError finite_chain_ring(2, h4, [[2, 0, 0], [1, 0, 0]], 3)
        @test_throws ArgumentError finite_chain_ring(2, [0, 1], [[2, 0], [1, 0]], 3)
        # h̃ not monic, of degree 0, or reducible modulo p
        @test_throws ArgumentError finite_chain_ring(2, [1, 1, 3], [[2, 0], [1, 0]], 3)
        @test_throws ArgumentError finite_chain_ring(2, [0, 2], [[2], [1]], 3)
        @test_throws ArgumentError finite_chain_ring(2, [1], [[2], [1]], 3)
        @test_throws ArgumentError finite_chain_ring(2, [1, 0, 1], [[2, 0], [1, 0]], 3)
        @test_throws ArgumentError finite_chain_ring(t + 2, 3; p=2, htilde=[1, 1, 3])
        @test_throws ArgumentError finite_chain_ring(
            [[2], [1]]; p=2, f=2, n=3, htilde=[1, 1, 3]
        )
        # E not Eisenstein
        @test_throws ArgumentError finite_chain_ring(2, [0, 1], [[1], [1]], 2)
        @test_throws ArgumentError finite_chain_ring(2, [0, 1], [[0], [1]], 2)
        @test_throws ArgumentError finite_chain_ring([[2], [0]]; p=2, f=1, n=3)
        @test_throws ArgumentError finite_chain_ring(t^2 + t + 3, 2; p=3)
        # non-integral p-adic and q-adic coefficients
        @test_throws ArgumentError finite_chain_ring(y + inv(Q3(3)), 3)
        @test_throws ArgumentError finite_chain_ring(inv(Q3(3)) * y^2 + 3, 3)
        @test_throws ArgumentError finite_chain_ring(z + inv(K(2)), 3)
        @test_throws ArgumentError finite_chain_ring(z^2 + inv(K(2)) * a * z + 2, 3)
        # residue rings whose modulus is not a prime power
        @test_throws ArgumentError finite_chain_ring(residue_ring(ZZ, 12)[1])
        @test_throws ArgumentError finite_chain_ring(residue_ring(ZZ, ZZ(12))[1])
        @test_throws ArgumentError finite_chain_ring(residue_ring(ZZ, 1)[1])
        @test_throws ArgumentError finite_chain_ring(residue_ring(ZZ, ZZ(2)^70 * 3)[1])
        # p not prime, n < 1, E known to too little precision
        @test_throws ArgumentError zpn(1, 2)
        @test_throws ArgumentError zpn(-2, 2)
        @test_throws ArgumentError finite_chain_ring(4, [0, 1], [[4], [1]], 2)
        @test_throws ArgumentError finite_chain_ring(t + 2, 3; p=4)
        @test_throws ArgumentError zpn(2, 0)
        @test_throws ArgumentError truncated_witt_vectors(2, 2, 0)
        @test_throws ArgumentError finite_chain_ring([[2], [1]]; p=2, f=2, n=0)
        @test_throws ArgumentError finite_chain_ring(
            [[3], [0, 1]]; p=3, f=2, n=-1, basis=:teichmuller
        )
        @test_throws ArgumentError finite_chain_ring(y + 3, 0)
        @test_throws ArgumentError S.with_power(zpn(2, 2), 0)
        @test_throws ArgumentError finite_chain_ring(2, [0, 1], [[2], [1]], 2; Eprec=1)
        @test_throws ArgumentError finite_chain_ring(
            [[3], [0, 1]]; p=3, f=2, n=5, basis=:teichmuller, precision=2
        )
        # a residue degree f < 1, or p not prime, where h̃ defaults to a Conway polynomial
        for f in (0, -1)
            @test_throws ArgumentError truncated_witt_vectors(2, f, 2)
            @test_throws ArgumentError finite_chain_ring(t + 2, 3; p=2, f=f)
            @test_throws ArgumentError finite_chain_ring([[2], [1]]; p=2, f=f, n=3)
        end
        @test_throws ArgumentError truncated_witt_vectors(4, 2, 2)
        @test_throws ArgumentError finite_chain_ring(t + 2, 3; p=4, f=2)
        @test_throws ArgumentError finite_chain_ring([[2], [1]]; p=4, f=2, n=3)
        # the weight i < 1
        X = zpn(2, 2)
        W = truncated_witt_vectors(2, 2, 2)
        for i in (0, -1)
            @test_throws ArgumentError syntomic_cohomology(X, i)
            @test_throws ArgumentError syntomic_cohomology(X, i; N=10)
            @test_throws ArgumentError syntomic_cohomology(X, i; precision_policy=:apriori)
            @test_throws ArgumentError syntomic_cohomology(X, i; precision_policy=:exponent)
            @test_throws ArgumentError syntomic_cohomology(W, i; descent=false)
            @test_throws ArgumentError syntomic_complex(X, i)
            @test_throws ArgumentError syntomic_complex(X, i; N=10)
            @test_throws ArgumentError syntomic_complex(X, i; precision_policy=:apriori)
            @test_throws ArgumentError syntomic_complex(W, i; descent=false)
            @test_throws ArgumentError k_groups(X, i)
            @test_throws ArgumentError SyntomicParams(2, 1, 2, i)
        end
        @test_throws ArgumentError SyntomicParams(2, 0, 2, 1)
        @test_throws ArgumentError SyntomicParams(4, 1, 2, 1)
        # n < 1 or i < 1 in a sweep
        @test_throws ArgumentError syntomic_sweep(X, 0:2, 1:2)
        @test_throws ArgumentError syntomic_sweep(X, [-1], [1])
        @test_throws ArgumentError syntomic_sweep(X, 1:2, 0:2)
        @test_throws ArgumentError syntomic_sweep(W, [2], [-1])
        @test_throws ArgumentError syntomic_sweep([X, W], [2], [0]; distributed=false)
        @test_throws ArgumentError k_theory_table(X, 0:2)
        # unknown precision policies
        @test_throws ArgumentError syntomic_complex(X, 2; precision_policy=:other)
        @test_throws ArgumentError syntomic_sweep(X, 1:2, 1:2; precision_policy=:other)
    end
end
