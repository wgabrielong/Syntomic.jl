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
        # exact data agree modulo p^⌈n/e⌉ + 2
        m = big(3)^(S.required_input_precision(3, 1) + 2)
        @test all(mod.(X1.E[k] .- X2.E[k], m) == [0, 0] for k in 1:2)
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
    end
end
