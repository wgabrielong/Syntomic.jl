# The low-level kernels of src/arith/kernels.jl: in-place FLINT primitives and products of
# W-polynomials (schoolbook, Karatsuba and Kronecker packing in x) against naive oracles.

# The coefficients a_0, …, a_{n-1} (in W(F_q)/p^N) of a W-polynomial.
function kernel_coeffs(R, A, n)
    return [ZZRingElem[lift(coeff(A[d], k)) for d in 1:(R.f)] for k in 0:(n - 1)]
end

# The naive truncated product of two W-polynomials, as a list of coefficients.
function kernel_oracle(R, A, B, len)
    n = max(len, maximum(length, A; init=0), maximum(length, B; init=0), 1)
    return naive_series_product(R, kernel_coeffs(R, A, n), kernel_coeffs(R, B, n), len)
end

# Whether the W-polynomial P equals the list of coefficients `expected` modulo z^len (and has no
# terms beyond it).
function kernel_matches(R, P, expected, len)
    return all(length(q) <= len for q in P) && kernel_coeffs(R, P, len) == expected
end

# A random W-polynomial with components of length ≤ len; some components zero with probability
# pzero, all coefficients p^N - 1 with probability pmax (the largest possible products).
function rand_wpoly(rng, R, len; pzero=0.15, pmax=0.05)
    if rand(rng) < pmax
        return [R.Znz([R.Zn(R.pN - 1) for _ in 1:len]) for _ in 1:(R.f)]
    end
    return [rand(rng) < pzero ? zero(R.Znz) : rand_poly(rng, R, len) for _ in 1:(R.f)]
end

# A W-polynomial with coefficients in Z_p (only the first component nonzero).
function rand_zp_wpoly(rng, R, len)
    return [d == 1 ? rand_poly(rng, R, len) : zero(R.Znz) for d in 1:(R.f)]
end

# The multiplication methods available for residue degree f.
function kernel_methods(f)
    return if f == 1
        (:single,)
    elseif f <= 4
        (:schoolbook, :karatsuba, :kronecker)
    else
        (:schoolbook, :kronecker)
    end
end

# Galois rings for the tests: Conway presentations with p^N below and above 2^64, and lifts of
# irreducible polynomials with large coefficients (so that the coordinates of x^k, k ≥ f, are
# not ±1).
function kernel_rings()
    rings = S.GaloisRing[]
    for (p, f, N) in (
        (2, 1, 10),
        (3, 1, 50),
        (2, 2, 8),
        (2, 2, 70),
        (3, 2, 6),
        (5, 2, 30),
        (2, 3, 20),
        (3, 3, 45),
        (2, 4, 12),
        (2, 4, 70),
        (3, 4, 10),
        (5, 5, 6),
        (2, 5, 66),
    )
        push!(rings, galois_ring_and_qadic(p, f, N)[1])
    end
    push!(rings, S.GaloisRing(3, [2 + 3 * 5, 1 + 3 * 7, 1], 41))           # x² + x + 2 mod 3
    push!(rings, S.GaloisRing(2, [1 + 2 * 11, 1 + 2 * 3, 2 * 5, 2 * 7, 1], 67)) # x⁴ + x + 1 mod 2
    push!(rings, S.GaloisRing(3, [1 + 3 * 4, 2, 3 * 2, 1], 9))                # x³ + 2x + 1 mod 3
    return rings
end

@testset "kernels" begin
    rng = Xoshiro(271828)

    @testset "in-place FLINT primitives" begin
        @test S.FLINT_INPLACE == S.flint_inplace_available()
        @test S.FLINT_INPLACE          # with the pinned Nemo version
        for (p, N) in ((2, 10), (3, 50), (2, 70), (5, 3))
            R = S.GaloisRing(p, N)
            for _ in 1:25
                la, lb = rand(rng, 0:9), rand(rng, 0:9)
                a, b = rand_poly(rng, R, la), rand_poly(rng, R, lb)
                n = rand(rng, 0:(la + lb + 2))
                expected = mullow(a, b, n)
                for v in (Val(true), Val(false))
                    @test S._poly_mullow!(v, zero(R.Znz), a, b, n) == expected
                    # aliasing: the result overwrites an operand
                    a2 = deepcopy(a)
                    @test S._poly_mullow!(v, a2, a2, b, n) == expected
                    b2 = deepcopy(b)
                    @test S._poly_mullow!(v, b2, a, b2, n) == expected
                    a3 = deepcopy(a)
                    @test S._poly_mullow!(v, a3, a3, a3, n) == mullow(a, a, n)
                end
                @test S.poly_mullow!(zero(R.Znz), a, b, n) == expected
                # a reused output polynomial of a different length
                z = rand_poly(rng, R, 12)
                @test S.poly_mullow!(z, a, b, n) == expected
                @test S.poly_mullow!(z, b, a, n) == mullow(b, a, n)

                c = ZZ(rand(rng, 0:(R.pN - 1)))
                for cc in (c, one(ZZ), R.pN - 1, zero(ZZ), -c, R.pN + c)
                    expected = a + b * R.Zn(cc)
                    for v in (Val(true), Val(false))
                        @test S._poly_addmul!(v, deepcopy(a), b, cc) == expected
                    end
                    @test S.poly_addmul!(deepcopy(a), b, cc) == expected
                end
                s = rand(rng, 0:5)
                for v in (Val(true), Val(false))
                    @test S._poly_shift_left!(v, deepcopy(a), s) == shift_left(a, s)
                    @test S._poly_add_trunc!(v, zero(R.Znz), a, b, n) == truncate(a + b, n)
                    a2 = deepcopy(a)
                    @test S._poly_add_trunc!(v, a2, a2, b, n) == truncate(a + b, n)
                    @test S._poly_set_trunc!(v, rand_poly(rng, R, 3), a, n) ==
                        truncate(a, n)
                    zz = S._poly_lift_trunc!(v, zero(R.Zz), a, n)
                    @test zz == truncate(lift(R.Zz, a), n)
                    @test all(0 <= c < R.pN for c in coefficients(zz))
                    w = R.Zz([rand(rng, (-R.pN^2):(R.pN^2)) for _ in 1:la])
                    @test S._poly_reduce!(v, zero(R.Znz), w) ==
                        map_coefficients(R.Zn, w; parent=R.Znz)
                    # integer polynomials: products, shifts by powers of 2 (floor division)
                    y = R.Zz([rand(rng, -1000:1000) for _ in 1:lb])
                    @test S._zz_mullow!(v, zero(R.Zz), w, y, n) == mullow(w, y, n)
                    e = rand(rng, 0:70)
                    @test S._zz_mul_2exp!(v, zero(R.Zz), w, e) == w * ZZ(2)^e
                    @test S._zz_fdiv_2exp!(v, zero(R.Zz), w, e) ==
                        map_coefficients(c -> fdiv(c, ZZ(2)^e), w; parent=R.Zz)
                end
            end
        end
    end

    @testset "products against the naive product" begin
        for R in kernel_rings()
            f = R.f
            @testset "p=$(R.p) f=$f N=$(R.N) h̃=$(R.htilde)" begin
                ws = S.WScratch(R)      # reused across all products of this ring
                for trial in 1:24
                    la, lb = rand(rng, 0:12), rand(rng, 0:12)
                    A = rand_wpoly(rng, R, la)
                    B = if trial % 5 == 0
                        A                                    # squares
                    elseif trial % 7 == 0
                        rand_zp_wpoly(rng, R, lb)            # sparse operand
                    else
                        rand_wpoly(rng, R, lb)
                    end
                    # truncation lengths shorter than the inputs, and longer than the product
                    len = if rand(rng) < 0.3
                        rand(rng, 1:max(1, min(la, lb)))
                    else
                        rand(rng, 1:(la + lb + 2))
                    end
                    expected = kernel_oracle(R, A, B, len)
                    # the default method, then each method explicitly
                    @test kernel_matches(R, S.wpoly_mullow(R, A, B, len), expected, len)
                    for m in kernel_methods(f)
                        P = S.wpoly_mullow(R, A, B, len; method=m)
                        @test length(P) == f
                        ok = kernel_matches(R, P, expected, len)
                        ok || @info "product mismatch" R m la lb len
                        @test ok
                        # the same product in reused scratch space
                        Q = [zero(R.Znz) for _ in 1:(2 * f - 1)]
                        S.wpoly_mul!(R, Q, A, B, len, ws; method=m)
                        @test kernel_matches(R, Q[1:f], expected, len)
                    end
                    @test S.wpoly_mullow(R, A, B, 0) == S.wpoly_zero(R)
                    @test all(iszero, S.wpoly_mullow(R, S.wpoly_zero(R), B, len))
                end
                # the largest possible coefficients (all equal to p^N - 1)
                M = [R.Znz([R.Zn(R.pN - 1) for _ in 1:12]) for _ in 1:f]
                M2 = deepcopy(M)
                expected = kernel_oracle(R, M, M2, 23)
                for m in kernel_methods(f), (X, Y) in ((M, M), (M, M2))
                    @test kernel_matches(
                        R, S.wpoly_mullow(R, X, Y, 23; method=m), expected, 23
                    )
                end
            end
        end
    end

    @testset "products in x before the reduction modulo h̃" begin
        for R in kernel_rings()
            f = R.f
            f in 2:4 || continue
            ws = S.WScratch(R)
            for _ in 1:10
                la, lb = rand(rng, 0:10), rand(rng, 0:10)
                A, B = rand_wpoly(rng, R, la), rand_wpoly(rng, R, lb)
                len = rand(rng, 1:(la + lb + 1))
                P1 = [zero(R.Znz) for _ in 1:(2 * f - 1)]
                P2 = [zero(R.Znz) for _ in 1:(2 * f - 1)]
                S.wpoly_products_schoolbook!(R, P1, A, B, len, ws)
                S.wpoly_products_karatsuba!(R, P2, A, B, len, ws)
                @test P1 == P2
                C = [zero(R.Zz) for _ in 1:(2 * f - 1)]
                S.wpoly_products_kronecker!(R, C, A, B, len, ws)
                @test [map_coefficients(R.Zn, c; parent=R.Znz) for c in C] == P1
                @test all(all(>=(0), coefficients(c)) for c in C)
                # the naive products Σ_{d+e=k} A_d B_e
                for k in 1:(2 * f - 1)
                    naive = zero(R.Znz)
                    for d in 1:f, e in 1:f
                        d + e - 1 == k && (naive += mullow(A[d], B[e], len))
                    end
                    @test P1[k] == naive
                end
            end
        end
        R5 = galois_ring_and_qadic(2, 5, 8)[1]
        Z5 = S.wpoly_zero(R5)
        P5 = [zero(R5.Znz) for _ in 1:9]
        @test_throws ArgumentError S.wpoly_products_karatsuba!(
            R5, P5, Z5, Z5, 3, S.WScratch(R5)
        )
        @test_throws ArgumentError S.wpoly_mullow(R5, Z5, Z5, 3; method=:fft)
    end

    @testset "in-place accumulation with valuation shifts" begin
        for R in kernel_rings()
            f = R.f
            ws = S.WScratch(R)
            for _ in 1:12
                L = rand(rng, 1:14)
                vA, vB = rand(rng, 0:4), rand(rng, 0:4)
                A, B = rand_wpoly(rng, R, rand(rng, 0:10)),
                rand_wpoly(rng, R, rand(rng, 0:10))
                out = rand_wpoly(rng, R, L; pmax=0.0)
                out0, A0, B0 = deepcopy(out), deepcopy(A), deepcopy(B)
                # (z^vA A)(z^vB B) mod z^L as a coefficient list
                zA = [shift_left(a, vA) for a in A]
                zB = [shift_left(b, vB) for b in B]
                expected = kernel_oracle(R, zA, zB, L)
                S.wpoly_addmul!(R, out, A, vA, B, vB, L, ws)
                @test kernel_coeffs(R, out, L) == [
                    S.w_add(R, c, e) for (c, e) in zip(kernel_coeffs(R, out0, L), expected)
                ]
                @test kernel_matches(
                    R, S.wpoly_mullow_shifted(R, A, vA, B, vB, L), expected, L
                )
                if vA + vB >= L
                    @test S.wpoly_product!(R, ws, A, vA, B, vB, L) === nothing
                end
                # the operands are not modified
                @test A == A0 && B == B0
            end
            # doubling and addition in place
            A = rand_wpoly(rng, R, 6; pmax=0.0)
            B = rand_wpoly(rng, R, 9; pmax=0.0)
            A0 = deepcopy(A)
            @test S.wpoly_double!(deepcopy(A), f) == [a + a for a in A0]
            @test S.wpoly_add!(A, B) == [A0[d] + B[d] for d in 1:f]
        end
    end

    @testset "scalars and σ" begin
        for R in kernel_rings()
            f = R.f
            for _ in 1:6
                A = rand_wpoly(rng, R, rand(rng, 0:10))
                n = max(maximum(length, A; init=0), 1)
                cA = kernel_coeffs(R, A, n)
                for a in (rand_welem(rng, R), S.w_one(R), S.w_int(R, -1), S.w_zero(R))
                    @test kernel_coeffs(R, S.wpoly_scale(R, A, a), n) ==
                        [S.w_mul(R, a, c) for c in cA]
                    out = rand_wpoly(rng, R, n; pmax=0.0)
                    co = kernel_coeffs(R, out, n)
                    S.wpoly_addmul_scalar!(R, out, A, a)
                    @test kernel_coeffs(R, out, n) ==
                        [S.w_add(R, c, S.w_mul(R, a, x)) for (c, x) in zip(co, cA)]
                end
                @test kernel_coeffs(R, S.wpoly_sigma(R, A), n) ==
                    [S.w_frobenius(R, c) for c in cA]
                a = rand_welem(rng, R)
                @test S.scalar_matrix(R, a) == S.mult_matrix(R, a)
                @test S.scalar_matrix(R, S.w_int(R, -1)) == S.mult_matrix(R, S.w_int(R, -1))
            end
        end
    end

    @testset "choice of method" begin
        R1 = S.GaloisRing(2, 30)
        A1 = [rand_poly(rng, R1, 8)]
        @test S.wpoly_mullow_method(R1, A1, A1, 8) == :single
        # full operands: Karatsuba for f = 2; Kronecker for f = 5
        R2 = galois_ring_and_qadic(2, 2, 60)[1]
        A2 = rand_wpoly(rng, R2, 16; pzero=0.0, pmax=0.0)
        @test S.wpoly_mullow_method(R2, A2, A2, 16) == :karatsuba
        # operands with coefficients in Z_p: schoolbook (a single polynomial product)
        Z2 = rand_zp_wpoly(rng, R2, 16)
        @test S.wpoly_mullow_method(R2, A2, Z2, 16) == :schoolbook
        @test S.wpoly_mullow_method(R2, Z2, Z2, 16) == :schoolbook
        # (Kronecker packing needs the in-place FLINT calls)
        kron(fallback) = S.FLINT_INPLACE ? :kronecker : fallback
        R5 = galois_ring_and_qadic(2, 5, 70)[1]
        A5 = rand_wpoly(rng, R5, 16; pzero=0.0, pmax=0.0)
        @test S.wpoly_mullow_method(R5, A5, A5, 16) == kron(:schoolbook)
        @test S.wpoly_mullow_method(R5, A5, A5, 16; ws=S.WScratch(R5)) == kron(:schoolbook)
        @test S.wpoly_mullow_method(R5, rand_zp_wpoly(rng, R5, 16), A5, 16) == :schoolbook
        # single-word coefficients (p^N < 2^62): no Kronecker packing
        R5s = galois_ring_and_qadic(2, 5, 48)[1]
        A5s = rand_wpoly(rng, R5s, 16; pzero=0.0, pmax=0.0)
        @test S.wpoly_mullow_method(R5s, A5s, A5s, 16) == :schoolbook
        # f = 4: Kronecker packing inside the benchmarked range of bits(p^N) and lengths, with a
        # reused scratch space; the range starts later when the reduction modulo h̃ needs only
        # additions and subtractions (p = 2 and a Conway polynomial)
        for (p, N, expected) in
            ((3, 60, kron(:karatsuba)), (2, 100, :karatsuba), (2, 150, :karatsuba))
            R4 = galois_ring_and_qadic(p, 4, N)[1]
            A4 = rand_wpoly(rng, R4, 200; pzero=0.0, pmax=0.0)
            ws4 = S.WScratch(R4)
            @test S._unit_reduction!(ws4, R4) == (p == 2)
            @test S.wpoly_mullow_method(R4, A4, A4, 32; ws=ws4) == expected
            @test S.wpoly_mullow_method(R4, A4, A4, 32) == :karatsuba
            @test S.wpoly_mullow_method(R4, A4, A4, 200; ws=ws4) == :karatsuba
        end
        # a lift of x⁴ + x + 1 with large coefficients: no unit reduction for p = 2
        Rh = S.GaloisRing(2, [1 + 2 * 11, 1 + 2 * 3, 2 * 5, 2 * 7, 1], 80)
        @test !S._unit_reduction!(S.WScratch(Rh), Rh)
        Ah = rand_wpoly(rng, Rh, 16; pzero=0.0, pmax=0.0)
        @test S.wpoly_mullow_method(Rh, Ah, Ah, 16; ws=S.WScratch(Rh)) == kron(:karatsuba)
        # f = 3: Karatsuba
        R3 = galois_ring_and_qadic(3, 3, 100)[1]
        A3 = rand_wpoly(rng, R3, 32; pzero=0.0, pmax=0.0)
        @test S.wpoly_mullow_method(R3, A3, A3, 32; ws=S.WScratch(R3)) == :karatsuba
        R4s = galois_ring_and_qadic(2, 4, 20)[1]
        A4s = rand_wpoly(rng, R4s, 32; pzero=0.0, pmax=0.0)
        @test S.wpoly_mullow_method(R4s, A4s, A4s, 32; ws=S.WScratch(R4s)) == :karatsuba
        R4 = galois_ring_and_qadic(3, 4, 60)[1]
        # the slot width of Kronecker packing bounds the largest possible coefficient
        for R in (R2, R4, R5)
            m = 20
            bound = R.f * m * (R.pN - 1)^2
            @test nbits(bound) < S.kronecker_bits(R, m)
        end
    end
end
