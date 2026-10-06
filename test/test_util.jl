@testset "utilities: memo tables, plain data, Step 4" begin
    @testset "OnceCache: every key is computed once" begin
        cache = S.OnceCache{Int,Int}()
        nkeys = 24
        calls = [Threads.Atomic{Int}(0) for _ in 1:nkeys]
        rng = Xoshiro(314)
        orders = [shuffle(rng, 1:nkeys) for _ in 1:48]
        tasks = map(orders) do order
            Threads.@spawn map(order) do k
                S.get_or_compute!(cache, k) do
                    Threads.atomic_add!(calls[k], 1)
                    sleep(0.001)                       # let other tasks ask for k meanwhile
                    k^2 + 1
                end
            end
        end
        results = fetch.(tasks)
        @test all(c[] == 1 for c in calls)
        @test all(r == order .^ 2 .+ 1 for (r, order) in zip(results, orders))
        @test length(cache) == nkeys
        # a stored value is returned without calling the function again
        @test S.get_or_compute!(() -> error("recomputed"), cache, 3) == 10
    end

    @testset "OnceCache: waiters receive the owner's value" begin
        cache = S.OnceCache{Symbol,Vector{Int}}()
        started = Base.Event()
        release = Base.Event()
        calls = Threads.Atomic{Int}(0)
        owner = Threads.@spawn S.get_or_compute!(cache, :key) do
            Threads.atomic_add!(calls, 1)
            notify(started)
            wait(release)
            [1, 2, 3]
        end
        wait(started)
        waiters = map(1:8) do _
            Threads.@spawn S.get_or_compute!(cache, :key) do
                Threads.atomic_add!(calls, 1)
                Int[]
            end
        end
        sleep(0.05)
        @test !any(istaskdone, waiters)                # blocked until the owner finishes
        notify(release)
        value = fetch(owner)
        @test value == [1, 2, 3]
        @test all(fetch(w) === value for w in waiters)   # the owner's object, not a copy
        @test calls[] == 1
    end

    @testset "OnceCache: a failure reaches the owner and every waiter" begin
        # The exceptions thrown by `get_or_compute!` in the owner and in the tasks waiting.
        function failing_owner(cache, key, err)
            started = Base.Event()
            release = Base.Event()
            function caught(g)
                try
                    g()
                catch e
                    return e
                end
                return nothing
            end
            owner = Threads.@spawn caught() do
                S.get_or_compute!(cache, key) do
                    notify(started)
                    wait(release)
                    throw(err)
                end
            end
            wait(started)
            waiters = map(1:6) do _
                Threads.@spawn caught(() -> S.get_or_compute!(() -> 0, cache, key))
            end
            notify(release)
            return fetch(owner), fetch.(waiters)
        end
        cache = S.OnceCache{Int,Int}()
        # an exception of the package's interface is rethrown unchanged
        owner_err, waiter_errs = failing_owner(cache, 1, ArgumentError("owner failed"))
        @test owner_err == ArgumentError("owner failed")
        @test all(e == ArgumentError("owner failed") for e in waiter_errs)
        # any other exception reaches the waiters wrapped with the owner's backtrace
        owner_err, waiter_errs = failing_owner(cache, 2, ErrorException("a bug"))
        @test owner_err == ErrorException("a bug")
        @test all(e isa CapturedException && e.ex == owner_err for e in waiter_errs)
        # the failed keys keep failing in the same way, without calling the function again
        calls = Ref(0)
        for _ in 1:3
            @test_throws ArgumentError S.get_or_compute!(() -> (calls[] += 1), cache, 1)
            err = try
                S.get_or_compute!(() -> (calls[] += 1), cache, 2)
            catch e
                e
            end
            @test err isa CapturedException && err.ex == ErrorException("a bug")
        end
        @test calls[] == 0
        # other keys are not affected
        @test S.get_or_compute!(() -> 5, cache, 3) == 5
        @test length(cache) == 3
    end

    @testset "OnceCache: values computed elsewhere" begin
        cache = S.OnceCache{Int,Vector{Int}}()
        @test !haskey(cache, 1)
        v = [1, 2]
        @test S.set_computed!(cache, 1, v) === v
        @test haskey(cache, 1)
        # stored values are not replaced, and are returned without computing
        @test S.set_computed!(cache, 1, [3]) === v
        @test S.get_or_compute!(() -> error("recomputed"), cache, 1) === v
        # a value being computed is waited for
        started = Base.Event()
        release = Base.Event()
        owner = Threads.@spawn S.get_or_compute!(cache, 2) do
            notify(started)
            wait(release)
            [4]
        end
        wait(started)
        @test haskey(cache, 2)
        other = Threads.@spawn S.set_computed!(cache, 2, [5])
        notify(release)
        @test fetch(owner) == [4]
        @test fetch(other) == [4]
        @test length(cache) == 2
    end

    @testset "plain data" begin
        rng = Xoshiro(2718)
        h = S.default_htilde(3, 2)
        R = S.GaloisRing(3, h, 30)
        A = [rand_welem(rng, R) for _ in 1:4, _ in 1:3]
        W = S.WMatrix(R, A, 25)
        P = S.to_plain(W)
        @test P isa S.PlainWMatrix
        @test (P.p, P.htilde, P.prec) == (3, h, 25)
        @test size(P) == (4, 3) && size(P, 1) == 4 && size(P, 2) == 3
        @test eltype(P.entries) == Vector{BigInt}
        # round trip
        W2 = S.from_plain(R, P)
        @test W2.R === R && W2.A == W.A && W2.prec == W.prec
        @test S.to_plain(W2).entries == P.entries
        # over a ring of smaller precision N: entries reduced modulo p^N, precision capped
        R10 = S.GaloisRing(3, h, 10)
        W10 = S.from_plain(R10, P)
        @test W10.prec == 10
        @test W10.A == [S.w_from_coordinates(R10, P.entries[i, j]) for i in 1:4, j in 1:3]
        @test S.from_plain(S.GaloisRing(3, h, 40), P).prec == 25
        # top-left blocks
        T = S.plain_top_left(P, 2, 3)
        @test size(T) == (2, 3)
        @test (T.p, T.htilde, T.prec) == (P.p, P.htilde, P.prec)
        @test S.from_plain(R, T).A == W.A[1:2, 1:3]
        @test size(S.plain_top_left(P, 0, 0)) == (0, 0)
        # a ring with another p or another presentation of W(F_q) is rejected
        @test_throws ArgumentError S.from_plain(S.GaloisRing(2, [1, 1, 1], 30), P)
        @test_throws ArgumentError S.from_plain(S.GaloisRing(3, [0, 1], 30), P)
        @test_throws ArgumentError S.from_plain(S.GaloisRing(3, [5, 2, 1], 30), P)  # h̃ + 3
        @test_throws ArgumentError S.from_plain(S.GaloisRing(3, [2, 0, 0, 1], 30), P)
        # ∇_OK on F^{[1,L-1]} is the top-left block of ∇_OK on F^{[1,L'-1]}, L ≤ L', which
        # is how sweeps use `plain_top_left` ([AKN24, Remark 4.67, p. 46])
        for (X, i) in ((zpn(3, 2), 2), (truncated_witt_vectors(2, 2, 2), 2))
            N = S.apriori_working_precision(X.p, 3, i)
            small = S.to_plain(S.syntomic_wmatrices(S.with_power(X, 2), i, N).nablaOK)
            large = S.to_plain(S.syntomic_wmatrices(S.with_power(X, 3), i, N).nablaOK)
            top = S.plain_top_left(large, size(small)...)
            m = big(X.p)^min(small.prec, top.prec)
            @test all(
                all(iszero, mod.(a .- b, m)) for (a, b) in zip(small.entries, top.entries)
            )
        end
    end

    @testset "Step 4: the unit w(v) ([AKN24, §7.1, Step 4, pp. 55–56])" begin
        # K = max(0, ⌈log_p L⌉ - 2) iterations
        iterations_p2 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3]
        @test [S.bk_unit_iterations(2, L) for L in 1:17] == iterations_p2
        iterations_p3 = [0, 1, 1, 2, 2, 3]
        @test [S.bk_unit_iterations(3, L) for L in (9, 10, 27, 28, 81, 82)] == iterations_p3
        bad = Tuple{Int,Int}[]
        for p in (2, 3, 5, 7), L in 1:500
            clog = 0                                   # ⌈log_p L⌉
            while big(p)^clog < L
                clog += 1
            end
            K = S.bk_unit_iterations(p, L)
            # x_K = ∏_{r ≤ K} φ^r(w(u)) agrees with w(v) modulo F^{≥p^{K+2}} ⊆ F^{≥L}
            ok = K == max(0, clog - 2) && big(p)^(K + 2) >= L
            ok || push!(bad, (p, L))
        end
        @test isempty(bad)

        # x ∈ F^{≥W}: the g^{μ(t)}-coefficient of x is divisible by z_0^{W-t}
        function in_filtration(env, x, W)
            m = ZZ(env.R.p)^x.prec
            for t in 0:(env.L - 1), c in x.c[t + 1], k in 0:(min(W, env.L) - t - 1)
                iszero(mod(lift(coeff(c, k)), m)) || return false
            end
            return true
        end
        K9, a = qadic_field(3, 2; precision=40, cached=false)
        K9z, z = polynomial_ring(K9, :z; cached=false)
        for (X, i) in (
            (zpn(2, 3), 6),                                 # L = 18: K = 3
            (zpn(3, 3), 4),                                 # L = 12: K = 1
            (zpn(5, 2), 7),                                 # L = 14: K = 0
            (truncated_witt_vectors(2, 2, 3), 3),           # L = 9: K = 2
            (finite_chain_ring(z^2 + 3a * z + 3a, 3), 4),   # W(F_9)-arithmetic, L = 12
        )
            prm = S.SyntomicParams(X.p, X.f, X.n, i)
            p, L = prm.p, prm.L
            N = S.apriori_working_precision(X.p, X.n, i)
            R = S.GaloisRing(X.p, X.htilde, N)
            ed = S.EisensteinData(R, X, L)
            lam = S.LambdaData(ed, max(prm.nf, prm.ng, 2) - 2)
            genv = S.GEnvelope(prm, R, ed, lam)
            wu = S.bk_unit_u(genv)
            K = S.bk_unit_iterations(p, L)
            # w(u) - 1 = φ(Q)·φ_1(g_0) ∈ F^{≥p}
            @test in_filtration(genv, S.g_sub(genv, wu, S.g_one(genv)), p)
            # x_k = w(u)·φ(x_{k-1}), and x_k - x_{k-1} ∈ F^{≥p^{k+1}}
            xs = [wu]
            for _ in 1:(K + 1)
                push!(xs, S.g_mul(genv, wu, S.g_frobenius(genv, xs[end])))
            end
            for k in 1:(K + 1)
                @test in_filtration(genv, S.g_sub(genv, xs[k + 1], xs[k]), p^(k + 1))
            end
            # w(v) = x_K, and it is a fixed point of x ↦ w(u)·φ(x) modulo F^{≥L}
            wv = S.bk_unit_v(genv, wu)
            @test S.g_isequal(genv, wv, xs[K + 1])
            @test S.g_isequal(genv, wv, S.g_mul(genv, wu, S.g_frobenius(genv, wv)))
            @test S.g_isequal(genv, xs[K + 2], xs[K + 1])
        end
    end
end
