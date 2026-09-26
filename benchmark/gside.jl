# Benchmarks of the g-side (the envelope of O_K over W(F_q)[[z_0, z_1]], Steps 4 and 5):
#
#   julia --project=benchmark -t 8 benchmark/gside.jl                 # all sections
#   julia --project=benchmark -t 1 benchmark/gside.jl model step4     # some sections
#
# Sections:
# - model: products in the g-envelope (general, squares, with the sparse unit w(u), of elements of
#   high F-weight) and φ, against `Syntomic.g_mul_cost` (the cost model of the powers w(v^i));
# - step4: w(v) with the F-weight ladder against K iterations modulo F^{≥L}, and w(v^i) by each
#   method of `Syntomic.bk_unit_v_power` (each on a new envelope, so that the normal forms are
#   computed as in an actual run);
# - step5: ∇_{O_K} by the forward chain and by the transposed chain;
# - sweeps: the shared data of a sweep (`Syntomic.shared_nabla_OK`): w(v^i) by each method of
#   `Syntomic.bk_unit_v_powers`, and ∇_{O_K}(i) modulo F^{≥n_max·i} against modulo F^{≥L_max};
# - table: Steps 3–5 with the table of normal forms computed level by level when the envelope is
#   built (`Syntomic.g_fill_normal_forms!`) against on first use, threaded and serial, and the
#   time of the levels alone.
# Times are minima over repetitions, in seconds unless stated otherwise.
using Syntomic, Nemo, Hecke
using Printf

const S = Syntomic

# The g-envelope of X in weight i at the working precision of the :exponent policy.
function genv(X, i; threaded=true, eager=threaded && S.threads_available())
    Xc = S.computing_ring(X, true)
    N = S.working_precision(Xc, i; precision_policy=:exponent, descent=false)
    prm = S.SyntomicParams(Xc.p, Xc.f, Xc.n, i)
    R = S.GaloisRing(Xc.p, Xc.htilde, N)
    ed = S.EisensteinData(R, Xc, prm.L)
    lam = S.LambdaData(ed, max(prm.nf, prm.ng, 2) - 2)
    return S.GEnvelope(prm, R, ed, lam; threaded=threaded, eager=eager)
end

# The g-envelope of a sweep over ns × is (F-precision n_max·i_max, the shared precision).
function sweep_genv(X, ns, is; threaded=true)
    Xc = S.computing_ring(S.with_power(X, maximum(ns)), true)
    Ns = S.sweep_precisions(Xc, ns, is; threaded=threaded)
    Lmax = maximum(ns) * maximum(is)
    N = maximum(values(Ns)) + S.num_g_variables(X.p, Lmax) + 1
    prm = S.SyntomicParams(Xc.p, Xc.f, Lmax, 1)
    R = S.GaloisRing(Xc.p, Xc.htilde, N)
    ed = S.EisensteinData(R, Xc, Lmax)
    lam = S.LambdaData(ed, max(prm.ng, 2) - 2)
    return S.GEnvelope(prm, R, ed, lam; threaded=threaded)
end

# Minimum over `reps` runs of f() (after one run to compile and fill caches).
function tmin(f; reps=3)
    f()
    return minimum(@elapsed(f()) for _ in 1:reps)
end

# Minimum over `reps` runs of f(env) on a new envelope made by mk() each time.
function tfresh(mk, f; reps=3)
    return minimum(1:reps) do _
        env = mk()
        @elapsed f(env)
    end
end

function ramified_ring()
    K, a = qadic_field(2, 2; precision=10, cached=false)
    Kz, z = polynomial_ring(K, :z; cached=false)
    return finite_chain_ring(z^2 + 2a * z + 2a, 4)
end

const SINGLE = [
    ("Z/4, i=30", () -> zpn(2, 2), 30),
    ("Z/4, i=40", () -> zpn(2, 2), 40),
    ("Z/9, i=24", () -> zpn(3, 2), 24),
    ("Z/25, i=20", () -> zpn(5, 2), 20),
    ("Q_4(√(-2a)), n=4, i=15", ramified_ring, 15),
]

const SWEEPS = [
    ("Z/4..Z/32, i=1:12", () -> zpn(2, 2), 2:5, 1:12),
    ("Z/9, Z/27, i=1:24", () -> zpn(3, 2), 2:3, 1:24),
    ("Z/25, Z/125, i=1:20", () -> zpn(5, 2), 2:3, 1:20),
]

function section_model()
    println("\n== model: measured time / time of a general product, against g_mul_cost")
    @printf("%-24s %-28s %9s %9s\n", "case", "product", "measured", "model")
    for (name, mk, i) in SINGLE
        env = genv(mk(), i; threaded=false)
        L, p = env.L, env.R.p
        wu = S.bk_unit_u(env)
        wv = S.bk_unit_v(env, wu)
        v2 = S.g_mul(env, wv, wv)
        wt, s = S._unit_shape(env, wu, L)
        base = tmin(() -> S.g_mul(env, v2, wv))
        B = S.binomial_powers(env, wv, div(L, 2))
        row(what, t, m) = @printf("%-24s %-28s %9.3f %9.3f\n", name, what, t / base, m)
        row("general, L=$L", base, 1.0)
        for T in (div(3 * L, 4), div(L, 2))
            row("general, T=$T", tmin(() -> S.g_mul(env, v2, wv; L=T)), (T / L)^4)
        end
        c(; kw...) = S.g_mul_cost(p, L; kw...) / S.g_mul_cost(p, L)
        row("square", tmin(() -> S.g_mul(env, wv, wv)), c(; square=true))
        row("with w(u) ($s components)", tmin(() -> S.g_mul(env, v2, wu)), c(; support=s))
        y = B.ypow
        for k in unique([2, div(length(y), 2), length(y)])
            k >= 2 || continue
            t = tmin(() -> S.g_mul(env, y[k - 1], y[1]))
            row("y^$(k - 1)·y, y = w(v) - 1", t, c(; w=k * B.wt))
        end
        row(
            "φ",
            tmin(() -> S.g_frobenius(env, wv)),
            S.g_frobenius_cost(p, L) / S.g_mul_cost(p, L),
        )
    end
end

# w(v) by K iterations modulo F^{≥L} (without the ladder).
function bk_unit_v_full(env, wu)
    x = wu
    for _ in 1:S.bk_unit_iterations(env.R.p, env.L)
        x = S.g_mul(env, wu, S.g_frobenius(env, x))
    end
    return x
end

function section_step4()
    println("\n== step4 (threads: $(Threads.nthreads()))")
    @printf(
        "%-24s %-8s %9s %9s | %11s %9s %9s %9s  %s\n",
        "case",
        "threaded",
        "w(v) full",
        "ladder",
        "fixed point",
        "binary",
        "binomial",
        "old",
        "auto"
    )
    for (name, mk, i) in SINGLE, threaded in (true, false)
        X = mk()
        mkenv() = genv(X, i; threaded=threaded)
        # w(v) after w(u) (which fills part of the caches)
        t_full = tfresh(mkenv, env -> bk_unit_v_full(env, S.bk_unit_u(env)))
        t_ladder = tfresh(mkenv, env -> S.bk_unit_v(env, S.bk_unit_u(env)))
        # w(v^i) from w(u), by each method, and as on main (w(v) by K iterations, then binary)
        tm(m) = tfresh(mkenv, env -> S.bk_unit_v_power(env, S.bk_unit_u(env), i; method=m))
        t_old = tfresh(mkenv, env -> S.g_pow(env, bk_unit_v_full(env, S.bk_unit_u(env)), i))
        env = mkenv()
        wu = S.bk_unit_u(env)
        wt, s = S._unit_shape(env, wu, env.L)
        auto = S.bk_unit_v_power_method(env.R.p, i, env.L, s, wt)
        @printf(
            "%-24s %-8s %9.3f %9.3f | %11.3f %9.3f %9.3f %9.3f  %s\n",
            name,
            threaded,
            t_full,
            t_ladder,
            tm(:fixed_point),
            tm(:binary),
            tm(:binomial),
            t_old,
            auto
        )
    end
end

function section_step5()
    println("\n== step5: ∇_{O_K} (threads: $(Threads.nthreads()))")
    @printf(
        "%-24s %-8s %9s %11s %9s\n", "case", "threaded", "forward", "transposed", "(chain)"
    )
    for (name, mk, i) in SINGLE, threaded in (true, false)
        env = genv(mk(), i; threaded=threaded)
        vi = S.bk_unit_v_power(env, S.bk_unit_u(env), i)
        t_fwd = tmin(() -> S.nabla_OK_matrix(env, vi; threaded=threaded, method=:forward))
        t_trn = tmin(() -> S.nabla_OK_matrix(env, vi; threaded=threaded))
        t_chain = tmin(() -> S.theta_chain(env; threaded=threaded))
        @printf("%-24s %-8s %9.3f %11.3f %9.3f\n", name, threaded, t_fwd, t_trn, t_chain)
    end
end

function section_sweeps()
    println("\n== sweeps: shared data (threads: $(Threads.nthreads()))")
    methods = (:fixed_point, :incremental, :binary, :binomial)
    @printf("%-22s %-8s", "case", "threaded")
    foreach(m -> @printf(" %11s", m), methods)
    @printf(" %-11s | %9s %9s %9s\n", " auto", "∇ at L_i", "at L_max", "forward")
    for (name, mk, ns, is) in SWEEPS, threaded in (true, false)
        X = mk()
        mkenv() = sweep_genv(X, ns, is; threaded=threaded)
        nmax = maximum(ns)
        Ls = Dict(i => nmax * i for i in is)
        ts = [
            tfresh(
                mkenv,
                env -> S.bk_unit_v_powers(env, S.bk_unit_u(env), is, Ls; method=m);
                reps=2,
            ) for m in methods
        ]
        env = mkenv()
        Lmax = env.L
        wu = S.bk_unit_u(env)
        wt, s = S._unit_shape(env, wu, Lmax)
        auto = S.bk_unit_v_powers_method(env.R.p, collect(is), Ls, s, wt)
        P = S.bk_unit_v_powers(env, wu, is, Ls)
        Pmax = S.bk_unit_v_powers(env, wu, is, Dict(i => Lmax for i in is))
        chain = S.theta_chain(env; threaded=threaded)
        function nab(Q; kw...)
            return foreach(i -> S.nabla_OK_matrix(env, Q[i]; threaded=threaded, kw...), is)
        end
        t_trunc = tmin(() -> nab(P; chain=chain); reps=2)
        t_max = tmin(() -> nab(Pmax; chain=chain); reps=2)
        t_fwd = tmin(() -> nab(Pmax; method=:forward); reps=1)
        @printf("%-22s %-8s", name, threaded)
        foreach(t -> @printf(" %11.3f", t), ts)
        @printf(" %-11s | %9.3f %9.3f %9.3f\n", " $auto", t_trunc, t_max, t_fwd)
    end
end

# The time of `g_fill_normal_forms!` for all levels, on the relations of env with empty tables.
function fill_time(env; threaded)
    empty = S.GEnvelope(
        env.prm,
        env.L,
        env.ng,
        env.R,
        env.ed,
        env.lam,
        env.Rp,
        env.rel,
        env.phi,
        S.OnceCache{Vector{Int},S.GElem}(),
        S.OnceCache{Int,S.GElem}(),
        env.digits,
        threaded,
        S.ScratchPool(env.R),
    )
    return @elapsed S.g_fill_normal_forms!(empty, env.ng - 2; threaded=threaded)
end

# Steps 3–5 for the ring X in weight i: the envelope, w(v^i) and ∇_{O_K}.
function steps_3_to_5(X, i; threaded, eager)
    env = genv(X, i; threaded=threaded, eager=eager)
    vi = S.bk_unit_v_power(env, S.bk_unit_u(env), i)
    S.nabla_OK_matrix(env, vi; threaded=threaded)
    return env
end

function section_table()
    println("\n== table: Steps 3-5 for O_K (threads: $(Threads.nthreads()))")
    @printf(
        "%-24s %-8s %11s %11s %11s %8s\n",
        "case",
        "threaded",
        "level-wise",
        "first use",
        "levels only",
        "entries"
    )
    for (name, mk, i) in SINGLE, threaded in (true, false)
        X = mk()
        function t(eager)
            return minimum(
                @elapsed(steps_3_to_5(X, i; threaded=threaded, eager=eager)) for _ in 1:3
            )
        end
        t_eager, t_lazy = t(true), t(false)
        env = genv(X, i; threaded=threaded)
        t_fill = minimum(fill_time(env; threaded=threaded) for _ in 1:3)
        @printf(
            "%-24s %-8s %11.3f %11.3f %11.3f %8d\n",
            name,
            threaded,
            t_eager,
            t_lazy,
            t_fill,
            length(env.nf_cache)
        )
    end
end

const SECTIONS = Dict(
    "model" => section_model,
    "step4" => section_step4,
    "step5" => section_step5,
    "sweeps" => section_sweeps,
    "table" => section_table,
)

println("threads: ", Threads.nthreads())
for name in (isempty(ARGS) ? ["model", "step4", "step5", "sweeps", "table"] : ARGS)
    SECTIONS[name]()
end
