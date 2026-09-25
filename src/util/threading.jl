# Thin threading layer on top of OhMyThreads.
#
# All parallel loops in the package go through `tmap`/`tforeach` so that
# * results are returned in input order (bit-identical output for any number of threads:
#   all arithmetic is exact modular arithmetic),
# * every parallel loop can be switched off with `threaded = false` (used by the serial
#   reference pipeline and by the determinism tests),
# * scheduling is dynamic with one item per task by default (the work per item is uneven),
# * FLINT itself stays single-threaded.

"""
    threads_available() -> Bool

Whether more than one Julia thread is available.
"""
threads_available() = Threads.nthreads() > 1

"""
    tmap(f, xs; threaded=true, chunksize=1) -> Vector

Apply `f` to every element of `xs`, in parallel when `threaded` is true and more than one thread
is available. Results are returned in the order of `xs`.
"""
function tmap(f, xs; threaded::Bool=true, chunksize::Integer=1)
    if threaded && threads_available() && length(xs) > 1
        return OhMyThreads.tmap(
            f, xs; scheduler=OhMyThreads.DynamicScheduler(; chunksize=chunksize)
        )
    else
        return map(f, xs)
    end
end

"""
    tforeach(f, xs; threaded=true, chunksize=1)

Call `f` on every element of `xs` for its side effects, in parallel when `threaded` is true.
`f` must only write to disjoint locations.
"""
function tforeach(f, xs; threaded::Bool=true, chunksize::Integer=1)
    if threaded && threads_available() && length(xs) > 1
        OhMyThreads.tforeach(
            f, xs; scheduler=OhMyThreads.DynamicScheduler(; chunksize=chunksize)
        )
    else
        foreach(f, xs)
    end
    return nothing
end

"""
    Computed(value)

An already-computed value that supports `fetch`, returned by [`spawn_if`](@ref) when running
serially.
"""
struct Computed{T}
    value::T
end

Base.fetch(c::Computed) = c.value

"""
    spawn_if(threaded, f) -> Task or Computed

Start `f()` concurrently when `threaded` is true (returning a `Task`), or run it immediately
(returning a [`Computed`](@ref)). Use `fetch` on the result in both cases.
"""
function spawn_if(threaded::Bool, f)
    if threaded && threads_available()
        return Threads.@spawn f()
    else
        return Computed(f())
    end
end
