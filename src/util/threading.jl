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
    unwrap_task_exception(err) -> Exception

The exception that caused a failed task: `TaskFailedException` and `CompositeException` are
unwrapped, so that errors such as [`PrecisionExhausted`](@ref) are rethrown with the same type
whether or not the computation ran on several threads.
"""
function unwrap_task_exception(err)
    while true
        if err isa TaskFailedException
            err = err.task.exception
        elseif err isa CompositeException && !isempty(err.exceptions)
            err = first(err.exceptions)
        else
            return err
        end
    end
end

"""
    tmap(f, xs; threaded=true, chunksize=1) -> Vector

Apply `f` to every element of `xs` (collected into a vector), in parallel when `threaded` is true
and more than one thread is available. Results are returned in the order of `xs`; exceptions
thrown by `f` are rethrown unwrapped (see [`unwrap_task_exception`](@ref)).
"""
function tmap(f, xs; threaded::Bool=true, chunksize::Integer=1)
    v = collect(xs)
    if threaded && threads_available() && length(v) > 1
        try
            return OhMyThreads.tmap(
                f, v; scheduler=OhMyThreads.DynamicScheduler(; chunksize=chunksize)
            )
        catch err
            throw(unwrap_task_exception(err))
        end
    else
        return map(f, v)
    end
end

"""
    tforeach(f, xs; threaded=true, chunksize=1)

Call `f` on every element of `xs` for its side effects, in parallel when `threaded` is true.
`f` must only write to disjoint locations. Exceptions are rethrown unwrapped.
"""
function tforeach(f, xs; threaded::Bool=true, chunksize::Integer=1)
    v = collect(xs)
    if threaded && threads_available() && length(v) > 1
        try
            OhMyThreads.tforeach(
                f, v; scheduler=OhMyThreads.DynamicScheduler(; chunksize=chunksize)
            )
        catch err
            throw(unwrap_task_exception(err))
        end
    else
        foreach(f, v)
    end
    return nothing
end

"""
    Spawned(task)

A task started by [`spawn_if`](@ref); `fetch` rethrows the task's exception unwrapped.
"""
struct Spawned
    task::Task
end

function Base.fetch(s::Spawned)
    try
        return fetch(s.task)
    catch err
        throw(unwrap_task_exception(err))
    end
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
    spawn_if(threaded, f) -> Spawned or Computed

Start `f()` concurrently when `threaded` is true (returning a [`Spawned`](@ref) task), or run it
immediately (returning a [`Computed`](@ref)). Use `fetch` on the result in both cases; errors are
rethrown with their original type.
"""
function spawn_if(threaded::Bool, f)
    if threaded && threads_available()
        return Spawned(Threads.@spawn f())
    else
        return Computed(f())
    end
end
