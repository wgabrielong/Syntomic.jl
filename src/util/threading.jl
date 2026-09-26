# Thin threading layer on Julia's task scheduler.
#
# All parallel work in the package goes through `tmap`, `tforeach` and `concurrently`, so that
# * results are returned in input order (bit-identical output for any number of threads:
#   all arithmetic is exact modular arithmetic),
# * every parallel loop can be switched off with `threaded = false` (used by the serial
#   reference pipeline and by the determinism tests),
# * scheduling is dynamic: tasks take the next chunk of items from a shared counter (the work
#   per item is uneven),
# * after a failure no new item is started, and every running task is waited for before the
#   exception is rethrown, so no computation outlives the call that started it,
# * FLINT itself stays single-threaded.

"""
    threads_available() -> Bool

Whether more than one Julia thread is available.
"""
threads_available() = Threads.nthreads() > 1

# The exceptions of the package's interface: raised inside a task, they are rethrown unwrapped,
# with the same type as in a serial computation. Other exceptions (bugs) are rethrown wrapped,
# keeping the backtrace of the task in which they occurred.
const InterfaceException = Union{
    PrecisionExhausted,CertificateError,ArgumentError,DimensionMismatch,InterruptException
}

"""
    unwrap_task_exception(err) -> Exception

The exception that caused `err`: `TaskFailedException`, `CompositeException`,
`CapturedException` and Distributed's `RemoteException` are unwrapped.
"""
function unwrap_task_exception(err)
    while true
        if err isa TaskFailedException
            err = err.task.exception
        elseif err isa CompositeException && !isempty(err.exceptions)
            err = first(err.exceptions)
        elseif err isa CapturedException
            err = err.ex
        elseif err isa Distributed.RemoteException
            err = err.captured
        else
            return err
        end
    end
end

# The exception to rethrow for `err`, caught with backtrace `bt` inside a task: an interface
# exception unwrapped, anything else wrapped together with its backtrace.
function _task_exception(err, bt)
    inner = unwrap_task_exception(err)
    inner isa InterfaceException && return inner
    err isa Union{TaskFailedException,CapturedException,Distributed.RemoteException} &&
        return err
    return CapturedException(err, bt)
end

"""
    rethrow_unwrapped(err)

For use in a `catch err` block around parallel or distributed work: throws the interface
exception (such as [`PrecisionExhausted`](@ref)) wrapped in `err`, and rethrows `err` itself
otherwise.
"""
function rethrow_unwrapped(err)
    inner = unwrap_task_exception(err)
    inner isa InterfaceException && throw(inner)
    return rethrow()
end

# The time (in nanoseconds) after which a worker of `_run_chunks` yields between two chunks.
const YIELD_INTERVAL_NS = 1_000_000

# Run `body(k)` for k = 1:n on tasks that take chunks of `chunksize` consecutive indices from a
# shared counter. There is one task per chunk, so that a task blocked in `body` (e.g. waiting for a
# memoized value, see `OnceCache`) does not keep the other chunks from running. After a failure no
# new chunk is started; all tasks are waited for, and the exception of the smallest failing index
# is rethrown. (Chunks are taken in increasing order and a started chunk runs until its first
# failure, so this index does not depend on the scheduling.)
function _run_chunks(body, n::Int, chunksize::Int)
    cs = max(chunksize, 1)
    nchunks = cld(n, cs)
    next = Threads.Atomic{Int}(1)
    failed = Threads.Atomic{Bool}(false)
    errors = Vector{Any}(nothing, nchunks)
    function worker()
        last_yield = time_ns()
        while !failed[]
            c = Threads.atomic_add!(next, 1)
            c > nchunks && return nothing
            for k in ((c - 1) * cs + 1):min(c * cs, n)
                try
                    body(k)
                catch err
                    errors[c] = _task_exception(err, catch_backtrace())
                    failed[] = true
                    break
                end
            end
            # let other tasks on this thread run between chunks (at most every millisecond), so
            # that the waiting caller can act on an interrupt while the chunks run
            if time_ns() - last_yield > YIELD_INTERVAL_NS
                yield()
                last_yield = time_ns()
            end
        end
        return nothing
    end
    tasks = [Threads.@spawn(worker()) for _ in 1:nchunks]
    try
        foreach(wait, tasks)
    catch
        # an exception delivered to this waiting task itself (e.g. an interrupt): start no new
        # chunks, let the running ones finish, then pass the exception on
        failed[] = true
        foreach(_wait_quietly, tasks)
        rethrow()
    end
    for e in errors
        e === nothing || throw(e)
    end
    return nothing
end

"""
    tmap(f, xs; threaded=true, chunksize=1) -> Vector

Apply `f` to every element of `xs` (collected into a vector), in parallel when `threaded` is true
and more than one thread is available. Results are returned in the order of `xs`. If `f` throws,
no further elements are started, the running ones are waited for, and the exception is rethrown
(unwrapped, see [`unwrap_task_exception`](@ref)).
"""
function tmap(f, xs; threaded::Bool=true, chunksize::Integer=1)
    v = collect(xs)
    (threaded && threads_available() && length(v) > 1) || return map(f, v)
    out = Vector{Any}(undef, length(v))
    _run_chunks(k -> (out[k] = f(v[k])), length(v), Int(chunksize))
    return [x for x in out]                 # the narrowest common element type
end

"""
    tforeach(f, xs; threaded=true, chunksize=1)

Call `f` on every element of `xs` for its side effects, in parallel when `threaded` is true.
`f` must only write to disjoint locations. Exceptions are handled as in [`tmap`](@ref).
"""
function tforeach(f, xs; threaded::Bool=true, chunksize::Integer=1)
    v = collect(xs)
    if threaded && threads_available() && length(v) > 1
        _run_chunks(k -> f(v[k]), length(v), Int(chunksize))
    else
        foreach(f, v)
    end
    return nothing
end

# Wait for `task`, ignoring its failure.
function _wait_quietly(task::Task)
    try
        wait(task)
    catch
    end
    return nothing
end

# Wait for `task`; `nothing` if it succeeded, otherwise the exception to rethrow. An exception
# delivered to the waiting task itself (e.g. an interrupt) is rethrown once `task` has finished.
function _wait_task(task::Task)
    try
        wait(task)
        return nothing
    catch err
        if !istaskdone(task)
            _wait_quietly(task)
            rethrow()
        end
        return _task_exception(err, catch_backtrace())
    end
end

"""
    concurrently(f, g; threaded=true) -> (f(), g())

Evaluate `f()` and `g()`: concurrently when `threaded` is true (`f` on a new task, `g` on the
current one), one after the other otherwise. Both are finished before the call returns or
throws; if both fail, the exception of `f` is rethrown, as in the serial order.
"""
function concurrently(f, g; threaded::Bool=true)
    (threaded && threads_available()) || return (f(), g())
    task = Threads.@spawn f()
    b = try
        g()
    catch
        e = _wait_task(task)
        e === nothing || throw(e)
        rethrow()
    end
    e = _wait_task(task)
    e === nothing || throw(e)
    return (fetch(task), b)
end
