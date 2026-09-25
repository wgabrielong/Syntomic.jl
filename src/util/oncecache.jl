# A thread-safe memo table in which every value is computed exactly once.
#
# The first task that asks for a missing key computes the value; tasks asking for the same key
# meanwhile wait for it instead of recomputing it. The dependencies between the memoized normal
# forms strictly decrease (in F-weight and digit sum), so waiting cannot deadlock.

mutable struct OnceSlot{V}
    value::Union{Nothing,V}
    error::Union{Nothing,Exception}
    const event::Base.Event
    OnceSlot{V}() where {V} = new{V}(nothing, nothing, Base.Event())
end

"""
    OnceCache{K,V}()

A thread-safe memo table: [`get_or_compute!`](@ref) computes the value of each key at most once.
"""
struct OnceCache{K,V}
    lock::ReentrantLock
    dict::Dict{K,OnceSlot{V}}
end

OnceCache{K,V}() where {K,V} = OnceCache{K,V}(ReentrantLock(), Dict{K,OnceSlot{V}}())

Base.length(c::OnceCache) = lock(() -> length(c.dict), c.lock)

"""
    get_or_compute!(f, cache::OnceCache, key)

The value stored for `key`, computing it as `f()` if it is missing. If another task is computing
it, waits for that task; if that computation failed, rethrows its exception.
"""
function get_or_compute!(f, cache::OnceCache{K,V}, key::K) where {K,V}
    slot, owner = lock(cache.lock) do
        s = get(cache.dict, key, nothing)
        if s === nothing
            s = OnceSlot{V}()
            cache.dict[key] = s
            (s, true)
        else
            (s, false)
        end
    end
    if owner
        try
            slot.value = f()
        catch err
            slot.error = err isa Exception ? err : ErrorException(string(err))
            rethrow()
        finally
            notify(slot.event)
        end
        return slot.value::V
    end
    wait(slot.event)
    slot.error === nothing || throw(slot.error)
    return slot.value::V
end
