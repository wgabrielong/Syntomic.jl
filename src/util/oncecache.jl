# A thread-safe memo table in which every value is computed exactly once.
#
# The first task that asks for a missing key computes the value; tasks asking for the same key
# meanwhile wait for it instead of recomputing it. Values computed elsewhere (e.g. the normal forms
# of the g-envelope, which are computed level by level, see `g_fill_normal_forms!`) are stored with
# `set_computed!`. Waiting cannot deadlock, because the value of a key only depends on keys that are
# strictly smaller for a well-founded order:
# * normal forms of the powers z_0^k in the f-envelope, keys k (`f_z_power`): computed without
#   other keys;
# * normal forms in the g-envelope, keys e for ∏ g_u^{e_u} (`g_monomial_normal_form`): rewriting
#   g_{u-1}^p lowers the F-weight of the monomial, or keeps it (the term c_{u-1} g_u) and lowers
#   the digit sum;
# * φ of g-monomials, keys t (`g_phi_monomial`): the value for t uses the one for t - p^v and
#   g-normal forms, which never use this cache.

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
    haskey(cache::OnceCache, key) -> Bool

Whether a value for `key` is stored or being computed.
"""
Base.haskey(c::OnceCache{K}, key::K) where {K} = lock(() -> haskey(c.dict, key), c.lock)

"""
    set_computed!(cache::OnceCache, key, value) -> value

Store `value`, computed elsewhere, for `key`. If a value for `key` is already stored or being
computed, the cache is not changed and that value is returned (after waiting for it).
"""
function set_computed!(cache::OnceCache{K,V}, key::K, value::V) where {K,V}
    slot, stored = lock(cache.lock) do
        s = get(cache.dict, key, nothing)
        s === nothing || return (s, false)
        s = OnceSlot{V}()
        s.value = value
        notify(s.event)
        cache.dict[key] = s
        (s, true)
    end
    stored && return value
    return _slot_value(slot)
end

# The value of a slot, once computed (rethrowing the exception of a failed computation).
function _slot_value(slot::OnceSlot{V}) where {V}
    wait(slot.event)
    slot.error === nothing || throw(slot.error)
    return slot.value::V
end

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
            slot.error = _task_exception(err, catch_backtrace())
            rethrow()
        finally
            notify(slot.event)
        end
        return slot.value::V
    end
    return _slot_value(slot)
end
