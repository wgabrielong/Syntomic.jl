# Export of results as JSON or CSV (plain text, no dependencies).
#
# Big integers (the coordinates of h̃ and E, orders of groups) are written as decimal strings in
# JSON; small integers (p, f, n, i, exponents, precisions) as numbers. In CSV, lists are written
# with ';' between entries (and ' ' between the coordinates of an element of W(F_q)), so that no
# field contains a comma; a field containing a comma, a quote or a line break (only possible in
# a label) is quoted as in RFC 4180.

const EXPORT_FORMAT_ID = "syntomic-results-v1"

"""
    save_results(path::AbstractString, results; format=:auto) -> path
    save_results(io::IO, results; format) -> io

Write `results` as JSON (`format = :json`) or CSV (`format = :csv`); with `format = :auto` the
format is chosen by the extension (`.json` or `.csv`) of `path`. `results` is

- a vector of [`SyntomicCohomology`](@ref) results (for instance from [`syntomic_sweep`](@ref); a
  single result, or a vector of such vectors from a sweep over several rings, is also accepted):
  one row per result with the columns `p`, `f`, `e`, `n`, `i`, `label`, `htilde`, `E` (the ring),
  `h1`, `h2` (the exponent lists of `H^1` and `H^2`), `N`, `prec0`, `prec1`, `policy`,
  `restarts` and `time_s` (the total of the recorded timings, in seconds);
- a [`KTheoryTable`](@ref): one row per degree with the columns `p`, `f`, `e`, `n`, `label`,
  `integral`, `degree`, `exponents`, `prime_to_p_order` and `group` (the primary decomposition,
  as displayed).

JSON output is an object `{"format", "kind", "conventions", "rows"}` (a K-theory table also has
`"ring"` and `"integral"`); big integers (`htilde`, `E`, `prime_to_p_order`) are decimal strings,
`E` is the list of the coordinate vectors of the coefficients of `E` (low to high degree), and the
JSON rows of cohomology results also carry the individual `timings`. In CSV, lists are written
with `;` between entries (`h1 = "1;1;3"`, empty for the empty list) and `E` as
`"c_0;c_1;…"` with the `f` coordinates of each coefficient separated by spaces.
"""
function save_results(path::AbstractString, results; format::Symbol=:auto)
    fmt = format == :auto ? _format_from_extension(path) : format
    open(io -> save_results(io, results; format=fmt), path, "w")
    return path
end

function save_results(io::IO, results; format::Symbol)
    format in (:json, :csv) ||
        throw(ArgumentError("format must be :json or :csv for an IO stream, got :$format"))
    rows, columns, header = _export_rows(results)
    if format == :json
        _write_json(io, vcat(header, Pair{Symbol,Any}[:rows => rows]))
        println(io)
    else
        _write_csv(io, rows, columns)
    end
    return io
end

"""
    _format_from_extension(path) -> Symbol

`:json` or `:csv` from the extension of `path` (case-insensitive); an `ArgumentError` otherwise.
"""
function _format_from_extension(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    ext == ".json" && return :json
    ext == ".csv" && return :csv
    throw(
        ArgumentError(
            "cannot infer the format from the extension of \"$path\"; use .json or .csv, or pass format=:json or format=:csv",
        ),
    )
end

# ---------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------

"""
    COHOMOLOGY_COLUMNS

The CSV columns of exported [`SyntomicCohomology`](@ref) results (JSON rows also have `timings`).
"""
const COHOMOLOGY_COLUMNS = [
    :p,
    :f,
    :e,
    :n,
    :i,
    :label,
    :htilde,
    :E,
    :h1,
    :h2,
    :N,
    :prec0,
    :prec1,
    :policy,
    :restarts,
    :time_s,
]

"""
    KTHEORY_COLUMNS

The columns of an exported [`KTheoryTable`](@ref).
"""
const KTHEORY_COLUMNS = [
    :p, :f, :e, :n, :label, :integral, :degree, :exponents, :prime_to_p_order, :group
]

"""
    _total_time(H::SyntomicCohomology) -> Float64

The total of the timings recorded in `H` (the entry `:total` if there is one).
"""
function _total_time(H::SyntomicCohomology)
    haskey(H.timings, :total) && return H.timings[:total]
    return sum(values(H.timings); init=0.0)
end

"""
    _export_row(H::SyntomicCohomology) -> Vector{Pair{Symbol,Any}}

The exported columns of one result, in the order of [`COHOMOLOGY_COLUMNS`](@ref), followed by
the individual `timings`.
"""
function _export_row(H::SyntomicCohomology)
    X = H.ring
    return Pair{Symbol,Any}[
        :p => X.p,
        :f => X.f,
        :e => X.e,
        :n => X.n,
        :i => H.i,
        :label => X.label,
        :htilde => X.htilde,
        :E => X.E,
        :h1 => H.h1,
        :h2 => H.h2,
        :N => H.N,
        :prec0 => H.prec0,
        :prec1 => H.prec1,
        :policy => String(H.policy),
        :restarts => H.restarts,
        :time_s => _total_time(H),
        :timings => sort!([String(k) => v for (k, v) in H.timings]; by=first),
    ]
end

"""
    _export_row(tab::KTheoryTable, r) -> Vector{Pair{Symbol,Any}}

The exported columns of the row `r` of a K-theory table, in the order of
[`KTHEORY_COLUMNS`](@ref).
"""
function _export_row(tab::KTheoryTable, r::KTheoryRow)
    X = tab.ring
    return Pair{Symbol,Any}[
        :p => X.p,
        :f => X.f,
        :e => X.e,
        :n => X.n,
        :label => X.label,
        :integral => tab.integral,
        :degree => r.degree,
        :exponents => r.exponents,
        :prime_to_p_order => r.prime_to_p_order,
        :group => group_string(r, X.p; integral=tab.integral),
    ]
end

"""
    _export_rows(results) -> (rows, columns, header)

The rows (vectors of `column => value` pairs), the CSV columns and the top-level JSON entries
preceding the rows, for [`save_results`](@ref).
"""
function _export_rows(results::AbstractVector{SyntomicCohomology})
    conventions =
        "h1, h2: sorted exponent lists [a_1, ...] meaning H^k = sum_j Z/p^a_j ([] = 0); " *
        "K_{2i-1}(R; Z_p) = H^1 and K_{2i-2}(R; Z_p) = H^2 for i >= 2. The ring is " *
        "W(F_q)[z]/(E, z^n) with W(F_q) = Z_p[x]/(htilde); htilde and E are lists low to high " *
        "degree, E[k+1] holding the coordinates of the coefficient of z^k in the basis " *
        "1, x, ..., x^(f-1). N: working precision; prec0, prec1: the p-adic precisions of the " *
        "two differentials; time_s: total time in seconds."
    header = Pair{Symbol,Any}[
        :format => EXPORT_FORMAT_ID,
        :kind => "syntomic_cohomology",
        :conventions => conventions,
    ]
    return map(_export_row, results), COHOMOLOGY_COLUMNS, header
end

_export_rows(H::SyntomicCohomology) = _export_rows([H])

function _export_rows(results::AbstractVector{<:AbstractVector{SyntomicCohomology}})
    return _export_rows(reduce(vcat, results; init=SyntomicCohomology[]))
end

function _export_rows(tab::KTheoryTable)
    X = tab.ring
    conventions =
        "exponents: sorted list [a_1, ...] and prime_to_p_order m meaning K_r = Z/m + " *
        "sum_j Z/p^a_j; exponents = null (empty in CSV) for K_0. group: the primary " *
        "decomposition. integral = false: p-adic K-groups K_r(R; Z_p)."
    ring = Pair{Symbol,Any}[
        :p => X.p,
        :f => X.f,
        :e => X.e,
        :n => X.n,
        :label => X.label,
        :htilde => X.htilde,
        :E => X.E,
    ]
    header = Pair{Symbol,Any}[
        :format => EXPORT_FORMAT_ID,
        :kind => "k_theory_table",
        :conventions => conventions,
        :ring => ring,
        :integral => tab.integral,
    ]
    return [_export_row(tab, r) for r in tab], KTHEORY_COLUMNS, header
end

function _export_rows(results)
    throw(
        ArgumentError(
            "save_results expects SyntomicCohomology results or a KTheoryTable, got $(typeof(results))",
        ),
    )
end

# ---------------------------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------------------------

"""
    _write_json(io, x)

Write `x` as JSON: vectors of pairs as objects (in order), vectors as arrays, `BigInt`s as
decimal strings, `nothing` as `null`.
"""
function _write_json(io::IO, x::AbstractVector{<:Pair})
    print(io, '{')
    for (k, (key, value)) in enumerate(x)
        k > 1 && print(io, ',')
        _write_json(io, String(key))
        print(io, ':')
        _write_json(io, value)
    end
    return print(io, '}')
end

function _write_json(io::IO, x::AbstractVector)
    print(io, '[')
    for (k, v) in enumerate(x)
        k > 1 && print(io, ',')
        _write_json(io, v)
    end
    return print(io, ']')
end

_write_json(io::IO, x::AbstractString) = _write_json_string(io, x)
_write_json(io::IO, x::BigInt) = _write_json_string(io, string(x))
_write_json(io::IO, x::Integer) = print(io, x)
_write_json(io::IO, x::Bool) = print(io, x ? "true" : "false")
_write_json(io::IO, ::Nothing) = print(io, "null")
function _write_json(io::IO, x::AbstractFloat)
    isfinite(x) || throw(ArgumentError("JSON cannot represent $x"))
    return print(io, Float64(x))
end

"""
    _write_json_string(io, s)

Write `s` as a JSON string literal (escaping quotes, backslashes and control characters).
"""
function _write_json_string(io::IO, s::AbstractString)
    print(io, '"')
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\t'
            print(io, "\\t")
        elseif c == '\r'
            print(io, "\\r")
        elseif c < ' '
            print(io, "\\u", string(UInt32(c); base=16, pad=4))
        else
            print(io, c)
        end
    end
    return print(io, '"')
end

# ---------------------------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------------------------

"""
    _csv_field(x) -> String

The CSV text of one value: lists joined with `;` (the coordinates of an element of `W(F_q)`
with spaces), `nothing` as the empty field; quoted as in RFC 4180 if it contains a comma, a
quote or a line break.
"""
_csv_field(x) = _csv_quote(_csv_text(x))

_csv_text(x::AbstractString) = String(x)
_csv_text(x::Union{Integer,AbstractFloat}) = string(x)
_csv_text(x::Bool) = x ? "true" : "false"
_csv_text(::Nothing) = ""
_csv_text(x::AbstractVector{<:Integer}) = join(string.(x), ';')
function _csv_text(x::AbstractVector{<:AbstractVector{<:Integer}})
    return join((join(string.(c), ' ') for c in x), ';')
end

function _csv_quote(s::String)
    any(c -> c in (',', '"', '\n', '\r'), s) || return s
    return string('"', replace(s, "\"" => "\"\""), '"')
end

"""
    _write_csv(io, rows, columns)

Write the given columns of the rows as CSV, with a header line.
"""
function _write_csv(io::IO, rows, columns::Vector{Symbol})
    println(io, join(String.(columns), ','))
    for row in rows
        cells = Dict(row)
        println(io, join((_csv_field(cells[c]) for c in columns), ','))
    end
    return nothing
end
