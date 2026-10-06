@testset "style and traceability" begin
    root = normpath(joinpath(@__DIR__, ".."))
    srcfiles = String[]
    for (dir, _, files) in walkdir(joinpath(root, "src")), file in files
        endswith(file, ".jl") && push!(srcfiles, joinpath(dir, file))
    end

    @testset "every step function cites [AKN24]" begin
        # the functions listed in the algorithm map of the documentation: its @docs block and
        # every name linked from its table
        map_md = read(joinpath(root, "docs", "src", "algorithm_map.md"), String)
        block = match(r"```@docs\n(.*?)```"s, map_md).captures[1]
        names = [strip(l) for l in split(block, '\n') if !isempty(strip(l))]
        @test length(names) > 40
        table = filter(l -> startswith(l, "|"), split(map_md, '\n'))
        linked = [
            m.captures[1] for l in table for m in eachmatch(r"\[`([^`]+)`\]\(@ref\)", l)
        ]
        @test length(unique(linked)) > 40
        for name in linked
            name in names || push!(names, name)
        end
        meta = Base.Docs.meta(Syntomic)
        for name in names
            sym = Symbol(replace(name, "Syntomic." => ""))
            @test isdefined(Syntomic, sym)
            binding = Base.Docs.Binding(Syntomic, sym)
            @test haskey(meta, binding)
            haskey(meta, binding) || continue
            text = join((join(string.(d.text)) for d in values(meta[binding].docs)), "\n")
            occursin("[AKN24", text) || @test "$name" == "a docstring citing [AKN24, …]"
        end
        # every file implementing a step of the algorithm cites the paper
        for file in srcfiles
            occursin(r"step|envelope|lambdas|g_normal_form|g_frobenius", basename(file)) ||
                continue
            @test occursin("[AKN24,", read(file, String))
        end
    end

    @testset "references" begin
        testfiles = String[]
        for (dir, _, files) in walkdir(joinpath(root, "test")), file in files
            endswith(file, ".jl") &&
                file != "test_style.jl" &&
                push!(testfiles, joinpath(dir, file))
        end
        for file in vcat(srcfiles, testfiles)
            text = read(file, String)
            @test !occursin("Algorithm.pdf", text)
            @test !occursin(r"Krause'?s notes|Copenhagen"i, text)
            @test !occursin(r"errat|misprint"i, text)
        end
        # the fixtures record data and page references only
        fixtures = joinpath(root, "test", "fixtures")
        jsonfiles = [
            joinpath(fixtures, f) for f in readdir(fixtures) if endswith(f, ".json")
        ]
        @test length(jsonfiles) >= 3
        for file in jsonfiles
            m = match(r"errat|misprint|corrected|printed_group"i, read(file, String))
            m === nothing ||
                @test "$(basename(file))" == "no erratum wording like $(m.match)"
        end
    end

    @testset "no numeric-literal juxtaposition with f" begin
        # `2f-1` parses as the Float32 literal 2.0f-1; the residue degree is called f.
        for file in srcfiles
            in_docstring = false
            for (k, line) in enumerate(eachline(file))
                if occursin("\"\"\"", line)
                    in_docstring = !in_docstring
                    continue
                end
                in_docstring && continue
                code = split(line, '#')[1]
                m = match(r"(?<![\w.])\d+f(?![\w])", code)
                m === nothing ||
                    @test "$(basename(file)):$k" == "no literal like $(m.match)"
            end
        end
    end
end
