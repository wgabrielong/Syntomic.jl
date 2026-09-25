@testset "style and traceability" begin
    root = normpath(joinpath(@__DIR__, ".."))
    srcfiles = String[]
    for (dir, _, files) in walkdir(joinpath(root, "src")), file in files
        endswith(file, ".jl") && push!(srcfiles, joinpath(dir, file))
    end

    @testset "every step function cites [AKN]" begin
        # the functions listed in the algorithm map of the documentation
        map_md = read(joinpath(root, "docs", "src", "algorithm_map.md"), String)
        block = match(r"```@docs\n(.*?)```"s, map_md).captures[1]
        names = [strip(l) for l in split(block, '\n') if !isempty(strip(l))]
        @test length(names) > 40
        for name in names
            sym = Symbol(replace(name, "Syntomic." => ""))
            @test isdefined(Syntomic, sym)
            @test haskey(Base.Docs.meta(Syntomic), Base.Docs.Binding(Syntomic, sym))
        end
        # every file implementing a step of the algorithm cites the paper
        for file in srcfiles
            occursin(r"step|envelope|lambdas|g_normal_form|g_frobenius", basename(file)) ||
                continue
            @test occursin("[AKN,", read(file, String))
        end
    end

    @testset "references" begin
        for file in srcfiles
            text = read(file, String)
            @test !occursin("Algorithm.pdf", text)
            @test !occursin(r"Krause'?s notes|Copenhagen"i, text)
            @test !occursin(r"errat|misprint"i, text)
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
