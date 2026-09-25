using Documenter
using Syntomic

DocMeta.setdocmeta!(Syntomic, :DocTestSetup, :(using Syntomic); recursive=true)

makedocs(;
    modules=[Syntomic],
    sitename="Syntomic.jl",
    authors="Gabriel Ong",
    format=Documenter.HTML(;
        prettyurls=false,
        repolink="https://github.com/wgabrielong/Syntomic.jl",
        edit_link=nothing,
    ),
    # The repository is private and the documentation is not deployed, so no source links.
    remotes=nothing,
    doctest=true,
    checkdocs=:exports,
    pages=[
        "Home" => "index.md",
        "Input conventions" => "input.md",
        "Precision" => "precision.md",
        "Parallelism" => "parallelism.md",
        "Algorithm map" => "algorithm_map.md",
        "API" => "api.md",
        "Reproducing the fixtures" => "fixtures.md",
    ],
)
