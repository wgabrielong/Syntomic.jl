using Aqua

@testset "Aqua" begin
    # Ambiguities are checked only for methods defined in Syntomic itself: Nemo/Hecke contain
    # ambiguities of their own that are outside our control.
    Aqua.test_all(Syntomic; ambiguities=false)
    Aqua.test_ambiguities(Syntomic; recursive=false)
end
