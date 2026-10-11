mutable struct OptResult
    name::String
    time::Float64
    f_final::Float64
    g_final::Float64
    iter::Int
    hvp::Int
    nobj::Int
    ngrad::Int
    x::Vector{Float64}
    details::NamedTuple
end

# Existing methods keep their result interface; reference methods additionally
# expose histories/counters (ANCG also separates Hessian-construction cost).
OptResult(name, time, f, g, iter, hvp, nobj, ngrad, x) =
    OptResult(name, time, f, g, iter, hvp, nobj, ngrad, x, NamedTuple())
