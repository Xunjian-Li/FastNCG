module FastNCG

using LinearAlgebra
using NLPModels
using Printf
using AdaptiveRegularization
import NLPModels: reset!, obj, grad

include("results.jl")
include("Fast_NCG.jl")
include("problems.jl")
include("GD.jl")
include("NAG.jl")
include("CR.jl")
include("ANCG.jl")
include("ARNCG.jl")
include("ARCqK.jl")

export OptResult, gd_optimize, nag_optimize, cr_optimize,
       ancg_optimize, arncg_official_optimize, arcqk_optimize

end # module FastNCG
