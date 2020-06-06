module Darknet
using Libdl, CEnum

using Darknet_jll

include(joinpath(@__DIR__, "..", "gen", "ctypes.jl"))
export Ctm, Ctime_t, Cclock_t
include(joinpath(@__DIR__, "..", "gen", "libdarknet_common.jl"))
include(joinpath(@__DIR__, "..", "gen", "libdarknet_api.jl"))

include("testing.jl")

include("utils.jl")

end # module Darknet
