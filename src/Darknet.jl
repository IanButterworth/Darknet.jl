module Darknet
using Libdl, CEnum

using Darknet_jll

include(joinpath(@__DIR__, "..", "gen", "ctypes.jl"))
export Ctm, Ctime_t, Cclock_t
include(joinpath(@__DIR__, "..", "lib", "LibDarknet.jl"))

include("testing.jl")

include("utils.jl")
struct Image{N}
    w::Cint
    h::Cint
    c::Cint
    data::Array{Cfloat, N}
    Image(w,h,c,data::Array{Cfloat, N}) where N = new{N}(w,h,c,data)
end
Base.cconvert(::Type{image}, im::Image) = im
Base.unsafe_convert(::Type{image}, im::Image) = image(im.w, im.h, im.c, pointer(im.data))

end # module Darknet
