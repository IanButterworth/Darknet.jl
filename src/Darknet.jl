module Darknet
using CEnum
using Downloads
using Libdl

using Darknet_jll

const datadir = joinpath(dirname(@__DIR__), "data")

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

function download_defaults()
    files = readdir(datadir)
    filter!(x->endswith(x, ".names"), files)

    for file in files
        classes = length(readlines(joinpath(datadir, file)))
        basename = splitext(file)[1]
        open(joinpath(datadir, "$basename.data"), "w") do io
            contents = """
            classes=$classes
            train  = $(datadir)/coco_train.txt
            valid  = $(datadir)/coco_test.txt
            names = $(datadir)/coco.names
            backup = $(datadir)/new-weights/
            eval=coco
            """
            write(io, contents)
        end
    end
    weightsfile = joinpath(datadir,"yolov3-tiny.weights")
    !isfile(weightsfile) && Downloads.download("https://github.com/IanButterworth/Darknet.jl/releases/download/v0.3.2/yolov3-tiny.weights", weightsfile)
end

# register finalizers on load
function load_network(cfg, weights, clear)
    # Must rename this method after generating the wrapper in LibDarknet.jl to avoid name conflict
    net = _load_network(cfg, weights, clear)
    finalizer(net) do net
        free_network_ptr(net)
    end
    return net
end
function load_network_custom(cfg, weights, clear, batch)
    # Must rename this method after generating the wrapper in LibDarknet.jl to avoid name conflict
    net = _load_network_custom(cfg, weights, clear, batch)
    finalizer(net) do net
        free_network_ptr(net)
    end
    return net
end

end # module Darknet
