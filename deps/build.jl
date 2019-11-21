using BinaryProvider

# Parse some basic command-line arguments
const verbose = "--verbose" in ARGS
const prefix = Prefix(get([a for a in ARGS if a != "--verbose"], 1, joinpath(@__DIR__, "usr")))

libpath = joinpath(@__DIR__, "usr/darknet-master")

products = Product[
    LibraryProduct(libpath,"libdarknet", :libdarknet)
    ]

# Download binaries from hosted location
bin_prefix = "https://github.com/ianshmean/bins/raw/master/3rdparty/Darknet"

download_info = Dict(
    Linux(:x86_64)  => ("$bin_prefix/darknet-AlexeyAB-YOLOv3-Ubuntu18.04-CPU-only.tar.gz", "bfa6ae50c5613fb0e8b71a884fce7fea92bb8e736674d64931e0c1fc3121251d"),
    MacOS(:x86_64)  => ("$bin_prefix/darknet-AlexeyAB-YOLOv3-MacOS.10.14.3-CPU-only.tar.gz", "c9d79e1918c785149d39920608b4efb22fc910895ab6baf9aa5f7f43169a37fe"),
    #MacOS(:x86_64)  => ("$bin_prefix/arcbasic.tar.gz", "06803eab8c89ec7c1bf39796ea448217614362b8a51a6c82eaf286be1574ba4d")
)
# First, check to see if we're all satisfied
@show satisfied(products[1]; verbose=true)
if any(!satisfied(p; verbose=false) for p in products)
    try
        # Download and install binaries
        url, tarball_hash = choose_download(download_info)
        install(url, tarball_hash; prefix=prefix, force=true, verbose=true)
    catch e
        if typeof(e) <: ArgumentError || typeof(e) <: MethodError
            error("Your platform $(Sys.MACHINE) is not supported by this package!")
        else
            rethrow(e)
        end
    end

    # Finally, write out a deps.jl file
    write_deps_file(joinpath(@__DIR__, "deps.jl"), products)
end

const datadir = joinpath(dirname(@__DIR__), "data")

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
!isfile(weightsfile) && download("https://pjreddie.com/media/files/yolov3-tiny.weights", weightsfile)
