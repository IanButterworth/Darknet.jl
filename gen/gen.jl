using Clang.Generators
using Darknet_jll

cd(@__DIR__)

include_dir = joinpath(Darknet_jll.artifact_dir, "include") |> normpath

# wrapper generator options
options = load_options(joinpath(@__DIR__, "generator.toml"))

# add compiler flags, e.g. "-DXXXXXXXXX"
args = get_default_args()
push!(args, "-I$include_dir")

header_dir = include_dir
headers = [joinpath(header_dir, header) for header in readdir(header_dir) if endswith(header, ".h")]

# Skip pthread_t
@add_def pthread_t

# create context
ctx = create_context(headers, args, options)

# run generator
build!(ctx)