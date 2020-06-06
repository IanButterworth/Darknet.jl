using Clang
using Darknet_jll

# LIBDARKNET_HEADERS are those headers to be wrapped.
const LIBDARKNET_INCLUDE = joinpath(dirname(Darknet_jll.libdarknet_path), "..", "include") |> normpath
const LIBDARKNET_HEADERS = [joinpath(LIBDARKNET_INCLUDE, header) for header in readdir(LIBDARKNET_INCLUDE) if endswith(header, ".h")]

wc = init(; headers = LIBDARKNET_HEADERS,
            output_file = joinpath(@__DIR__, "libdarknet_api.jl"),
            common_file = joinpath(@__DIR__, "libdarknet_common.jl"),
            clang_includes = vcat(LIBDARKNET_INCLUDE, CLANG_INCLUDE),
            clang_args = ["-I", joinpath(LIBDARKNET_INCLUDE, "..")],
            header_wrapped = (root, current)->root == current,
            header_library = x->"libdarknet",
            clang_diagnostics = true,
            )

run(wc)