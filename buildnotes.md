# Buildnotes

## Step 1: Make binary
Before Making, make sure the libdarknet.so filename setting has the 
appropriate extension  for your target operating system (.dylib for MacOS)



## Step 2: Build C-api with Clang

Clang instructions

Install MacOS dev tools:
`xcode-select --install`

Run clang:

```julia
using Clang

# LIBCLANG_HEADERS are those headers to be wrapped.
LIBCLANG_INCLUDE = joinpath("/Users/IanB/Github_Personal/darknet-master/include") |> normpath
LIBCLANG_HEADERS = [joinpath(LIBCLANG_INCLUDE, header) for header in readdir(LIBCLANG_INCLUDE) if endswith(header, ".h")]

wc = init(; headers = LIBCLANG_HEADERS,
            output_file = joinpath(@__DIR__, "libdarknet_api.jl"),
            common_file = joinpath(@__DIR__, "libdarknet_common.jl"),
            clang_includes = vcat(LIBCLANG_INCLUDE, CLANG_INCLUDE),
            clang_args = ["-I", joinpath(LIBCLANG_INCLUDE, "..")],
            header_wrapped = (root, current)->root == current,
            header_library = x->"libdarknet",
            clang_diagnostics = true,
            )

run(wc)
```