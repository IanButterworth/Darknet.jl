# Training API: drive darknet's native training (forward + backward + update)
# from Julia.

"""
    TrainingData

Holds a darknet `data` struct for detection training together with the Julia
buffers it points into (kept alive for GC safety). Build with
[`training_data`](@ref); pass to [`train_network!`](@ref).
"""
struct TrainingData
    d::data
    xbuf::Matrix{Cfloat}
    ybuf::Matrix{Cfloat}
    xptrs::Vector{Ptr{Cfloat}}
    yptrs::Vector{Ptr{Cfloat}}
end

# The output ([yolo]/[region]) layer carries the truth geometry
# (l.truths = max_boxes * truth_size). net.truths is not populated by the
# parser, so scan from the last layer for one with truths set.
function _output_layer(net::network)
    for i in net.n:-1:1
        l = unsafe_load(net.layers, i)
        l.truths > 0 && return l
    end
    throw(ArgumentError("no layer with truth outputs found; is this a detection network?"))
end

"""
    training_data(net, x::AbstractArray{Float32,4}, boxes::AbstractVector)

Build a [`TrainingData`](@ref) for `net` from an input batch and ground-truth
boxes.

`x` is `(W, H, C, N)` with darknet's memory layout (x fastest, then y, then
channel — the natural layout for column-major Julia arrays holding row-major
images, as used by e.g. ObjectDetector.jl).

`boxes` is a length-`N` vector of `5×K` matrices, one column per object:
`[class, cx, cy, w, h]` with 1-based class and box center/size normalized to
the image (darknet label convention, except 1-based class). Truth rows are
encoded as darknet expects: `[cx, cy, w, h, class0]` strided by the network's
`truth_size`, up to `max_boxes` boxes per image.
"""
function training_data(net::Network, x::AbstractArray{Float32,4}, boxes::AbstractVector)
    n = unsafe_load(net.ptr)
    w, h, c, nim = size(x)
    (w, h, c) == (n.w, n.h, n.c) || throw(ArgumentError("input batch is $((w, h, c)), network expects $((n.w, n.h, n.c))"))
    length(boxes) == nim || throw(ArgumentError("$(nim) images but $(length(boxes)) box sets"))
    l = _output_layer(n)
    truths = Int(l.truths)
    stride = l.truth_size > 0 ? Int(l.truth_size) : 5
    maxboxes = truths ÷ stride

    npix = w * h * c
    xbuf = Matrix{Cfloat}(undef, npix, nim)
    copyto!(xbuf, reshape(x, npix, nim))
    ybuf = zeros(Cfloat, truths, nim)
    for (i, bxs) in enumerate(boxes)
        size(bxs, 1) == 5 || throw(ArgumentError("boxes must be 5×K ([class, cx, cy, w, h] columns)"))
        for k in 1:min(size(bxs, 2), maxboxes)
            off = (k - 1) * stride
            ybuf[off+1, i] = bxs[2, k] # cx
            ybuf[off+2, i] = bxs[3, k] # cy
            ybuf[off+3, i] = bxs[4, k] # w
            ybuf[off+4, i] = bxs[5, k] # h
            ybuf[off+5, i] = bxs[1, k] - 1 # 1-based -> darknet 0-based class id
        end
    end
    xptrs = [pointer(xbuf, (i - 1) * npix + 1) for i in 1:nim]
    yptrs = [pointer(ybuf, (i - 1) * truths + 1) for i in 1:nim]
    X = matrix(nim, npix, pointer(xptrs))
    y = matrix(nim, truths, pointer(yptrs))
    d = data(w, h, X, y, 1, C_NULL, C_NULL) # shallow=1: darknet must not free our buffers
    return TrainingData(d, xbuf, ybuf, xptrs, yptrs)
end

"""
    train_network!(net, td::TrainingData) -> Float32

Run darknet training (forward + backward + weight update, using the cfg's
learning-rate schedule) on one batch of data. The data row count should equal
the cfg's `batch` (with `subdivisions=1`). Returns the batch loss.
"""
function train_network!(net::Network, td::TrainingData)
    GC.@preserve td begin
        return ccall((:train_network, libdarknet), Cfloat, (network, data),
                     unsafe_load(net.ptr), td.d)
    end
end

"""
    save_weights(net, path)

Write the network's weights in darknet `.weights` format.
"""
function save_weights(net::Network, path::AbstractString)
    ccall((:save_weights, libdarknet), Cvoid, (network, Cstring), unsafe_load(net.ptr), path)
    return path
end

"""
    current_iteration(net)

The network's training iteration counter (advanced by `train_network!`).
"""
current_iteration(net::Network) =
    Int(ccall((:get_current_iteration, libdarknet), Clonglong, (network,), unsafe_load(net.ptr)))

"""
    current_rate(net)

The current learning rate, per the cfg's schedule (burn-in, steps, ...).
"""
current_rate(net::Network) =
    Float32(ccall((:get_current_rate, libdarknet), Cfloat, (network,), unsafe_load(net.ptr)))
