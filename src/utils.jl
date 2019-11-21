"""
array_to_image(arr::Array{Float32},arr_permuted::Array{Float32}) -> image (darknet type with pointers)

Convert array to darknet image type, avoiding allocation
"""
function array_to_image(arr::Array{Float32}, arr_permuted::Array{Float32})
    if ndims(arr) == 3
        permutedims!(arr_permuted, arr, [3, 2, 1])
    elseif lndims(arr) == 2
        permutedims!(arr_permuted, arr, [2, 1])
    else
        error("Image does not have 2 or 3 dims")
    end
    w, h, c = size(arr_permuted)
    if c > 1
        return image(w, h, c, pointer(arr_permuted))
    else
        return image(w, h, c, pointer(arr_permuted))
    end
end

"""
array_to_image(arr::Array{Float32}) -> image (darknet type with pointers)

Convert array to darknet image type
"""
function array_to_image(arr::Array{Float32})
    if ndims(arr) == 3
        arr_permuted = permutedims(arr, [3, 2, 1])
    elseif ndims(arr) == 2
        arr_permuted = reshape(
            permutedims(arr, [2, 1]),
            (size(arr, 2), size(arr, 1), 1),
        )
    else
        error("Image does not have 2 or 3 dims")
    end
    @show w, h, c = size(arr_permuted)
    return image(w, h, c, pointer(arr_permuted))
end
