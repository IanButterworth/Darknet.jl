"""
array_to_image(arr::Array{Float32},arr_permuted::Array{Float32}) -> image (darknet type with pointers)

Convert array to darknet image type, avoiding allocation
"""
function array_to_image(arr::Array{Float32},arr_permuted::Array{Float32})
    if length(size(arr)) == 3 
        permutedims!(arr_permuted,arr,[2,1,3])
    else
        permutedims!(arr_permuted,arr,[2,1])
    end
    w = size(arr_permuted,1)
    h = size(arr_permuted,2)
    c = size(arr_permuted,3)
    if c > 1
        return image(w,h,c,pointer(arr_permuted))
    else
        return image(w,h,c,pointer(arr_permuted))
    end
end

"""
array_to_image(arr::Array{Float32}) -> image (darknet type with pointers)

Convert array to darknet image type
"""
function array_to_image(arr::Array{Float32})
    if length(size(arr)) == 3 
        arr_permuted = permutedims(arr,[2,1,3])
    else
        arr_permuted = permutedims(arr,[2,1])
    end
    w = size(arr_permuted,1)
    h = size(arr_permuted,2)
    c = size(arr_permuted,3)
    if c > 1
        return image(w,h,c,pointer(arr_permuted))
    else
        return image(w,h,c,pointer(arr_permuted))
    end
end
