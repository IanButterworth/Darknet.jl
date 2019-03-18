# Python function awaiting conversion:

#function array_to_image(arr)
#    import numpy as np
#    # need to return old values to avoid python freeing memory
#    arr = arr.transpose(2,0,1)
#    c = arr.shape[0]
#    h = arr.shape[1]
#    w = arr.shape[2]
#    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
#    data = arr.ctypes.data_as(POINTER(c_float))
#    im = image(w,h,c,data)
#    return im, arr
#end