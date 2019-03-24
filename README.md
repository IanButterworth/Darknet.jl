# Darknet.jl

[![Build Status](https://travis-ci.org/ianshmean/Darknet.jl.svg?branch=master)](https://travis-ci.org/ianshmean/Darknet.jl)
[![Codecov](https://codecov.io/gh/ianshmean/Darknet.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ianshmean/Darknet.jl)

Wrapper for https://github.com/AlexeyAB/darknet based on pre-build binaries.

Currently only works on:
-  MacOS, based on a CPU-only darknet built binary

Further platform support, and support of CUDA, Tensor Cores, OpenMP, etc. requires of tarball-ed binaries of such builds to be added. PRs welcomed.

## Installation

For the MacOS build, you just need to:
```
] add Darknet
```

## Testing
```julia
using Darknet, Images
d = "/path/to/weights_and_config_files/"
weightsfile = "yolov3-tiny.weights"
cfgfile = "yolov3-tiny.cfg"
namesfile = "coco.data"

imagefile = "/path/to/images/test.jpg"

net = Darknet.load_network(joinpath(d,cfgfile), joinpath(d,weightsfile),1)
meta = Darknet.get_metadata(joinpath(d,namesfile));

```
Reading in an image from file:
```julia
# Read image using Darknet method
img = load(imagefile)  #Image for plotting in julia purposes only (below)
img_d = Darknet.load_image_color(imagefile,0,0);  #Darknet native way to read in image from file. Produces an image type with pointers
```
or from an array in julia memory:

```julia
# Send image via an image in julia memory
img = convert(Array{Float32}, load(imagefile)) #Read in array via a julia method
img_d = Darknet.array_to_image(img) #Darknet image type with pointers to source data
```

or for looping through images from julia, avoid reallocation due to permuted dims:

```julia
img = convert(Array{Float32}, load(imagefile)) #Read in array via a julia method

# Darknet flips the first 2 dims of an image (cols,rows,colorchannels)
# so preallocate a permuted dims array to prevent reallocation in 
if size(img,3) > 1 #if more than 1 color channel 
    img_permuted = Array{Float32}(undef,size(img,2),size(img,1),size(img,3)) 
else
    img_permuted = Array{Float32}(undef,size(img,2),size(img,1)) 
end

img_d = Darknet.array_to_image(img,img_permuted) #Darknet image type with pointers to source data
```

Run detection
```julia
results = Darknet.detect(net,meta,img_d,thresh=0.1,nms=0.3)
```

Preview result using Makie:
```julia 
using Makie, GeometryTypes
scene = Scene(resolution = size(img'))
image!(scene,img',scale_plot = false)

for res in results
    bbox = res[3]
    poly!(scene,[Rectangle{Float32}(bbox[1]-(bbox[3]/2),bbox[2]-(bbox[4]/2),bbox[3],bbox[4])],color=RGBA(0,1,0,clamp(conf,0.05,0.5)))
end
scene
```


