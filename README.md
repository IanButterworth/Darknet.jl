# Darknet.jl

Wrapper for https://github.com/AlexeyAB/darknet

Currently only works on:
-  MacOS, based on a CPU-only darknet built binary

## Installation

For the MacOS build, you just need to:
```
] add https://github.com/ianshmean/Darknet.jl
```

## Testing
```
using Darknet, Images
d = "/path/to/weights_and_config_files/"
p = "/path/to/imagess"
weightsfile = "yolov3-tiny.weights"
cfgfile = "yolov3-tiny.cfg"
namesfile = "coco.data"
picfile = "test.JPEG"

net = Darknet.load_network(joinpath(d,cfgfile), joinpath(d,weightsfile),1)
meta = Darknet.get_metadata(joinpath(d,namesfile));

img_d = Darknet.load_image_color(joinpath(p,picfile),0,0);
img = convert(Array{Float32}, load(joinpath(p,picfile)))

results = Darknet.detect(net,meta,img_d,thresh=0.1,nms=0.3)

using Makie, GeometryTypes
scene = Scene(resolution = size(img'))
image!(scene,img',scale_plot = false)

for res in results
    bbox = (res[3])
    poly!(scene,[Rectangle{Float32}(bbox[1]-(bbox[3]/2),bbox[2]-(bbox[4]/2),bbox[3],bbox[4])],color=RGBA(0,1,0,0.2), strokewidth = 1, strokecolor = :green)
end
scene
```


