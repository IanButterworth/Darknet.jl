using Darknet
using Test

darknet_img = Darknet.array_to_image(rand(Float32,100,100))

@test typeof(darknet_img) == Darknet.image