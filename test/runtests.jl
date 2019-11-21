using Darknet
using FileIO
using ImageCore
using Test

datadir = joinpath(dirname(@__DIR__), "data")

@testset "Create Darknet image" begin
    darknet_img = Darknet.array_to_image(rand(Float32,100,10))
    @test typeof(darknet_img) == Darknet.image
    @test darknet_img.h == 100
    @test darknet_img.w == 10
    @test darknet_img.c == 1

    darknet_img = Darknet.array_to_image(rand(Float32,3,100,10))
    @test typeof(darknet_img) == Darknet.image
    @test darknet_img.h == 100
    @test darknet_img.w == 10
    @test darknet_img.c == 3
end

@testset "Read Metadata" begin
    meta = Darknet.get_metadata(joinpath(dirname(@__DIR__),"data","coco.data"))
    @test 80 == meta.classes
end

@testset "Load and run" begin
    weightsfile = "yolov3-tiny.weights"
    cfgfile = "yolov3-tiny.cfg"
    datafile = "coco.data"

    net = Darknet.load_network(joinpath(datadir, cfgfile), joinpath(datadir, weightsfile), 1)
    @show typeof(net)
    meta = Darknet.get_metadata(joinpath(datadir, datafile));
    @show typeof(meta)

    imagefile = joinpath(@__DIR__, "examples", "dog-cycle-car.png")

    img = convert(Array{Float32}, channelview(load(imagefile))) #Read in array via a julia method
    img = img[1:3, :, :] #throw away the alpha channel
    img_d = Darknet.array_to_image(img) #Darknet image type with pointers to source data

    # results = Darknet.detect(net, meta, img_d, thresh=0.1, nms=0.3)
    # @show results
end
