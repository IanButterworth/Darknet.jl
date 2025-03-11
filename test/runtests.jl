using Darknet
using FileIO
using ImageCore
using Test
using TimerOutputs

Darknet.download_defaults()

const to = TimerOutput()

datadir = joinpath(dirname(@__DIR__), "data")

@testset "Create Darknet image" begin
    darknet_img = Darknet.array_to_image(rand(Float32,100,10))
    @test typeof(darknet_img) == Darknet.Image{2}
    @test darknet_img.h == 100
    @test darknet_img.w == 10
    @test darknet_img.c == 1

    darknet_img = Darknet.array_to_image(rand(Float32,3,100,10))
    @test typeof(darknet_img) == Darknet.Image{3}
    @test darknet_img.h == 100
    @test darknet_img.w == 10
    @test darknet_img.c == 3
end

@testset "Read Metadata" begin
    meta = Darknet.get_metadata(joinpath(dirname(@__DIR__),"data","coco.data"))
    @test 80 == meta.classes
end

n = 5
@testset "Load and run $n times" begin
    weightsfile = "yolov3-tiny.weights"
    cfgfile = "yolov3-tiny.cfg"
    datafile = "coco.data"

    @timeit to "Load network" net = Darknet.load_network(joinpath(datadir, cfgfile), joinpath(datadir, weightsfile), 1)
    @timeit to "Get metadata" meta = Darknet.get_metadata(joinpath(datadir, datafile));

    imagefile = joinpath(@__DIR__, "examples", "dog-cycle-car.png")

    img = convert(Array{Float32}, channelview(load(imagefile))) #Read in array via a julia method
    img = img[1:3, :, :] #throw away the alpha channel
    results = nothing
    for _ in 1:n
        @timeit to "Send image to darknet" img_d = Darknet.array_to_image(img) #Darknet image type with pointers to source data
        @timeit to "Run detection" results = Darknet.detect(net, meta, img_d, thresh=0.1, nms=0.3)
        @test length(results) == 7
    end
    @info "Objects detected: $(length(results))"
    println(to)
end
