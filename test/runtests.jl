using Darknet
using FileIO
using ImageCore
using Test
using TimerOutputs

Darknet.download_defaults()

datadir = joinpath(dirname(@__DIR__), "data")

@testset "Create Darknet image" begin
    darknet_img = Darknet.array_to_image(rand(Float32, 100, 10))
    @test typeof(darknet_img) == Darknet.Image{2}
    @test darknet_img.h == 100
    @test darknet_img.w == 10
    @test darknet_img.c == 1

    darknet_img = Darknet.array_to_image(rand(Float32, 3, 100, 10))
    @test typeof(darknet_img) == Darknet.Image{3}
    @test darknet_img.h == 100
    @test darknet_img.w == 10
    @test darknet_img.c == 3
end

@testset "Read Metadata" begin
    meta = Darknet.get_metadata(joinpath(dirname(@__DIR__), "data", "coco.data"))
    @test 80 == meta.classes
end

expected_results = Any[
    ("dog", 0.71402943f0, (184.52151f0, 247.44235f0, 160.45847f0, 235.32379f0)),
    ("bicycle", 0.3607869f0, (273.65576f0, 211.16771f0, 268.42145f0, 263.27127f0)),
    ("car", 0.5302138f0, (386.61285f0, 57.511528f0, 54.860207f0, 74.604774f0)),
    ("truck", 0.19213372f0, (386.61285f0, 57.511528f0, 54.860207f0, 74.604774f0)),
    ("car", 0.6647044f0, (392.53403f0, 56.250816f0, 33.373787f0, 34.995895f0)),
    ("truck", 0.2542767f0, (388.00473f0, 56.554222f0, 22.064653f0, 55.60034f0)),
    ("pottedplant", 0.12004495f0, (41.906433f0, 81.79831f0, 67.07692f0, 132.3367f0))
]

n = 5
@testset "Load and run $n times" begin
    to = TimerOutput()
    weightsfile = "yolov3-tiny.weights"
    cfgfile = "yolov3-tiny.cfg"
    datafile = "coco.data"

    @timeit to "Load network" net = Darknet.load_network(joinpath(datadir, cfgfile), joinpath(datadir, weightsfile), 1)
    @timeit to "Get metadata" meta = Darknet.get_metadata(joinpath(datadir, datafile))

    imagefile = joinpath(@__DIR__, "examples", "dog-cycle-car.png")

    img = convert(Array{Float32}, channelview(load(imagefile))) #Read in array via a julia method
    img = img[1:3, :, :] #throw away the alpha channel
    results = nothing
    for _ in 1:n
        @timeit to "Send image to darknet" img_d = Darknet.array_to_image(img) #Darknet image type with pointers to source data
        @timeit to "Run detection" results = Darknet.detect(net, meta, img_d, thresh=0.1, nms=0.3)
        @test length(results) == 7
    end

    @test length(results) == length(expected_results)
    sort!(results, by=x->x[2], rev=true)
    sort!(expected_results, by=x->x[2], rev=true)
    for (result, expected) in zip(results, expected_results)
        @test result == expected
    end
    println(to)
end
