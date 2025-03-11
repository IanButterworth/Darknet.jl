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
    ("dog", 0.71404153f0, (184.5216f0, 247.4424f0, 160.45781f0, 235.32559f0)),
    ("car", 0.66470796f0, (392.534f0, 56.250736f0, 33.37399f0, 34.995964f0)),
    ("car", 0.5302366f0, (386.61276f0, 57.51152f0, 54.860195f0, 74.6036f0)),
    ("bicycle", 0.36078787f0, (273.65543f0, 211.16792f0, 268.42365f0, 263.27005f0)),
    ("truck", 0.25427902f0, (388.0046f0, 56.55401f0, 22.06442f0, 55.60041f0)),
    ("truck", 0.1921224f0, (386.61276f0, 57.51152f0, 54.860195f0, 74.6036f0)),
    ("pottedplant", 0.12004832f0, (41.90645f0, 81.798485f0, 67.07667f0, 132.33755f0))
]
# sort!(expected_results, by=x->x[2], rev=true)

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
        @test length(results) == length(expected_results)

        # @show results
        # order of results varies across platform (I don't think order is guaranteed)
        # expected_results is manually sorted
        sort!(results, by=x -> x[2], rev=true)

        for (result, expected) in zip(results, expected_results)
            @test result[1] == expected[1]
            @test result[2] ≈ expected[2]
            for (r, e) in zip(result[3], expected[3])
                @test r ≈ e
            end
        end
    end
    println(to)
end
