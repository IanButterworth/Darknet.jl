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
    ("car", 0.6647044f0, (392.53403f0, 56.250816f0, 33.373787f0, 34.995895f0)),
    ("car", 0.5302138f0, (386.61285f0, 57.511528f0, 54.860207f0, 74.604774f0)),
    ("bicycle", 0.3607869f0, (273.65576f0, 211.16771f0, 268.42145f0, 263.27127f0)),
    ("truck", 0.2542767f0, (388.00473f0, 56.554222f0, 22.064653f0, 55.60034f0)),
    ("truck", 0.19213372f0, (386.61285f0, 57.511528f0, 54.860207f0, 74.604774f0)),
    ("pottedplant", 0.12004495f0, (41.906433f0, 81.79831f0, 67.07692f0, 132.3367f0))
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
    all_results = []
    for _ in 1:n
        @timeit to "Send image to darknet" img_d = Darknet.array_to_image(img) #Darknet image type with pointers to source data
        @timeit to "Run detection" results = Darknet.detect(net, meta, img_d, thresh=0.1, nms=0.3)
        @test length(results) == length(expected_results)


        # order of results varies across platform (I don't think order is guaranteed)
        # expected_results is manually sorted
        sort!(results, by=x -> x[2], rev=true)
        # @show results

        for (result, expected) in zip(results, expected_results)
            @test result[1] == expected[1]
            @test result[2] == expected[2]
            for (r, e) in zip(result[3], expected[3])
                @test r == e
            end
        end
        push!(all_results, results)
    end

    println(to)
end

@testset "Training" begin
    # small training config: 160x160, batch of 4, live LR from iteration 1
    cfgsrc = read(joinpath(datadir, "yolov3-tiny.cfg"), String)
    setkey(s, key, val) = replace(s, Regex("(?m)^$(key)\\s*=.*\$") => "$key=$val"; count=1)
    cfg = setkey(cfgsrc, "batch", 4)
    cfg = setkey(cfg, "subdivisions", 1)
    cfg = setkey(cfg, "width", 160)
    cfg = setkey(cfg, "height", 160)
    cfg = setkey(cfg, "burn_in", 0)
    mktempdir() do dir
        cfgpath = joinpath(dir, "train.cfg")
        write(cfgpath, cfg)
        net = Darknet.load_network(cfgpath, joinpath(datadir, "yolov3-tiny.weights"), 1)

        # toy batch: white rectangles on dark backgrounds, class 1
        x = fill(0.1f0, 160, 160, 3, 4)
        boxes = Vector{Matrix{Float32}}(undef, 4)
        rects = [(40:80, 30:60), (20:50, 100:140), (90:130, 70:110), (100:150, 20:45)]
        for (i, (xr, yr)) in enumerate(rects)
            x[xr, yr, :, i] .= 1f0
            boxes[i] = Float32[1; (first(xr)+last(xr))/2/160; (first(yr)+last(yr))/2/160; length(xr)/160; length(yr)/160;;]
        end

        td = training_data(net, x, boxes)
        @test td.d.X.rows == 4
        @test td.d.y.rows == 4

        losses = Float32[]
        redirect_stdout(devnull) do # silence per-layer training prints
            for _ in 1:20
                push!(losses, train_network!(net, td))
            end
        end
        @test all(isfinite, losses)
        @test losses[end] < losses[1]
        @test current_iteration(net) == 20
        @test current_rate(net) > 0

        wout = joinpath(dir, "trained.weights")
        Darknet.save_weights(net, wout)
        @test filesize(wout) == filesize(joinpath(datadir, "yolov3-tiny.weights"))
    end
end
