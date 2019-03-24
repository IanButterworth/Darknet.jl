using Darknet
using Test

@testset "Create Darknet image" begin
    darknet_img = Darknet.array_to_image(rand(Float32,100,10))
    @test typeof(darknet_img) == Darknet.image
    @test darknet_img.h == 100
    @test darknet_img.w == 10
    @test darknet_img.c == 1
end

@testset "Read Metadata" begin
    cd(dirname(Darknet.libdarknet))
    meta = Darknet.get_metadata("cfg/coco.data")
    @test 80 == meta.classes
end