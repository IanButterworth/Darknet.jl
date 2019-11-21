
const datadir = joinpath(dirname(@__DIR__), "data")

files = readdir(datadir)
filter!(x->endswith(x, ".names"), files)

for file in files
    classes = length(readlines(joinpath(datadir, file)))
    basename = splitext(file)[1]
    open(joinpath(datadir, "$basename.data"), "w") do io
        contents = """
        classes=$classes
        train  = $(datadir)/coco_train.txt
        valid  = $(datadir)/coco_test.txt
        names = $(datadir)/coco.names
        backup = $(datadir)/new-weights/
        eval=coco
        """
        write(io, contents)
    end
end
weightsfile = joinpath(datadir,"yolov3-tiny.weights")
!isfile(weightsfile) && download("https://pjreddie.com/media/files/yolov3-tiny.weights", weightsfile)
