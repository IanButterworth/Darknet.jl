# Julia wrapper for header: darknet-master/include/darknet.h
# Automatically generated using Clang.jl wrap_c


function load_network(cfg, weights, clear)
    ccall((:load_network, libdarknet), Ptr{network}, (Cstring, Cstring, Cint), cfg, weights, clear)
end

function load_network_custom(cfg, weights, clear, batch)
    ccall((:load_network_custom, libdarknet), Ptr{network}, (Cstring, Cstring, Cint, Cint), cfg, weights, clear, batch)
end

function load_network(cfg, weights, clear)
    ccall((:load_network, libdarknet), Ptr{network}, (Cstring, Cstring, Cint), cfg, weights, clear)
end

function get_base_args(net)
    ccall((:get_base_args, libdarknet), load_args, (Ptr{network},), net)
end

function do_nms_sort(dets, total, classes, thresh)
    ccall((:do_nms_sort, libdarknet), Cvoid, (Ptr{detection}, Cint, Cint, Cfloat), dets, total, classes, thresh)
end

function do_nms_obj(dets, total, classes, thresh)
    ccall((:do_nms_obj, libdarknet), Cvoid, (Ptr{detection}, Cint, Cint, Cfloat), dets, total, classes, thresh)
end

function network_predict(net, input)
    ccall((:network_predict, libdarknet), Ptr{Cfloat}, (network, Ptr{Cfloat}), net, input)
end

function network_predict_ptr(net, input)
    ccall((:network_predict_ptr, libdarknet), Ptr{Cfloat}, (Ptr{network}, Ptr{Cfloat}), net, input)
end

function get_network_boxes(net, w, h, thresh, hier, map, relative, num, letter)
    ccall((:get_network_boxes, libdarknet), Ptr{detection}, (Ptr{network}, Cint, Cint, Cfloat, Cfloat, Ptr{Cint}, Cint, Ptr{Cint}, Cint), net, w, h, thresh, hier, map, relative, num, letter)
end

function free_detections(dets, n)
    ccall((:free_detections, libdarknet), Cvoid, (Ptr{detection}, Cint), dets, n)
end

function fuse_conv_batchnorm(net)
    ccall((:fuse_conv_batchnorm, libdarknet), Cvoid, (network,), net)
end

function calculate_binary_weights(net)
    ccall((:calculate_binary_weights, libdarknet), Cvoid, (network,), net)
end

function detection_to_json(dets, nboxes, classes, names, frame_id, filename)
    ccall((:detection_to_json, libdarknet), Cstring, (Ptr{detection}, Cint, Cint, Ptr{Cstring}, Clonglong, Cstring), dets, nboxes, classes, names, frame_id, filename)
end

function get_network_layer(net, i)
    ccall((:get_network_layer, libdarknet), Ptr{layer}, (Ptr{network}, Cint), net, i)
end

function make_network_boxes(net, thresh, num)
    ccall((:make_network_boxes, libdarknet), Ptr{detection}, (Ptr{network}, Cfloat, Ptr{Cint}), net, thresh, num)
end

function reset_rnn(net)
    ccall((:reset_rnn, libdarknet), Cvoid, (Ptr{network},), net)
end

function network_predict_image(net, im)
    ccall((:network_predict_image, libdarknet), Ptr{Cfloat}, (Ptr{network}, image), net, im)
end

function validate_detector_map(datacfg, cfgfile, weightfile, thresh_calc_avg_iou, iou_thresh, existing_net)
    ccall((:validate_detector_map, libdarknet), Cfloat, (Cstring, Cstring, Cstring, Cfloat, Cfloat, Ptr{network}), datacfg, cfgfile, weightfile, thresh_calc_avg_iou, iou_thresh, existing_net)
end

function train_detector(datacfg, cfgfile, weightfile, gpus, ngpus, clear, dont_show, calc_map, mjpeg_port)
    ccall((:train_detector, libdarknet), Cvoid, (Cstring, Cstring, Cstring, Ptr{Cint}, Cint, Cint, Cint, Cint, Cint), datacfg, cfgfile, weightfile, gpus, ngpus, clear, dont_show, calc_map, mjpeg_port)
end

function test_detector(datacfg, cfgfile, weightfile, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile)
    ccall((:test_detector, libdarknet), Cvoid, (Cstring, Cstring, Cstring, Cstring, Cfloat, Cfloat, Cint, Cint, Cint, Cstring), datacfg, cfgfile, weightfile, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile)
end

function network_width(net)
    ccall((:network_width, libdarknet), Cint, (Ptr{network},), net)
end

function network_height(net)
    ccall((:network_height, libdarknet), Cint, (Ptr{network},), net)
end

function optimize_picture(net, orig, max_layer, scale, rate, thresh, norm)
    ccall((:optimize_picture, libdarknet), Cvoid, (Ptr{network}, image, Cint, Cfloat, Cfloat, Cfloat, Cint), net, orig, max_layer, scale, rate, thresh, norm)
end

function resize_image(im, w, h)
    ccall((:resize_image, libdarknet), image, (image, Cint, Cint), im, w, h)
end

function copy_image_from_bytes(im, pdata)
    ccall((:copy_image_from_bytes, libdarknet), Cvoid, (image, Cstring), im, pdata)
end

function letterbox_image(im, w, h)
    ccall((:letterbox_image, libdarknet), image, (image, Cint, Cint), im, w, h)
end

function rgbgr_image(im)
    ccall((:rgbgr_image, libdarknet), Cvoid, (image,), im)
end

function make_image(w, h, c)
    ccall((:make_image, libdarknet), image, (Cint, Cint, Cint), w, h, c)
end

function load_image_color(filename, w, h)
    ccall((:load_image_color, libdarknet), image, (Cstring, Cint, Cint), filename, w, h)
end

function free_image(m)
    ccall((:free_image, libdarknet), Cvoid, (image,), m)
end

function free_layer(arg1)
    ccall((:free_layer, libdarknet), Cvoid, (layer,), arg1)
end

function free_data(d)
    ccall((:free_data, libdarknet), Cvoid, (data,), d)
end

function load_data()
    ccall((:load_data, libdarknet), Cint, ())
end

function load_data_in_thread()
    ccall((:load_data_in_thread, libdarknet), Cint, ())
end

function cuda_pull_array(x_gpu, x, n)
    ccall((:cuda_pull_array, libdarknet), Cvoid, (Ptr{Cfloat}, Ptr{Cfloat}, Cint), x_gpu, x, n)
end

function cuda_pull_array_async(x_gpu, x, n)
    ccall((:cuda_pull_array_async, libdarknet), Cvoid, (Ptr{Cfloat}, Ptr{Cfloat}, Cint), x_gpu, x, n)
end

function cuda_set_device(n)
    ccall((:cuda_set_device, libdarknet), Cvoid, (Cint,), n)
end

function free_ptrs(ptrs, n)
    ccall((:free_ptrs, libdarknet), Cvoid, (Ptr{Ptr{Cvoid}}, Cint), ptrs, n)
end

function top_k(a, n, k, index)
    ccall((:top_k, libdarknet), Cvoid, (Ptr{Cfloat}, Cint, Cint, Ptr{Cint}), a, n, k, index)
end

function read_tree(filename)
    ccall((:read_tree, libdarknet), Ptr{tree}, (Cstring,), filename)
end

function get_metadata(file)
    ccall((:get_metadata, libdarknet), metadata, (Cstring,), file)
end

function get_time_point()
    ccall((:get_time_point, libdarknet), Cdouble, ())
end

function start_timer()
    ccall((:start_timer, libdarknet), Cvoid, ())
end

function stop_timer()
    ccall((:stop_timer, libdarknet), Cvoid, ())
end

function get_time()
    ccall((:get_time, libdarknet), Cdouble, ())
end

function stop_timer_and_show()
    ccall((:stop_timer_and_show, libdarknet), Cvoid, ())
end

function stop_timer_and_show_name(name)
    ccall((:stop_timer_and_show_name, libdarknet), Cvoid, (Cstring,), name)
end

function show_total_time()
    ccall((:show_total_time, libdarknet), Cvoid, ())
end
