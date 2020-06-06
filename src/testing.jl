# based on examples like https://gitlab.eeecs.qub.ac.uk/40126401/csc4006-EdgeBenchmarking/blob/f56338fe1c6ae5b87a8ba59ea17ac64387e8213b/Experiments/YOLO/yolo/darknet.py
function detect(net, meta, img; thresh=0.5, hier_thresh=0.5, nms=0)
    num_dets = Ref(Cint(0))
    network_predict_image(net, img)
    dets = get_network_boxes(net, img.w, img.h, thresh, hier_thresh, C_NULL, 0, num_dets, 0)
    if nms > 0
        do_nms_sort(dets, num_dets[], meta.classes, nms)
    end
    res = []
    detections = unsafe_wrap(Array, dets, num_dets[])
    names = unsafe_wrap(Array, meta.names, meta.classes)
    for j in 1:num_dets[]
        prob = unsafe_wrap(Array, detections[j].prob, meta.classes)
        for i in 1:meta.classes  
            if prob[i] > 0
                b = detections[j].bbox
                nameTag = names[i]
                push!(res, (nameTag, prob[i], (b.x, b.y, b.w, b.h)))
            end
        end
    end
    free_detections(dets, num_dets[])
    return res
end
