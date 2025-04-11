# based on examples like https://gitlab.eeecs.qub.ac.uk/40126401/csc4006-EdgeBenchmarking/blob/f56338fe1c6ae5b87a8ba59ea17ac64387e8213b/Experiments/YOLO/yolo/darknet.py
function detect(net, meta, img; thresh=0.5, hier_thresh=0.5, nms=0)
    num_dets = Ref(Cint(0))
    network_predict_image(net, img)
    dets = get_network_boxes(
        net,         # The YOLO neural network model (contains weights & config)
        img.w,       # Width of the input image (after preprocessing if any)
        img.h,       # Height of the input image (after preprocessing if any)
        thresh,      # Detection threshold (filter out boxes with lower confidence)
        hier_thresh, # Hierarchical threshold (used in hierarchical classification, typically ignored in standard YOLO)
        C_NULL,      # Class mapping array (NULL means no remapping of class IDs)
        0,           # Whether to return relative coordinates (0 = absolute pixel values)
        num_dets,    # Pointer to an integer storing the number of detections
        0            # Letterboxing flag (0 = image was resized without padding)
    )
    if nms > 0
        do_nms_sort(dets, num_dets[], meta.classes, nms)
    end
    res = []
    detections = unsafe_wrap(Array, dets, num_dets[])
    names = unsafe_wrap(Array, meta.names, meta.classes)
    for detection in detections
        probs = unsafe_wrap(Array, detection.prob, meta.classes)
        b = detection.bbox
        for (i, prob) in enumerate(probs)
            if prob > 0
                name = unsafe_string(names[i])
                push!(res, (name, prob, (b.x, b.y, b.w, b.h)))
            end
        end
    end
    free_detections(dets, num_dets[])
    return res
end