function detect(net::Ptr{network}, meta::metadata, img::image; thresh=0.5, hier_thresh=0.5, nms=0.0)
    num = Cint(0)
    pnum = Ref(num)
    network_predict_image(net, img)
    dets = get_network_boxes(net, img.w, img.h, Cfloat(thresh), Cfloat(hier_thresh), C_NULL, 0, pnum, 0)
    num = pnum[]
    if nms > 0
        do_nms_sort(dets, num, meta.classes, Cfloat(nms))
    end
    res = []
    detections = unsafe_wrap(Array, dets, num)
    names = unsafe_wrap(Array, meta.names, meta.classes)
    for j in 1:num
        prob = unsafe_wrap(Array, detections[j].prob, meta.classes)
        for i in 1:meta.classes
            if prob[i] > 0
                b = detections[j].bbox
                nameTag = names[i]
                push!(res, (nameTag, prob[i], (b.x, b.y, b.w, b.h)))
            end
        end
    end
    free_detections(dets, num)
    return res
end
