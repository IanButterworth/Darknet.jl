@cenum UNUSED_ENUM_TYPE::UInt32 begin
    UNUSED_DEF_VAL = 0
end

@cenum LAYER_TYPE::UInt32 begin
    CONVOLUTIONAL = 0
    DECONVOLUTIONAL = 1
    CONNECTED = 2
    MAXPOOL = 3
    LOCAL_AVGPOOL = 4
    SOFTMAX = 5
    DETECTION = 6
    DROPOUT = 7
    CROP = 8
    ROUTE = 9
    COST = 10
    NORMALIZATION = 11
    AVGPOOL = 12
    LOCAL = 13
    SHORTCUT = 14
    SCALE_CHANNELS = 15
    SAM = 16
    ACTIVE = 17
    RNN = 18
    GRU = 19
    LSTM = 20
    CONV_LSTM = 21
    HISTORY = 22
    CRNN = 23
    BATCHNORM = 24
    NETWORK = 25
    XNOR = 26
    REGION = 27
    YOLO = 28
    GAUSSIAN_YOLO = 29
    ISEG = 30
    REORG = 31
    REORG_OLD = 32
    UPSAMPLE = 33
    LOGXENT = 34
    L2NORM = 35
    EMPTY = 36
    BLANK = 37
    CONTRASTIVE = 38
    IMPLICIT = 39
end

@cenum ACTIVATION::UInt32 begin
    LOGISTIC = 0
    RELU = 1
    RELU6 = 2
    RELIE = 3
    LINEAR = 4
    RAMP = 5
    TANH = 6
    PLSE = 7
    REVLEAKY = 8
    LEAKY = 9
    ELU = 10
    LOGGY = 11
    STAIR = 12
    HARDTAN = 13
    LHTAN = 14
    SELU = 15
    GELU = 16
    SWISH = 17
    MISH = 18
    HARD_MISH = 19
    NORM_CHAN = 20
    NORM_CHAN_SOFTMAX = 21
    NORM_CHAN_SOFTMAX_MAXVAL = 22
end

@cenum COST_TYPE::UInt32 begin
    SSE = 0
    MASKED = 1
    L1 = 2
    SEG = 3
    SMOOTH = 4
    WGAN = 5
end

@cenum WEIGHTS_TYPE_T::UInt32 begin
    NO_WEIGHTS = 0
    PER_FEATURE = 1
    PER_CHANNEL = 2
end

@cenum WEIGHTS_NORMALIZATION_T::UInt32 begin
    NO_NORMALIZATION = 0
    RELU_NORMALIZATION = 1
    SOFTMAX_NORMALIZATION = 2
end

struct contrastive_params
    sim::Cfloat
    exp_sim::Cfloat
    P::Cfloat
    i::Cint
    j::Cint
    time_step_i::Cint
    time_step_j::Cint
end

@cenum IOU_LOSS::UInt32 begin
    IOU = 0
    GIOU = 1
    MSE = 2
    DIOU = 3
    CIOU = 4
end

@cenum NMS_KIND::UInt32 begin
    DEFAULT_NMS = 0
    GREEDY_NMS = 1
    DIOU_NMS = 2
    CORNERS_NMS = 3
end

@cenum YOLO_POINT::UInt32 begin
    YOLO_CENTER = 1
    YOLO_LEFT_TOP = 2
    YOLO_RIGHT_BOTTOM = 4
end

struct tree
    leaf::Ptr{Cint}
    n::Cint
    parent::Ptr{Cint}
    child::Ptr{Cint}
    group::Ptr{Cint}
    name::Ptr{Cstring}
    groups::Cint
    group_size::Ptr{Cint}
    group_offset::Ptr{Cint}
end

struct layer
    type::LAYER_TYPE
    activation::ACTIVATION
    lstm_activation::ACTIVATION
    cost_type::COST_TYPE
    forward::Ptr{Cvoid}
    backward::Ptr{Cvoid}
    update::Ptr{Cvoid}
    forward_gpu::Ptr{Cvoid}
    backward_gpu::Ptr{Cvoid}
    update_gpu::Ptr{Cvoid}
    share_layer::Ptr{layer}
    train::Cint
    avgpool::Cint
    batch_normalize::Cint
    shortcut::Cint
    batch::Cint
    dynamic_minibatch::Cint
    forced::Cint
    flipped::Cint
    inputs::Cint
    outputs::Cint
    mean_alpha::Cfloat
    nweights::Cint
    nbiases::Cint
    extra::Cint
    truths::Cint
    h::Cint
    w::Cint
    c::Cint
    out_h::Cint
    out_w::Cint
    out_c::Cint
    n::Cint
    max_boxes::Cint
    truth_size::Cint
    groups::Cint
    group_id::Cint
    size::Cint
    side::Cint
    stride::Cint
    stride_x::Cint
    stride_y::Cint
    dilation::Cint
    antialiasing::Cint
    maxpool_depth::Cint
    maxpool_zero_nonmax::Cint
    out_channels::Cint
    reverse::Cfloat
    coordconv::Cint
    flatten::Cint
    spatial::Cint
    pad::Cint
    sqrt::Cint
    flip::Cint
    index::Cint
    scale_wh::Cint
    binary::Cint
    xnor::Cint
    peephole::Cint
    use_bin_output::Cint
    keep_delta_gpu::Cint
    optimized_memory::Cint
    steps::Cint
    history_size::Cint
    bottleneck::Cint
    time_normalizer::Cfloat
    state_constrain::Cint
    hidden::Cint
    truth::Cint
    smooth::Cfloat
    dot::Cfloat
    deform::Cint
    grad_centr::Cint
    sway::Cint
    rotate::Cint
    stretch::Cint
    stretch_sway::Cint
    angle::Cfloat
    jitter::Cfloat
    resize::Cfloat
    saturation::Cfloat
    exposure::Cfloat
    shift::Cfloat
    ratio::Cfloat
    learning_rate_scale::Cfloat
    clip::Cfloat
    focal_loss::Cint
    classes_multipliers::Ptr{Cfloat}
    label_smooth_eps::Cfloat
    noloss::Cint
    softmax::Cint
    classes::Cint
    detection::Cint
    embedding_layer_id::Cint
    embedding_output::Ptr{Cfloat}
    embedding_size::Cint
    sim_thresh::Cfloat
    track_history_size::Cint
    dets_for_track::Cint
    dets_for_show::Cint
    track_ciou_norm::Cfloat
    coords::Cint
    background::Cint
    rescore::Cint
    objectness::Cint
    does_cost::Cint
    joint::Cint
    noadjust::Cint
    reorg::Cint
    log::Cint
    tanh::Cint
    mask::Ptr{Cint}
    total::Cint
    bflops::Cfloat
    adam::Cint
    B1::Cfloat
    B2::Cfloat
    eps::Cfloat
    t::Cint
    alpha::Cfloat
    beta::Cfloat
    kappa::Cfloat
    coord_scale::Cfloat
    object_scale::Cfloat
    noobject_scale::Cfloat
    mask_scale::Cfloat
    class_scale::Cfloat
    bias_match::Cint
    random::Cfloat
    ignore_thresh::Cfloat
    truth_thresh::Cfloat
    iou_thresh::Cfloat
    thresh::Cfloat
    focus::Cfloat
    classfix::Cint
    absolute::Cint
    assisted_excitation::Cint
    onlyforward::Cint
    stopbackward::Cint
    train_only_bn::Cint
    dont_update::Cint
    burnin_update::Cint
    dontload::Cint
    dontsave::Cint
    dontloadscales::Cint
    numload::Cint
    temperature::Cfloat
    probability::Cfloat
    dropblock_size_rel::Cfloat
    dropblock_size_abs::Cint
    dropblock::Cint
    scale::Cfloat
    receptive_w::Cint
    receptive_h::Cint
    receptive_w_scale::Cint
    receptive_h_scale::Cint
    cweights::Cstring
    indexes::Ptr{Cint}
    input_layers::Ptr{Cint}
    input_sizes::Ptr{Cint}
    layers_output::Ptr{Ptr{Cfloat}}
    layers_delta::Ptr{Ptr{Cfloat}}
    weights_type::WEIGHTS_TYPE_T
    weights_normalization::WEIGHTS_NORMALIZATION_T
    map::Ptr{Cint}
    counts::Ptr{Cint}
    sums::Ptr{Ptr{Cfloat}}
    rand::Ptr{Cfloat}
    cost::Ptr{Cfloat}
    labels::Ptr{Cint}
    class_ids::Ptr{Cint}
    contrastive_neg_max::Cint
    cos_sim::Ptr{Cfloat}
    exp_cos_sim::Ptr{Cfloat}
    p_constrastive::Ptr{Cfloat}
    contrast_p_gpu::Ptr{contrastive_params}
    state::Ptr{Cfloat}
    prev_state::Ptr{Cfloat}
    forgot_state::Ptr{Cfloat}
    forgot_delta::Ptr{Cfloat}
    state_delta::Ptr{Cfloat}
    combine_cpu::Ptr{Cfloat}
    combine_delta_cpu::Ptr{Cfloat}
    concat::Ptr{Cfloat}
    concat_delta::Ptr{Cfloat}
    binary_weights::Ptr{Cfloat}
    biases::Ptr{Cfloat}
    bias_updates::Ptr{Cfloat}
    scales::Ptr{Cfloat}
    scale_updates::Ptr{Cfloat}
    weights_ema::Ptr{Cfloat}
    biases_ema::Ptr{Cfloat}
    scales_ema::Ptr{Cfloat}
    weights::Ptr{Cfloat}
    weight_updates::Ptr{Cfloat}
    scale_x_y::Cfloat
    objectness_smooth::Cint
    new_coords::Cint
    show_details::Cint
    max_delta::Cfloat
    uc_normalizer::Cfloat
    iou_normalizer::Cfloat
    obj_normalizer::Cfloat
    cls_normalizer::Cfloat
    delta_normalizer::Cfloat
    iou_loss::IOU_LOSS
    iou_thresh_kind::IOU_LOSS
    nms_kind::NMS_KIND
    beta_nms::Cfloat
    yolo_point::YOLO_POINT
    align_bit_weights_gpu::Cstring
    mean_arr_gpu::Ptr{Cfloat}
    align_workspace_gpu::Ptr{Cfloat}
    transposed_align_workspace_gpu::Ptr{Cfloat}
    align_workspace_size::Cint
    align_bit_weights::Cstring
    mean_arr::Ptr{Cfloat}
    align_bit_weights_size::Cint
    lda_align::Cint
    new_lda::Cint
    bit_align::Cint
    col_image::Ptr{Cfloat}
    delta::Ptr{Cfloat}
    output::Ptr{Cfloat}
    activation_input::Ptr{Cfloat}
    delta_pinned::Cint
    output_pinned::Cint
    loss::Ptr{Cfloat}
    squared::Ptr{Cfloat}
    norms::Ptr{Cfloat}
    spatial_mean::Ptr{Cfloat}
    mean::Ptr{Cfloat}
    variance::Ptr{Cfloat}
    mean_delta::Ptr{Cfloat}
    variance_delta::Ptr{Cfloat}
    rolling_mean::Ptr{Cfloat}
    rolling_variance::Ptr{Cfloat}
    x::Ptr{Cfloat}
    x_norm::Ptr{Cfloat}
    m::Ptr{Cfloat}
    v::Ptr{Cfloat}
    bias_m::Ptr{Cfloat}
    bias_v::Ptr{Cfloat}
    scale_m::Ptr{Cfloat}
    scale_v::Ptr{Cfloat}
    z_cpu::Ptr{Cfloat}
    r_cpu::Ptr{Cfloat}
    h_cpu::Ptr{Cfloat}
    stored_h_cpu::Ptr{Cfloat}
    prev_state_cpu::Ptr{Cfloat}
    temp_cpu::Ptr{Cfloat}
    temp2_cpu::Ptr{Cfloat}
    temp3_cpu::Ptr{Cfloat}
    dh_cpu::Ptr{Cfloat}
    hh_cpu::Ptr{Cfloat}
    prev_cell_cpu::Ptr{Cfloat}
    cell_cpu::Ptr{Cfloat}
    f_cpu::Ptr{Cfloat}
    i_cpu::Ptr{Cfloat}
    g_cpu::Ptr{Cfloat}
    o_cpu::Ptr{Cfloat}
    c_cpu::Ptr{Cfloat}
    stored_c_cpu::Ptr{Cfloat}
    dc_cpu::Ptr{Cfloat}
    binary_input::Ptr{Cfloat}
    bin_re_packed_input::Ptr{UInt32}
    t_bit_input::Cstring
    input_layer::Ptr{layer}
    self_layer::Ptr{layer}
    output_layer::Ptr{layer}
    reset_layer::Ptr{layer}
    update_layer::Ptr{layer}
    state_layer::Ptr{layer}
    input_gate_layer::Ptr{layer}
    state_gate_layer::Ptr{layer}
    input_save_layer::Ptr{layer}
    state_save_layer::Ptr{layer}
    input_state_layer::Ptr{layer}
    state_state_layer::Ptr{layer}
    input_z_layer::Ptr{layer}
    state_z_layer::Ptr{layer}
    input_r_layer::Ptr{layer}
    state_r_layer::Ptr{layer}
    input_h_layer::Ptr{layer}
    state_h_layer::Ptr{layer}
    wz::Ptr{layer}
    uz::Ptr{layer}
    wr::Ptr{layer}
    ur::Ptr{layer}
    wh::Ptr{layer}
    uh::Ptr{layer}
    uo::Ptr{layer}
    wo::Ptr{layer}
    vo::Ptr{layer}
    uf::Ptr{layer}
    wf::Ptr{layer}
    vf::Ptr{layer}
    ui::Ptr{layer}
    wi::Ptr{layer}
    vi::Ptr{layer}
    ug::Ptr{layer}
    wg::Ptr{layer}
    softmax_tree::Ptr{tree}
    workspace_size::Csize_t
    indexes_gpu::Ptr{Cint}
    stream::Cint
    wait_stream_id::Cint
    z_gpu::Ptr{Cfloat}
    r_gpu::Ptr{Cfloat}
    h_gpu::Ptr{Cfloat}
    stored_h_gpu::Ptr{Cfloat}
    bottelneck_hi_gpu::Ptr{Cfloat}
    bottelneck_delta_gpu::Ptr{Cfloat}
    temp_gpu::Ptr{Cfloat}
    temp2_gpu::Ptr{Cfloat}
    temp3_gpu::Ptr{Cfloat}
    dh_gpu::Ptr{Cfloat}
    hh_gpu::Ptr{Cfloat}
    prev_cell_gpu::Ptr{Cfloat}
    prev_state_gpu::Ptr{Cfloat}
    last_prev_state_gpu::Ptr{Cfloat}
    last_prev_cell_gpu::Ptr{Cfloat}
    cell_gpu::Ptr{Cfloat}
    f_gpu::Ptr{Cfloat}
    i_gpu::Ptr{Cfloat}
    g_gpu::Ptr{Cfloat}
    o_gpu::Ptr{Cfloat}
    c_gpu::Ptr{Cfloat}
    stored_c_gpu::Ptr{Cfloat}
    dc_gpu::Ptr{Cfloat}
    m_gpu::Ptr{Cfloat}
    v_gpu::Ptr{Cfloat}
    bias_m_gpu::Ptr{Cfloat}
    scale_m_gpu::Ptr{Cfloat}
    bias_v_gpu::Ptr{Cfloat}
    scale_v_gpu::Ptr{Cfloat}
    combine_gpu::Ptr{Cfloat}
    combine_delta_gpu::Ptr{Cfloat}
    forgot_state_gpu::Ptr{Cfloat}
    forgot_delta_gpu::Ptr{Cfloat}
    state_gpu::Ptr{Cfloat}
    state_delta_gpu::Ptr{Cfloat}
    gate_gpu::Ptr{Cfloat}
    gate_delta_gpu::Ptr{Cfloat}
    save_gpu::Ptr{Cfloat}
    save_delta_gpu::Ptr{Cfloat}
    concat_gpu::Ptr{Cfloat}
    concat_delta_gpu::Ptr{Cfloat}
    binary_input_gpu::Ptr{Cfloat}
    binary_weights_gpu::Ptr{Cfloat}
    bin_conv_shortcut_in_gpu::Ptr{Cfloat}
    bin_conv_shortcut_out_gpu::Ptr{Cfloat}
    mean_gpu::Ptr{Cfloat}
    variance_gpu::Ptr{Cfloat}
    m_cbn_avg_gpu::Ptr{Cfloat}
    v_cbn_avg_gpu::Ptr{Cfloat}
    rolling_mean_gpu::Ptr{Cfloat}
    rolling_variance_gpu::Ptr{Cfloat}
    variance_delta_gpu::Ptr{Cfloat}
    mean_delta_gpu::Ptr{Cfloat}
    col_image_gpu::Ptr{Cfloat}
    x_gpu::Ptr{Cfloat}
    x_norm_gpu::Ptr{Cfloat}
    weights_gpu::Ptr{Cfloat}
    weight_updates_gpu::Ptr{Cfloat}
    weight_deform_gpu::Ptr{Cfloat}
    weight_change_gpu::Ptr{Cfloat}
    weights_gpu16::Ptr{Cfloat}
    weight_updates_gpu16::Ptr{Cfloat}
    biases_gpu::Ptr{Cfloat}
    bias_updates_gpu::Ptr{Cfloat}
    bias_change_gpu::Ptr{Cfloat}
    scales_gpu::Ptr{Cfloat}
    scale_updates_gpu::Ptr{Cfloat}
    scale_change_gpu::Ptr{Cfloat}
    input_antialiasing_gpu::Ptr{Cfloat}
    output_gpu::Ptr{Cfloat}
    output_avg_gpu::Ptr{Cfloat}
    activation_input_gpu::Ptr{Cfloat}
    loss_gpu::Ptr{Cfloat}
    delta_gpu::Ptr{Cfloat}
    cos_sim_gpu::Ptr{Cfloat}
    rand_gpu::Ptr{Cfloat}
    drop_blocks_scale::Ptr{Cfloat}
    drop_blocks_scale_gpu::Ptr{Cfloat}
    squared_gpu::Ptr{Cfloat}
    norms_gpu::Ptr{Cfloat}
    gt_gpu::Ptr{Cfloat}
    a_avg_gpu::Ptr{Cfloat}
    input_sizes_gpu::Ptr{Cint}
    layers_output_gpu::Ptr{Ptr{Cfloat}}
    layers_delta_gpu::Ptr{Ptr{Cfloat}}
    srcTensorDesc::Ptr{Cvoid}
    dstTensorDesc::Ptr{Cvoid}
    srcTensorDesc16::Ptr{Cvoid}
    dstTensorDesc16::Ptr{Cvoid}
    dsrcTensorDesc::Ptr{Cvoid}
    ddstTensorDesc::Ptr{Cvoid}
    dsrcTensorDesc16::Ptr{Cvoid}
    ddstTensorDesc16::Ptr{Cvoid}
    normTensorDesc::Ptr{Cvoid}
    normDstTensorDesc::Ptr{Cvoid}
    normDstTensorDescF16::Ptr{Cvoid}
    weightDesc::Ptr{Cvoid}
    weightDesc16::Ptr{Cvoid}
    dweightDesc::Ptr{Cvoid}
    dweightDesc16::Ptr{Cvoid}
    convDesc::Ptr{Cvoid}
    fw_algo::UNUSED_ENUM_TYPE
    fw_algo16::UNUSED_ENUM_TYPE
    bd_algo::UNUSED_ENUM_TYPE
    bd_algo16::UNUSED_ENUM_TYPE
    bf_algo::UNUSED_ENUM_TYPE
    bf_algo16::UNUSED_ENUM_TYPE
    poolingDesc::Ptr{Cvoid}
end

@cenum learning_rate_policy::UInt32 begin
    CONSTANT = 0
    STEP = 1
    EXP = 2
    POLY = 3
    STEPS = 4
    SIG = 5
    RANDOM = 6
    SGDR = 7
end

struct network
    n::Cint
    batch::Cint
    seen::Ptr{UInt64}
    badlabels_reject_threshold::Ptr{Cfloat}
    delta_rolling_max::Ptr{Cfloat}
    delta_rolling_avg::Ptr{Cfloat}
    delta_rolling_std::Ptr{Cfloat}
    weights_reject_freq::Cint
    equidistant_point::Cint
    badlabels_rejection_percentage::Cfloat
    num_sigmas_reject_badlabels::Cfloat
    ema_alpha::Cfloat
    cur_iteration::Ptr{Cint}
    loss_scale::Cfloat
    t::Ptr{Cint}
    epoch::Cfloat
    subdivisions::Cint
    layers::Ptr{layer}
    output::Ptr{Cfloat}
    policy::learning_rate_policy
    benchmark_layers::Cint
    total_bbox::Ptr{Cint}
    rewritten_bbox::Ptr{Cint}
    learning_rate::Cfloat
    learning_rate_min::Cfloat
    learning_rate_max::Cfloat
    batches_per_cycle::Cint
    batches_cycle_mult::Cint
    momentum::Cfloat
    decay::Cfloat
    gamma::Cfloat
    scale::Cfloat
    power::Cfloat
    time_steps::Cint
    step::Cint
    max_batches::Cint
    num_boxes::Cint
    train_images_num::Cint
    seq_scales::Ptr{Cfloat}
    scales::Ptr{Cfloat}
    steps::Ptr{Cint}
    num_steps::Cint
    burn_in::Cint
    cudnn_half::Cint
    adam::Cint
    B1::Cfloat
    B2::Cfloat
    eps::Cfloat
    inputs::Cint
    outputs::Cint
    truths::Cint
    notruth::Cint
    h::Cint
    w::Cint
    c::Cint
    max_crop::Cint
    min_crop::Cint
    max_ratio::Cfloat
    min_ratio::Cfloat
    center::Cint
    flip::Cint
    gaussian_noise::Cint
    blur::Cint
    mixup::Cint
    label_smooth_eps::Cfloat
    resize_step::Cint
    attention::Cint
    adversarial::Cint
    adversarial_lr::Cfloat
    max_chart_loss::Cfloat
    letter_box::Cint
    mosaic_bound::Cint
    contrastive::Cint
    contrastive_jit_flip::Cint
    contrastive_color::Cint
    unsupervised::Cint
    angle::Cfloat
    aspect::Cfloat
    exposure::Cfloat
    saturation::Cfloat
    hue::Cfloat
    random::Cint
    track::Cint
    augment_speed::Cint
    sequential_subdivisions::Cint
    init_sequential_subdivisions::Cint
    current_subdivision::Cint
    try_fix_nan::Cint
    gpu_index::Cint
    hierarchy::Ptr{tree}
    input::Ptr{Cfloat}
    truth::Ptr{Cfloat}
    delta::Ptr{Cfloat}
    workspace::Ptr{Cfloat}
    train::Cint
    index::Cint
    cost::Ptr{Cfloat}
    clip::Cfloat
    delta_gpu::Ptr{Cfloat}
    output_gpu::Ptr{Cfloat}
    input_state_gpu::Ptr{Cfloat}
    input_pinned_cpu::Ptr{Cfloat}
    input_pinned_cpu_flag::Cint
    input_gpu::Ptr{Ptr{Cfloat}}
    truth_gpu::Ptr{Ptr{Cfloat}}
    input16_gpu::Ptr{Ptr{Cfloat}}
    output16_gpu::Ptr{Ptr{Cfloat}}
    max_input16_size::Ptr{Csize_t}
    max_output16_size::Ptr{Csize_t}
    wait_stream::Cint
    cuda_graph::Ptr{Cvoid}
    cuda_graph_exec::Ptr{Cvoid}
    use_cuda_graph::Cint
    cuda_graph_ready::Ptr{Cint}
    global_delta_gpu::Ptr{Cfloat}
    state_delta_gpu::Ptr{Cfloat}
    max_delta_gpu_size::Csize_t
    optimized_memory::Cint
    dynamic_minibatch::Cint
    workspace_size_limit::Csize_t
end

struct network_state
    truth::Ptr{Cfloat}
    input::Ptr{Cfloat}
    delta::Ptr{Cfloat}
    workspace::Ptr{Cfloat}
    train::Cint
    index::Cint
    net::network
end

struct image
    w::Cint
    h::Cint
    c::Cint
    data::Ptr{Cfloat}
end

struct box
    x::Cfloat
    y::Cfloat
    w::Cfloat
    h::Cfloat
end

struct detection
    bbox::box
    classes::Cint
    best_class_idx::Cint
    prob::Ptr{Cfloat}
    mask::Ptr{Cfloat}
    objectness::Cfloat
    sort_class::Cint
    uc::Ptr{Cfloat}
    points::Cint
    embeddings::Ptr{Cfloat}
    embedding_size::Cint
    sim::Cfloat
    track_id::Cint
end

struct matrix
    rows::Cint
    cols::Cint
    vals::Ptr{Ptr{Cfloat}}
end

struct data
    w::Cint
    h::Cint
    X::matrix
    y::matrix
    shallow::Cint
    num_boxes::Ptr{Cint}
    boxes::Ptr{Ptr{box}}
end

@cenum data_type::UInt32 begin
    CLASSIFICATION_DATA = 0
    DETECTION_DATA = 1
    CAPTCHA_DATA = 2
    REGION_DATA = 3
    IMAGE_DATA = 4
    COMPARE_DATA = 5
    WRITING_DATA = 6
    SWAG_DATA = 7
    TAG_DATA = 8
    OLD_CLASSIFICATION_DATA = 9
    STUDY_DATA = 10
    DET_DATA = 11
    SUPER_DATA = 12
    LETTERBOX_DATA = 13
    REGRESSION_DATA = 14
    SEGMENTATION_DATA = 15
    INSTANCE_DATA = 16
    ISEG_DATA = 17
end

struct load_args
    threads::Cint
    paths::Ptr{Cstring}
    path::Cstring
    n::Cint
    m::Cint
    labels::Ptr{Cstring}
    h::Cint
    w::Cint
    c::Cint
    out_w::Cint
    out_h::Cint
    nh::Cint
    nw::Cint
    num_boxes::Cint
    truth_size::Cint
    min::Cint
    max::Cint
    size::Cint
    classes::Cint
    background::Cint
    scale::Cint
    center::Cint
    coords::Cint
    mini_batch::Cint
    track::Cint
    augment_speed::Cint
    letter_box::Cint
    mosaic_bound::Cint
    show_imgs::Cint
    dontuse_opencv::Cint
    contrastive::Cint
    contrastive_jit_flip::Cint
    contrastive_color::Cint
    jitter::Cfloat
    resize::Cfloat
    flip::Cint
    gaussian_noise::Cint
    blur::Cint
    mixup::Cint
    label_smooth_eps::Cfloat
    angle::Cfloat
    aspect::Cfloat
    saturation::Cfloat
    exposure::Cfloat
    hue::Cfloat
    d::Ptr{data}
    im::Ptr{image}
    resized::Ptr{image}
    type::data_type
    hierarchy::Ptr{tree}
end

struct metadata
    classes::Cint
    names::Ptr{Cstring}
end

@cenum IMTYPE::UInt32 begin
    PNG = 0
    BMP = 1
    TGA = 2
    JPG = 3
end

@cenum BINARY_ACTIVATION::UInt32 begin
    MULT = 0
    ADD = 1
    SUB = 2
    DIV = 3
end

struct update_args
    batch::Cint
    learning_rate::Cfloat
    momentum::Cfloat
    decay::Cfloat
    adam::Cint
    B1::Cfloat
    B2::Cfloat
    eps::Cfloat
    t::Cint
end

struct boxabs
    left::Cfloat
    right::Cfloat
    top::Cfloat
    bot::Cfloat
end

struct dxrep
    dt::Cfloat
    db::Cfloat
    dl::Cfloat
    dr::Cfloat
end

struct ious
    iou::Cfloat
    giou::Cfloat
    diou::Cfloat
    ciou::Cfloat
    dx_iou::dxrep
    dx_giou::dxrep
end

struct det_num_pair
    num::Cint
    dets::Ptr{detection}
end

const pdet_num_pair = Ptr{det_num_pair}

struct box_label
    id::Cint
    track_id::Cint
    x::Cfloat
    y::Cfloat
    w::Cfloat
    h::Cfloat
    left::Cfloat
    right::Cfloat
    top::Cfloat
    bottom::Cfloat
end

function _load_network(cfg, weights, clear)
    ccall((:load_network, libdarknet), Ptr{network}, (Cstring, Cstring, Cint), cfg, weights, clear)
end

function _load_network_custom(cfg, weights, clear, batch)
    ccall((:load_network_custom, libdarknet), Ptr{network}, (Cstring, Cstring, Cint, Cint), cfg, weights, clear, batch)
end

function free_network(net)
    ccall((:free_network, libdarknet), Cvoid, (network,), net)
end

function free_network_ptr(net)
    ccall((:free_network_ptr, libdarknet), Cvoid, (Ptr{network},), net)
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

function diounms_sort(dets, total, classes, thresh, nms_kind, beta1)
    ccall((:diounms_sort, libdarknet), Cvoid, (Ptr{detection}, Cint, Cint, Cfloat, NMS_KIND, Cfloat), dets, total, classes, thresh, nms_kind, beta1)
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

function network_predict_batch(net, im, batch_size, w, h, thresh, hier, map, relative, letter)
    ccall((:network_predict_batch, libdarknet), Ptr{det_num_pair}, (Ptr{network}, image, Cint, Cint, Cint, Cfloat, Cfloat, Ptr{Cint}, Cint, Cint), net, im, batch_size, w, h, thresh, hier, map, relative, letter)
end

function free_detections(dets, n)
    ccall((:free_detections, libdarknet), Cvoid, (Ptr{detection}, Cint), dets, n)
end

function free_batch_detections(det_num_pairs, n)
    ccall((:free_batch_detections, libdarknet), Cvoid, (Ptr{det_num_pair}, Cint), det_num_pairs, n)
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

function network_predict_image_letterbox(net, im)
    ccall((:network_predict_image_letterbox, libdarknet), Ptr{Cfloat}, (Ptr{network}, image), net, im)
end

function validate_detector_map(datacfg, cfgfile, weightfile, thresh_calc_avg_iou, iou_thresh, map_points, letter_box, existing_net)
    ccall((:validate_detector_map, libdarknet), Cfloat, (Cstring, Cstring, Cstring, Cfloat, Cfloat, Cint, Cint, Ptr{network}), datacfg, cfgfile, weightfile, thresh_calc_avg_iou, iou_thresh, map_points, letter_box, existing_net)
end

function train_detector(datacfg, cfgfile, weightfile, gpus, ngpus, clear, dont_show, calc_map, thresh, iou_thresh, mjpeg_port, show_imgs, benchmark_layers, chart_path)
    ccall((:train_detector, libdarknet), Cvoid, (Cstring, Cstring, Cstring, Ptr{Cint}, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cint, Cint, Cint, Cstring), datacfg, cfgfile, weightfile, gpus, ngpus, clear, dont_show, calc_map, thresh, iou_thresh, mjpeg_port, show_imgs, benchmark_layers, chart_path)
end

function test_detector(datacfg, cfgfile, weightfile, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers)
    ccall((:test_detector, libdarknet), Cvoid, (Cstring, Cstring, Cstring, Cstring, Cfloat, Cfloat, Cint, Cint, Cint, Cstring, Cint, Cint), datacfg, cfgfile, weightfile, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers)
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

function make_image_red(im)
    ccall((:make_image_red, libdarknet), Cvoid, (image,), im)
end

function make_attention_image(img_size, original_delta_cpu, original_input_cpu, w, h, c, alpha)
    ccall((:make_attention_image, libdarknet), image, (Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint, Cfloat), img_size, original_delta_cpu, original_input_cpu, w, h, c, alpha)
end

function resize_image(im, w, h)
    ccall((:resize_image, libdarknet), image, (image, Cint, Cint), im, w, h)
end

function quantize_image(im)
    ccall((:quantize_image, libdarknet), Cvoid, (image,), im)
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

function crop_image(im, dx, dy, w, h)
    ccall((:crop_image, libdarknet), image, (image, Cint, Cint, Cint, Cint), im, dx, dy, w, h)
end

function resize_min(im, min)
    ccall((:resize_min, libdarknet), image, (image, Cint), im, min)
end

function free_layer_custom(l, keep_cudnn_desc)
    ccall((:free_layer_custom, libdarknet), Cvoid, (layer, Cint), l, keep_cudnn_desc)
end

function free_layer(l)
    ccall((:free_layer, libdarknet), Cvoid, (layer,), l)
end

function free_data(d)
    ccall((:free_data, libdarknet), Cvoid, (data,), d)
end

function load_data(args)
    ccall((:load_data, libdarknet), pthread_t, (load_args,), args)
end

function free_load_threads(ptr)
    ccall((:free_load_threads, libdarknet), Cvoid, (Ptr{Cvoid},), ptr)
end

function load_data_in_thread(args)
    ccall((:load_data_in_thread, libdarknet), pthread_t, (load_args,), args)
end

function load_thread(ptr)
    ccall((:load_thread, libdarknet), Ptr{Cvoid}, (Ptr{Cvoid},), ptr)
end

function cuda_pull_array(x_gpu, x, n)
    ccall((:cuda_pull_array, libdarknet), Cvoid, (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t), x_gpu, x, n)
end

function cuda_pull_array_async(x_gpu, x, n)
    ccall((:cuda_pull_array_async, libdarknet), Cvoid, (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t), x_gpu, x, n)
end

function cuda_set_device(n)
    ccall((:cuda_set_device, libdarknet), Cvoid, (Cint,), n)
end

# no prototype is found for this function at darknet.h:1086:15, please use with caution
function cuda_get_context()
    ccall((:cuda_get_context, libdarknet), Ptr{Cvoid}, ())
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

# no prototype is found for this function at darknet.h:1100:14, please use with caution
function delete_json_sender()
    ccall((:delete_json_sender, libdarknet), Cvoid, ())
end

function send_json_custom(send_buf, port, timeout)
    ccall((:send_json_custom, libdarknet), Cvoid, (Cstring, Cint, Cint), send_buf, port, timeout)
end

# no prototype is found for this function at darknet.h:1102:16, please use with caution
function get_time_point()
    ccall((:get_time_point, libdarknet), Cdouble, ())
end

# no prototype is found for this function at darknet.h:1103:6, please use with caution
function start_timer()
    ccall((:start_timer, libdarknet), Cvoid, ())
end

# no prototype is found for this function at darknet.h:1104:6, please use with caution
function stop_timer()
    ccall((:stop_timer, libdarknet), Cvoid, ())
end

# no prototype is found for this function at darknet.h:1105:8, please use with caution
function get_time()
    ccall((:get_time, libdarknet), Cdouble, ())
end

# no prototype is found for this function at darknet.h:1106:6, please use with caution
function stop_timer_and_show()
    ccall((:stop_timer_and_show, libdarknet), Cvoid, ())
end

function stop_timer_and_show_name(name)
    ccall((:stop_timer_and_show_name, libdarknet), Cvoid, (Cstring,), name)
end

# no prototype is found for this function at darknet.h:1108:6, please use with caution
function show_total_time()
    ccall((:show_total_time, libdarknet), Cvoid, ())
end

function set_track_id(new_dets, new_dets_num, thresh, sim_thresh, track_ciou_norm, deque_size, dets_for_track, dets_for_show)
    ccall((:set_track_id, libdarknet), Cvoid, (Ptr{detection}, Cint, Cfloat, Cfloat, Cfloat, Cint, Cint, Cint), new_dets, new_dets_num, thresh, sim_thresh, track_ciou_norm, deque_size, dets_for_track, dets_for_show)
end

function fill_remaining_id(new_dets, new_dets_num, new_track_id, thresh)
    ccall((:fill_remaining_id, libdarknet), Cint, (Ptr{detection}, Cint, Cint, Cfloat), new_dets, new_dets_num, new_track_id, thresh)
end

# no prototype is found for this function at darknet.h:1115:14, please use with caution
function init_cpu()
    ccall((:init_cpu, libdarknet), Cvoid, ())
end

const SECRET_NUM = -1234

