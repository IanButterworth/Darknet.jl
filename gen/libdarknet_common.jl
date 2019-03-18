# Automatically generated using Clang.jl wrap_c


const NFRAMES = 3
const SECRET_NUM = -1234

@cenum(LAYER_TYPE,
    CONVOLUTIONAL = 0,
    DECONVOLUTIONAL = 1,
    CONNECTED = 2,
    MAXPOOL = 3,
    SOFTMAX = 4,
    DETECTION = 5,
    DROPOUT = 6,
    CROP = 7,
    ROUTE = 8,
    COST = 9,
    NORMALIZATION = 10,
    AVGPOOL = 11,
    LOCAL = 12,
    SHORTCUT = 13,
    ACTIVE = 14,
    RNN = 15,
    GRU = 16,
    LSTM = 17,
    CRNN = 18,
    BATCHNORM = 19,
    NETWORK = 20,
    XNOR = 21,
    REGION = 22,
    YOLO = 23,
    ISEG = 24,
    REORG = 25,
    REORG_OLD = 26,
    UPSAMPLE = 27,
    LOGXENT = 28,
    L2NORM = 29,
    BLANK = 30,
)
@cenum(ACTIVATION,
    LOGISTIC = 0,
    RELU = 1,
    RELIE = 2,
    LINEAR = 3,
    RAMP = 4,
    TANH = 5,
    PLSE = 6,
    LEAKY = 7,
    ELU = 8,
    LOGGY = 9,
    STAIR = 10,
    HARDTAN = 11,
    LHTAN = 12,
    SELU = 13,
)
@cenum(COST_TYPE,
    SSE = 0,
    MASKED = 1,
    L1 = 2,
    SEG = 3,
    SMOOTH = 4,
    WGAN = 5,
)

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
    cost_type::COST_TYPE
    forward::Ptr{Cvoid}
    backward::Ptr{Cvoid}
    update::Ptr{Cvoid}
    forward_gpu::Ptr{Cvoid}
    backward_gpu::Ptr{Cvoid}
    update_gpu::Ptr{Cvoid}
    batch_normalize::Cint
    shortcut::Cint
    batch::Cint
    forced::Cint
    flipped::Cint
    inputs::Cint
    outputs::Cint
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
    groups::Cint
    size::Cint
    side::Cint
    stride::Cint
    reverse::Cint
    flatten::Cint
    spatial::Cint
    pad::Cint
    sqrt::Cint
    flip::Cint
    index::Cint
    binary::Cint
    xnor::Cint
    use_bin_output::Cint
    steps::Cint
    hidden::Cint
    truth::Cint
    smooth::Cfloat
    dot::Cfloat
    angle::Cfloat
    jitter::Cfloat
    saturation::Cfloat
    exposure::Cfloat
    shift::Cfloat
    ratio::Cfloat
    learning_rate_scale::Cfloat
    clip::Cfloat
    focal_loss::Cint
    noloss::Cint
    softmax::Cint
    classes::Cint
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
    random::Cint
    ignore_thresh::Cfloat
    truth_thresh::Cfloat
    thresh::Cfloat
    focus::Cfloat
    classfix::Cint
    absolute::Cint
    onlyforward::Cint
    stopbackward::Cint
    dontload::Cint
    dontsave::Cint
    dontloadscales::Cint
    numload::Cint
    temperature::Cfloat
    probability::Cfloat
    scale::Cfloat
    cweights::Cstring
    indexes::Ptr{Cint}
    input_layers::Ptr{Cint}
    input_sizes::Ptr{Cint}
    map::Ptr{Cint}
    counts::Ptr{Cint}
    sums::Ptr{Ptr{Cfloat}}
    rand::Ptr{Cfloat}
    cost::Ptr{Cfloat}
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
    weights::Ptr{Cfloat}
    weight_updates::Ptr{Cfloat}
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
    uf::Ptr{layer}
    wf::Ptr{layer}
    ui::Ptr{layer}
    wi::Ptr{layer}
    ug::Ptr{layer}
    wg::Ptr{layer}
    softmax_tree::Ptr{tree}
    workspace_size::Cint
end

@cenum(learning_rate_policy,
    CONSTANT = 0,
    STEP = 1,
    EXP = 2,
    POLY = 3,
    STEPS = 4,
    SIG = 5,
    RANDOM = 6,
)

struct network
    n::Cint
    batch::Cint
    seen::Ptr{UInt64}
    t::Ptr{Cint}
    epoch::Cfloat
    subdivisions::Cint
    layers::Ptr{layer}
    output::Ptr{Cfloat}
    policy::learning_rate_policy
    learning_rate::Cfloat
    momentum::Cfloat
    decay::Cfloat
    gamma::Cfloat
    scale::Cfloat
    power::Cfloat
    time_steps::Cint
    step::Cint
    max_batches::Cint
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
    angle::Cfloat
    aspect::Cfloat
    exposure::Cfloat
    saturation::Cfloat
    hue::Cfloat
    random::Cint
    track::Cint
    augment_speed::Cint
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
    prob::Ptr{Cfloat}
    mask::Ptr{Cfloat}
    objectness::Cfloat
    sort_class::Cint
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

@cenum(data_type,
    CLASSIFICATION_DATA = 0,
    DETECTION_DATA = 1,
    CAPTCHA_DATA = 2,
    REGION_DATA = 3,
    IMAGE_DATA = 4,
    COMPARE_DATA = 5,
    WRITING_DATA = 6,
    SWAG_DATA = 7,
    TAG_DATA = 8,
    OLD_CLASSIFICATION_DATA = 9,
    STUDY_DATA = 10,
    DET_DATA = 11,
    SUPER_DATA = 12,
    LETTERBOX_DATA = 13,
    REGRESSION_DATA = 14,
    SEGMENTATION_DATA = 15,
    INSTANCE_DATA = 16,
    ISEG_DATA = 17,
)

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
    jitter::Cfloat
    flip::Cint
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

@cenum(IMTYPE,
    PNG = 0,
    BMP = 1,
    TGA = 2,
    JPG = 3,
)
@cenum(BINARY_ACTIVATION,
    MULT = 0,
    ADD = 1,
    SUB = 2,
    DIV = 3,
)

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

struct box_label
    id::Cint
    x::Cfloat
    y::Cfloat
    w::Cfloat
    h::Cfloat
    left::Cfloat
    right::Cfloat
    top::Cfloat
    bottom::Cfloat
end
