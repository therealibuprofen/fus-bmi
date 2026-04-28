function model = train_classifier_cnn_twohead(trainData, trainLabels, varargin)
% train_classifier_cnn_twohead  Train a shared-backbone two-head CNN decoder.
%
% The backbone is shared across tasks, then two independent classification
% heads predict horizontal and vertical classes respectively.

p = inputParser;
p.addParameter('input_size', []);
p.addParameter('num_filters', [8 16 32]);
p.addParameter('temporal_kernel_size', 3);
p.addParameter('spatial_kernel_size', [3 3]);
p.addParameter('dropout', 0.2);
p.addParameter('batch_norm', true);
p.addParameter('max_epochs', 60);
p.addParameter('mini_batch_size', 8);
p.addParameter('initial_learn_rate', 1e-3);
p.addParameter('l2_regularization', 1e-4);
p.addParameter('gradient_threshold', 1.0);
p.addParameter('learn_rate_drop_factor', 0.5);
p.addParameter('learn_rate_drop_period', 15);
p.addParameter('validation_ratio', 0.2);
p.addParameter('validation_patience', 8); %#ok<*NVREPL>
p.addParameter('output_network', 'best-validation-loss');
p.addParameter('verbose', false, @islogical);
p.addParameter('random_seed', 42);
p.addParameter('class_weight_mode', 'balanced');
p.addParameter('normalize_in_network', true, @islogical);
p.addParameter('execution_environment', 'gpu');
p.addParameter('initial_model', []);
p.addParameter('fine_tune_epochs', 3);
p.KeepUnmatched = true;
p.parse(varargin{:});
cfg = p.Results;

rng(cfg.random_seed);

if size(trainLabels, 2) ~= 2
    error('Two-head CNN requires trainLabels to be N x 2.');
end

[X, inputSize, nSamples] = normalize_input_shape(trainData, cfg.input_size);
y = double(trainLabels);
if size(y, 1) ~= nSamples
    error('训练标签数量(%d)与样本数量(%d)不一致。', size(y, 1), nSamples);
end

validMask = squeeze(all(all(all(all(isfinite(X), 1), 2), 3), 4));
validMask = reshape(validMask, [], 1) & all(~isnan(y), 2);
X = X(:, :, :, :, validMask);
y = y(validMask, :);

horzVals = unique(y(:, 1), 'sorted');
vertVals = unique(y(:, 2), 'sorted');
yH = encode_labels(y(:, 1), horzVals);
yV = encode_labels(y(:, 2), vertVals);

[idxTrain, idxVal, hasValidation] = split_indices_multitask(y, cfg.validation_ratio, cfg.random_seed);
XTrainRaw = X(:, :, :, :, idxTrain);
yTrainH = yH(idxTrain);
yTrainV = yV(idxTrain);
if hasValidation
    XValRaw = X(:, :, :, :, idxVal);
    yValH = yH(idxVal);
    yValV = yV(idxVal);
else
    XValRaw = [];
    yValH = [];
    yValV = [];
end

mu = mean(XTrainRaw, 5, 'omitnan');
sigma = std(XTrainRaw, 0, 5, 'omitnan');
sigma(sigma == 0) = 1;
if cfg.normalize_in_network
    XTrain = XTrainRaw;
    XVal = XValRaw;
else
    XTrain = apply_zscore_samples(XTrainRaw, mu, sigma);
    XVal = apply_zscore_samples(XValRaw, mu, sigma);
end

dlnet = initialize_network(cfg, inputSize, numel(horzVals), numel(vertVals), mu, sigma);
warmStartUsed = false;
if has_valid_initial_model(cfg.initial_model, 'CNN_TWOHEAD')
    dlnet = cfg.initial_model.dlnet;
    warmStartUsed = true;
end

weightsH = resolve_class_weights(yTrainH, numel(horzVals), cfg.class_weight_mode);
weightsV = resolve_class_weights(yTrainV, numel(vertVals), cfg.class_weight_mode);

[dlnet, bestMetric] = train_network( ...
    dlnet, XTrain, yTrainH, yTrainV, XVal, yValH, yValV, weightsH, weightsV, cfg);

model = struct();
model.method = 'CNN_TWOHEAD';
model.baseMethod = 'CNN';
model.dlnet = dlnet;
model.inputSize = inputSize;
model.mu = mu;
model.sigma = sigma;
model.normalizeInNetwork = cfg.normalize_in_network;
model.horzClassValues = horzVals;
model.vertClassValues = vertVals;
model.config = cfg;
model.warmStartUsed = warmStartUsed;
model.trainingMetric = bestMetric;
end

function dlnet = initialize_network(cfg, inputSize, nHorz, nVert, mu, sigma)
numFilters = cfg.num_filters(:)';
if isempty(numFilters)
    error('num_filters must be non-empty.');
end

lg = layerGraph();
lg = addLayers(lg, image3dInputLayer([inputSize 1], 'Name', 'input', 'Normalization', 'none'));

prev = 'input';
if cfg.normalize_in_network
    lg = addLayers(lg, functionLayer(@(X) apply_fixed_zscore_dl(X, mu, sigma), ...
        'Name', 'fixed_zscore', 'Formattable', true));
    lg = connectLayers(lg, prev, 'fixed_zscore');
    prev = 'fixed_zscore';
end

for i = 1:numel(numFilters)
    block = [
        convolution3dLayer([cfg.spatial_kernel_size(:)' 1], numFilters(i), ...
            'Padding', 'same', 'Name', sprintf('spatial_conv%d', i))
        reluLayer('Name', sprintf('spatial_relu%d', i))
        convolution3dLayer([1 1 cfg.temporal_kernel_size], numFilters(i), ...
            'Padding', 'same', 'Name', sprintf('temporal_conv%d', i))
        reluLayer('Name', sprintf('temporal_relu%d', i))
        maxPooling3dLayer([2 2 1], 'Stride', [2 2 1], 'Name', sprintf('pool%d', i))
    ];
    if cfg.batch_norm
        block = [
            convolution3dLayer([cfg.spatial_kernel_size(:)' 1], numFilters(i), ...
                'Padding', 'same', 'Name', sprintf('spatial_conv%d', i))
            batchNormalizationLayer('Name', sprintf('spatial_bn%d', i))
            reluLayer('Name', sprintf('spatial_relu%d', i))
            convolution3dLayer([1 1 cfg.temporal_kernel_size], numFilters(i), ...
                'Padding', 'same', 'Name', sprintf('temporal_conv%d', i))
            batchNormalizationLayer('Name', sprintf('temporal_bn%d', i))
            reluLayer('Name', sprintf('temporal_relu%d', i))
            maxPooling3dLayer([2 2 1], 'Stride', [2 2 1], 'Name', sprintf('pool%d', i))
        ];
    end
    if cfg.dropout > 0
        block = [block
            dropoutLayer(cfg.dropout, 'Name', sprintf('drop%d', i))];
    end
    lg = addLayers(lg, block);
    lg = connectLayers(lg, prev, sprintf('spatial_conv%d', i));
    if cfg.dropout > 0
        prev = sprintf('drop%d', i);
    else
        prev = sprintf('pool%d', i);
    end
end

sharedLayers = [
    fullyConnectedLayer(max(32, numFilters(end)), 'Name', 'shared_fc')
    reluLayer('Name', 'shared_relu')
];
lg = addLayers(lg, sharedLayers);
lg = connectLayers(lg, prev, 'shared_fc');

if cfg.dropout > 0
    lg = addLayers(lg, dropoutLayer(cfg.dropout, 'Name', 'shared_drop'));
    lg = connectLayers(lg, 'shared_relu', 'shared_drop');
    prevHead = 'shared_drop';
else
    prevHead = 'shared_relu';
end

headH = fullyConnectedLayer(nHorz, 'Name', 'fc_horz');
headV = fullyConnectedLayer(nVert, 'Name', 'fc_vert');
lg = addLayers(lg, headH);
lg = addLayers(lg, headV);
lg = connectLayers(lg, prevHead, 'fc_horz');
lg = connectLayers(lg, prevHead, 'fc_vert');

dlnet = dlnetwork(lg);
end

function [dlnet, bestMetric] = train_network(dlnet, XTrain, yTrainH, yTrainV, XVal, yValH, yValV, weightsH, weightsV, cfg)
numTrain = size(XTrain, 5);
miniBatchSize = min(cfg.mini_batch_size, max(numTrain, 1));
numIterationsPerEpoch = max(1, ceil(numTrain / miniBatchSize));
trailingAvg = [];
trailingAvgSq = [];
bestDlnet = dlnet;
bestMetric = inf;
bestEpoch = 0;
iteration = 0;
learnRate = cfg.initial_learn_rate;
patienceCounter = 0;
maxEpochs = resolve_max_epochs(cfg);

for epoch = 1:maxEpochs
    order = randperm(numTrain);
    for iter = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        batchIdx = order((iter - 1) * miniBatchSize + 1:min(iter * miniBatchSize, numTrain));
        dlX = dlarray(XTrain(:, :, :, :, batchIdx), 'SSSCB');
        batchYH = yTrainH(batchIdx);
        batchYV = yTrainV(batchIdx);
        if use_gpu(cfg.execution_environment)
            dlX = gpuArray(dlX);
        end

        [gradients, state, loss] = dlfeval(@model_gradients, dlnet, dlX, batchYH, batchYV, weightsH, weightsV);
        dlnet.State = state;
        gradients = threshold_gradients(gradients, cfg.gradient_threshold);
        [dlnet, trailingAvg, trailingAvgSq] = adamupdate( ...
            dlnet, gradients, trailingAvg, trailingAvgSq, iteration, learnRate);
    end

    if cfg.learn_rate_drop_period > 0 && mod(epoch, cfg.learn_rate_drop_period) == 0
        learnRate = learnRate * cfg.learn_rate_drop_factor;
    end

    metric = gather_numeric_scalar(loss);
    if ~isempty(XVal)
        metric = evaluate_loss(dlnet, XVal, yValH, yValV, weightsH, weightsV, cfg);
    end

    if metric < bestMetric
        bestMetric = metric;
        bestDlnet = dlnet;
        bestEpoch = epoch;
        patienceCounter = 0;
    else
        patienceCounter = patienceCounter + 1;
    end

    if cfg.verbose
        fprintf('[train_classifier_cnn_twohead] epoch %d/%d, metric=%.4f\n', epoch, maxEpochs, metric);
    end

    if ~isempty(XVal) && cfg.validation_patience > 0 && patienceCounter >= cfg.validation_patience
        break;
    end
end

if isempty(XVal) || strcmpi(cfg.output_network, 'last-iteration')
    return;
end

if bestEpoch > 0
    dlnet = bestDlnet;
end
end

function metric = evaluate_loss(dlnet, XVal, yValH, yValV, weightsH, weightsV, cfg)
miniBatchSize = min(cfg.mini_batch_size, max(size(XVal, 5), 1));
numVal = size(XVal, 5);
numBatches = max(1, ceil(numVal / miniBatchSize));
losses = zeros(numBatches, 1);
for i = 1:numBatches
    batchIdx = (i - 1) * miniBatchSize + 1:min(i * miniBatchSize, numVal);
    dlX = dlarray(XVal(:, :, :, :, batchIdx), 'SSSCB');
    if use_gpu(cfg.execution_environment)
        dlX = gpuArray(dlX);
    end
    [losses(i), ~] = dlfeval(@model_loss, dlnet, dlX, yValH(batchIdx), yValV(batchIdx), weightsH, weightsV);
    losses(i) = gather_numeric_scalar(losses(i));
end
metric = mean(losses);
end

function [gradients, state, loss] = model_gradients(dlnet, dlX, yH, yV, weightsH, weightsV)
[dlYH, dlYV, state] = forward(dlnet, dlX, 'Outputs', {'fc_horz', 'fc_vert'});
lossH = weighted_crossentropy(dlYH, yH, weightsH);
lossV = weighted_crossentropy(dlYV, yV, weightsV);
loss = 0.5 * (lossH + lossV);
gradients = dlgradient(loss, dlnet.Learnables);
end

function [loss, state] = model_loss(dlnet, dlX, yH, yV, weightsH, weightsV)
[dlYH, dlYV, state] = forward(dlnet, dlX, 'Outputs', {'fc_horz', 'fc_vert'});
lossH = weighted_crossentropy(dlYH, yH, weightsH);
lossV = weighted_crossentropy(dlYV, yV, weightsV);
loss = 0.5 * (lossH + lossV);
end

function loss = weighted_crossentropy(logits, labels, classWeights)
logits = logits_to_class_batch(logits);
probs = softmax(logits, 'DataFormat', 'CB');
nClasses = size(probs, 1);
nObs = size(probs, 2);
targets = zeros(nClasses, nObs, 'single');
targets(sub2ind([nClasses, nObs], double(labels(:))', 1:nObs)) = 1;
targets = dlarray(cast(targets, 'like', probs));
sampleWeights = reshape(classWeights(double(labels(:))), 1, []);
sampleWeights = dlarray(cast(sampleWeights, 'like', probs));
lossPerObs = -sum(targets .* log(probs + eps('single')), 1);
loss = mean(lossPerObs .* sampleWeights, 'all');
end

function logits = logits_to_class_batch(logits)
logits = stripdims(logits);
logits = squeeze(logits);
if isvector(logits)
    logits = reshape(logits, [], 1);
end
end

function yIdx = encode_labels(y, classVals)
yIdx = zeros(size(y));
for i = 1:numel(classVals)
    yIdx(y == classVals(i)) = i;
end
end

function [idxTrain, idxVal, hasValidation] = split_indices_multitask(labels, validationRatio, seed)
n = size(labels, 1);
hasValidation = n >= 10 && validationRatio > 0;
idxTrain = true(n, 1);
idxVal = false(n, 1);
if ~hasValidation
    return;
end

combo = strcat(string(labels(:, 1)), "_", string(labels(:, 2)));
try
    rng(seed);
    cv = cvpartition(categorical(combo), 'HoldOut', validationRatio);
    idxTrain = training(cv);
    idxVal = test(cv);
catch
    rng(seed);
    order = randperm(n);
    nVal = max(1, round(validationRatio * n));
    idxVal(order(1:nVal)) = true;
    idxTrain = ~idxVal;
end
end

function weights = resolve_class_weights(labelIdx, nClasses, mode)
switch lower(string(mode))
    case "balanced"
        counts = accumarray(double(labelIdx(:)), 1, [nClasses 1])';
        imbalanceRatio = max(counts) / max(min(counts(counts > 0)), 1);
        if imbalanceRatio >= 3
            weights = effective_num_weights(counts, 0.999);
        else
            weights = median(counts(counts > 0)) ./ max(counts, 1);
            weights = weights / mean(weights);
        end
    otherwise
        weights = ones(1, nClasses, 'single');
end
weights = single(weights(:)');
end

function maxEpochs = resolve_max_epochs(cfg)
maxEpochs = cfg.max_epochs;
if has_valid_initial_model(cfg.initial_model, 'CNN_TWOHEAD')
    maxEpochs = min(cfg.max_epochs, cfg.fine_tune_epochs);
end
end

function gradients = threshold_gradients(gradients, threshold)
if isempty(threshold) || threshold <= 0
    return;
end
gradients = dlupdate(@(g) clip_gradient(g, threshold), gradients);
end

function g = clip_gradient(g, threshold)
if isempty(g)
    return;
end
g = max(min(g, threshold), -threshold);
end

function tf = has_valid_initial_model(initialModel, methodName)
tf = isstruct(initialModel) && isfield(initialModel, 'dlnet') && ...
    isfield(initialModel, 'method') && strcmpi(string(initialModel.method), string(methodName));
end

function tf = use_gpu(executionEnvironment)
tf = strcmpi(string(executionEnvironment), "gpu") && canUseGPU();
end

function value = gather_numeric_scalar(value)
value = gather_numeric_compat(value);
value = double(value);
if ~isscalar(value)
    value = value(1);
end
end

function value = gather_numeric_compat(value)
try
    value = extractdata(value);
catch
end
try
    value = gather(value);
catch
end
end

function XNorm = apply_zscore_samples(X, mu, sigma)
if isempty(X)
    XNorm = X;
    return;
end
XNorm = (X - mu) ./ sigma;
XNorm(~isfinite(XNorm)) = 0;
end

function XNorm = apply_fixed_zscore_dl(X, mu, sigma)
XNorm = (X - cast(mu, 'like', X)) ./ cast(sigma, 'like', X);
XNorm(~isfinite(XNorm)) = 0;
end

function weights = effective_num_weights(counts, beta)
counts = double(counts(:)');
effectiveNum = (1 - beta.^counts) ./ max(1 - beta, eps);
weights = 1 ./ max(effectiveNum, eps);
weights = weights / mean(weights);
end

function [X, inputSize, nSamples] = normalize_input_shape(trainData, inputSizeArg)
X = single(trainData);
switch ndims(X)
    case 2
        if isempty(inputSizeArg)
            error('CNN decoder 使用扁平输入时必须提供 input_size = [H W T]。');
        end
        inputSize = double(inputSizeArg(:)');
        if numel(inputSize) ~= 3
            error('input_size 必须是 [H W T]。');
        end
        nSamples = size(X, 1);
        if size(X, 2) ~= prod(inputSize)
            error('扁平输入特征数(%d)与 input_size 乘积(%d)不匹配。', size(X, 2), prod(inputSize));
        end
        X = reshape(X', [inputSize 1 nSamples]);
    case 4
        inputSize = [size(X, 1), size(X, 2), size(X, 3)];
        nSamples = size(X, 4);
        X = reshape(X, [inputSize 1 nSamples]);
    case 5
        inputSize = [size(X, 1), size(X, 2), size(X, 3)];
        if size(X, 4) ~= 1
            error('当前 CNN 解码器仅支持单通道体数据。');
        end
        nSamples = size(X, 5);
    otherwise
        error('不支持的 CNN 输入维度：ndims(trainData) = %d', ndims(X));
end
end
