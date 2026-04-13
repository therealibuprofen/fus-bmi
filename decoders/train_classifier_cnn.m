function model = train_classifier_cnn(trainData, trainLabels, varargin)
% train_classifier_cnn  Train factorized CNN decoder for classification.
%
% This model replaces full 3D convolutions with a spatial 2D convolution
% [k k 1] followed by a temporal 1D convolution [1 1 t], which reduces
% parameter count and usually shortens decode time.
%
% INPUTS:
%   trainData:    [H x W x T x N] or [H x W x T x 1 x N]
%                 Flattened [N x (H*W*T)] is also accepted when input_size
%                 is provided.
%   trainLabels:  (N x 1) or (1 x N)

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
p.addParameter('learn_rate_schedule', 'piecewise');
p.addParameter('learn_rate_drop_factor', 0.5);
p.addParameter('learn_rate_drop_period', 15);
p.addParameter('validation_ratio', 0.2);
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

[X, inputSize, nSamples] = normalize_input_shape(trainData, cfg.input_size);
y = trainLabels(:);
if numel(y) ~= nSamples
    error('训练标签数量(%d)与样本数量(%d)不一致。', numel(y), nSamples);
end

validMask = squeeze(all(all(all(all(isfinite(X), 1), 2), 3), 4));
validMask = reshape(validMask, [], 1) & ~isnan(y);
X = X(:, :, :, :, validMask);
y = y(validMask);

classVals = unique(y, 'sorted');
classNames = string(classVals);
yCat = categorical(y, classVals, classNames);

[XTrain, yTrain, XVal, yVal, hasValidation, mu, sigma] = split_and_prepare(X, yCat, cfg.validation_ratio, cfg.normalize_in_network);

numClasses = numel(categories(yTrain));
numFilters = cfg.num_filters(:)';
if isempty(numFilters)
    error('num_filters must be non-empty.');
end

layers = image3dInputLayer([inputSize 1], 'Name', 'input', 'Normalization', 'none');
if cfg.normalize_in_network
    layers = [
        layers
        functionLayer(@(X) apply_fixed_zscore_dl(X, mu, sigma), ...
            'Name', 'fixed_zscore', ...
            'Formattable', true)
    ];
end
for i = 1:numel(numFilters)
    layers = [layers
        convolution3dLayer([cfg.spatial_kernel_size(:)' 1], numFilters(i), ...
            'Padding', 'same', 'Name', sprintf('spatial_conv%d', i))
        reluLayer('Name', sprintf('spatial_relu%d', i))];
    if cfg.batch_norm
        layers = [layers
            batchNormalizationLayer('Name', sprintf('spatial_bn%d', i))];
    end
    layers = [layers
        convolution3dLayer([1 1 cfg.temporal_kernel_size], numFilters(i), ...
            'Padding', 'same', 'Name', sprintf('temporal_conv%d', i))
        reluLayer('Name', sprintf('temporal_relu%d', i))];
    if cfg.batch_norm
        layers = [layers
            batchNormalizationLayer('Name', sprintf('temporal_bn%d', i))];
    end
    layers = [layers
        maxPooling3dLayer([2 2 1], 'Stride', [2 2 1], 'Name', sprintf('pool%d', i))];
    if cfg.dropout > 0
        layers = [layers
            dropoutLayer(cfg.dropout, 'Name', sprintf('drop%d', i))];
    end
end

layers = [layers
    fullyConnectedLayer(max(32, numFilters(end)), 'Name', 'fc1')
    reluLayer('Name', 'fc1_relu')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    build_classification_layer(yTrain, cfg.class_weight_mode)];

opts = build_training_options(cfg, hasValidation, XTrain, XVal, yVal);
[net, warmStartUsed] = train_with_optional_warm_start(XTrain, yTrain, layers, opts, cfg);

model = struct();
model.method = 'CNN';
model.net = net;
model.mu = mu;
model.sigma = sigma;
model.inputSize = inputSize;
model.classValues = classVals;
model.classNames = categories(yTrain);
model.config = cfg;
model.normalizeInNetwork = cfg.normalize_in_network;
model.warmStartUsed = warmStartUsed;
end

function [XTrain, yTrain, XVal, yVal, hasValidation, mu, sigma] = split_and_prepare(X, yCat, validationRatio, normalizeInNetwork)
    hasValidation = numel(yCat) >= 10 && validationRatio > 0;
    if hasValidation
        try
            cv = cvpartition(yCat, 'HoldOut', validationRatio);
            idxTrain = training(cv);
            idxVal = test(cv);
            XTrainRaw = X(:, :, :, :, idxTrain);
            yTrain = yCat(idxTrain);
            XVal = X(:, :, :, :, idxVal);
            yVal = yCat(idxVal);
        catch
            hasValidation = false;
            XTrainRaw = X;
            yTrain = yCat;
            XVal = [];
            yVal = categorical();
        end
    else
        XTrainRaw = X;
        yTrain = yCat;
        XVal = [];
        yVal = categorical();
    end

    mu = mean(XTrainRaw, 5, 'omitnan');
    sigma = std(XTrainRaw, 0, 5, 'omitnan');
    sigma(sigma == 0) = 1;
    if normalizeInNetwork
        XTrain = XTrainRaw;
        if hasValidation
            XVal = XVal;
        end
    else
        XTrain = apply_zscore_samples(XTrainRaw, mu, sigma);
        if hasValidation
            XVal = apply_zscore_samples(XVal, mu, sigma);
        end
    end
end

function opts = build_training_options(cfg, hasValidation, XTrain, XVal, yVal)
    maxEpochs = cfg.max_epochs;
    if has_valid_initial_model(cfg.initial_model)
        maxEpochs = min(cfg.max_epochs, cfg.fine_tune_epochs);
    end

    opts = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', cfg.mini_batch_size, ...
        'InitialLearnRate', cfg.initial_learn_rate, ...
        'L2Regularization', cfg.l2_regularization, ...
        'LearnRateSchedule', cfg.learn_rate_schedule, ...
        'LearnRateDropFactor', cfg.learn_rate_drop_factor, ...
        'LearnRateDropPeriod', cfg.learn_rate_drop_period, ...
        'ExecutionEnvironment', char(cfg.execution_environment), ...
        'Shuffle', 'every-epoch', ...
        'Verbose', cfg.verbose, ...
        'Plots', 'none');

    if hasValidation
        opts.ValidationData = {XVal, yVal};
        opts.ValidationFrequency = max(1, floor(size(XTrain, 5) / cfg.mini_batch_size));
    end
end

function [net, warmStartUsed] = train_with_optional_warm_start(XTrain, yTrain, layers, opts, cfg)
    warmStartUsed = false;
    if has_valid_initial_model(cfg.initial_model)
        try
            warmGraph = layerGraph(cfg.initial_model.net);
            net = trainNetwork(XTrain, yTrain, warmGraph, opts);
            warmStartUsed = true;
            return;
        catch ME
            warning('CNN warm-start failed, fallback to cold start: %s', ME.message);
        end
    end
    net = trainNetwork(XTrain, yTrain, layers, opts);
end

function tf = has_valid_initial_model(initialModel)
    tf = isstruct(initialModel) && isfield(initialModel, 'net') && ~isempty(initialModel.net);
end

function [X, inputSize, nSamples] = normalize_input_shape(trainData, inputSizeArg)
    X = double(trainData);
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

function XNorm = apply_zscore_samples(X, mu, sigma)
    XNorm = (X - mu) ./ sigma;
    XNorm(~isfinite(XNorm)) = 0;
end

function XNorm = apply_fixed_zscore_dl(X, mu, sigma)
    XNorm = (X - cast(mu, 'like', X)) ./ cast(sigma, 'like', X);
    XNorm(~isfinite(XNorm)) = 0;
end

function layer = build_classification_layer(yTrain, mode)
    classes = categories(yTrain);
    switch lower(string(mode))
        case "balanced"
            counts = countcats(yTrain);
            weights = median(counts) ./ max(counts, 1);
            layer = classificationLayer('Name', 'classOutput', ...
                'Classes', classes, 'ClassWeights', weights);
        otherwise
            layer = classificationLayer('Name', 'classOutput', 'Classes', classes);
    end
end
