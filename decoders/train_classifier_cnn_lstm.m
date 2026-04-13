function model = train_classifier_cnn_lstm(trainData, trainLabels, varargin)
% train_classifier_cnn_lstm  Train CNN+LSTM decoder for classification.
%
% Each frame is encoded by shared 2D CNN layers, then the resulting
% per-frame features are modeled by an LSTM over time.

p = inputParser;
p.addParameter('input_size', []);
p.addParameter('num_filters', [8 16]);
p.addParameter('spatial_kernel_size', [3 3]);
p.addParameter('dropout', 0.2);
p.addParameter('batch_norm', true);
p.addParameter('lstm_units', 64);
p.addParameter('max_epochs', 60);
p.addParameter('mini_batch_size', 8);
p.addParameter('initial_learn_rate', 1e-3);
p.addParameter('l2_regularization', 1e-4);
p.addParameter('gradient_threshold', 1.0);
p.addParameter('gradient_threshold_method', 'l2norm');
p.addParameter('learn_rate_schedule', 'piecewise');
p.addParameter('learn_rate_drop_factor', 0.5);
p.addParameter('learn_rate_drop_period', 15);
p.addParameter('validation_ratio', 0.2);
p.addParameter('validation_patience', 8);
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

hasValidation = numel(y) >= 10 && cfg.validation_ratio > 0;
if hasValidation
    try
        cv = cvpartition(yCat, 'HoldOut', cfg.validation_ratio);
        idxTrain = training(cv);
        idxVal = test(cv);
        XTrainRaw = X(:, :, :, :, idxTrain);
        yTrain = yCat(idxTrain);
        XValRaw = X(:, :, :, :, idxVal);
        yVal = yCat(idxVal);
    catch
        hasValidation = false;
        XTrainRaw = X;
        yTrain = yCat;
        XValRaw = [];
        yVal = categorical();
    end
else
    XTrainRaw = X;
    yTrain = yCat;
    XValRaw = [];
    yVal = categorical();
end

[mu, sigma] = compute_nan_mean_std_dim5(XTrainRaw);
if ~cfg.normalize_in_network
    XTrainRaw = apply_zscore_samples(XTrainRaw, mu, sigma);
    if hasValidation
        XValRaw = apply_zscore_samples(XValRaw, mu, sigma);
    end
end

XTrain = volumes_to_sequence_cells(XTrainRaw);
if hasValidation
    XVal = volumes_to_sequence_cells(XValRaw);
end

numClasses = numel(categories(yTrain));
numFilters = cfg.num_filters(:)';
if isempty(numFilters)
    error('num_filters must be non-empty.');
end

lg = layerGraph();
lg = addLayers(lg, sequenceInputLayer([inputSize(1) inputSize(2) 1], ...
    'Name', 'input', 'Normalization', 'none'));
lg = addLayers(lg, sequenceFoldingLayer('Name', 'fold'));
lg = connectLayers(lg, 'input', 'fold/in');

prev = 'fold/out';
if cfg.normalize_in_network
    lg = addLayers(lg, functionLayer(@(X) apply_fixed_zscore_dl(X, mu, sigma), ...
        'Name', 'fixed_zscore', ...
        'Formattable', true));
    lg = connectLayers(lg, prev, 'fixed_zscore');
    prev = 'fixed_zscore';
end
for i = 1:numel(numFilters)
    block = [
        convolution2dLayer(cfg.spatial_kernel_size, numFilters(i), ...
            'Padding', 'same', 'Name', sprintf('conv%d', i))
        reluLayer('Name', sprintf('relu%d', i))
        maxPooling2dLayer(2, 'Stride', 2, 'Name', sprintf('pool%d', i))
    ];
    if cfg.batch_norm
        block = [
            convolution2dLayer(cfg.spatial_kernel_size, numFilters(i), ...
                'Padding', 'same', 'Name', sprintf('conv%d', i))
            batchNormalizationLayer('Name', sprintf('bn%d', i))
            reluLayer('Name', sprintf('relu%d', i))
            maxPooling2dLayer(2, 'Stride', 2, 'Name', sprintf('pool%d', i))
        ];
    end
    lg = addLayers(lg, block);
    lg = connectLayers(lg, prev, sprintf('conv%d', i));
    prev = sprintf('pool%d', i);
end

tail = [
    sequenceUnfoldingLayer('Name', 'unfold')
    flattenLayer('Name', 'flatten')
    lstmLayer(cfg.lstm_units, 'OutputMode', 'last', 'Name', 'lstm')
];
if cfg.dropout > 0
    tail = [tail
        dropoutLayer(cfg.dropout, 'Name', 'drop')];
end
tail = [tail
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    build_classification_layer(yTrain, cfg.class_weight_mode)];

lg = addLayers(lg, tail);
lg = connectLayers(lg, prev, 'unfold/in');
lg = connectLayers(lg, 'fold/miniBatchSize', 'unfold/miniBatchSize');

opts = trainingOptions('adam', ...
    'MaxEpochs', resolve_max_epochs(cfg), ...
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

if cfg.gradient_threshold > 0
    try
        opts.GradientThreshold = cfg.gradient_threshold;
        opts.GradientThresholdMethod = char(cfg.gradient_threshold_method);
    catch
        % For older MATLAB versions without this option.
    end
end

if hasValidation
    opts.ValidationData = {XVal, yVal};
    opts.ValidationFrequency = max(1, floor(numel(XTrain) / cfg.mini_batch_size));
    if cfg.validation_patience > 0
        try
            opts.ValidationPatience = cfg.validation_patience;
        catch
            % For older MATLAB versions without this option.
        end
    end
    try
        opts.OutputNetwork = char(cfg.output_network);
    catch
        % For older MATLAB versions without this option.
    end
end

[net, warmStartUsed] = train_with_optional_warm_start(XTrain, yTrain, lg, opts, cfg);

model = struct();
model.method = 'CNN+LSTM';
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

function maxEpochs = resolve_max_epochs(cfg)
    maxEpochs = cfg.max_epochs;
    if has_valid_initial_model(cfg.initial_model)
        maxEpochs = min(cfg.max_epochs, cfg.fine_tune_epochs);
    end
end

function [net, warmStartUsed] = train_with_optional_warm_start(XTrain, yTrain, lg, opts, cfg)
    warmStartUsed = false;
    if has_valid_initial_model(cfg.initial_model)
        try
            warmGraph = layerGraph(cfg.initial_model.net);
            net = trainNetwork(XTrain, yTrain, warmGraph, opts);
            warmStartUsed = true;
            return;
        catch ME
            warning('CNN+LSTM warm-start failed, fallback to cold start: %s', ME.message);
        end
    end
    net = trainNetwork(XTrain, yTrain, lg, opts);
end

function tf = has_valid_initial_model(initialModel)
    tf = isstruct(initialModel) && isfield(initialModel, 'net') && ~isempty(initialModel.net);
end

function seq = volumes_to_sequence_cells(X)
    nSamples = size(X, 5);
    seq = cell(nSamples, 1);
    for i = 1:nSamples
        seq{i} = permute(X(:, :, :, :, i), [1 2 4 3]);
    end
end

function [X, inputSize, nSamples] = normalize_input_shape(trainData, inputSizeArg)
    X = single(trainData);
    switch ndims(X)
        case 2
            if isempty(inputSizeArg)
                error('CNN+LSTM decoder 使用扁平输入时必须提供 input_size = [H W T]。');
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
                error('当前 CNN+LSTM 解码器仅支持单通道体数据。');
            end
            nSamples = size(X, 5);
        otherwise
            error('不支持的 CNN+LSTM 输入维度：ndims(trainData) = %d', ndims(X));
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
            imbalanceRatio = max(counts) / max(min(counts), 1);
            if imbalanceRatio >= 3
                weights = effective_num_weights(counts, 0.999);
            else
                weights = median(counts) ./ max(counts, 1);
                weights = weights / mean(weights);
            end
            layer = classificationLayer('Name', 'classOutput', ...
                'Classes', classes, 'ClassWeights', weights);
        otherwise
            layer = classificationLayer('Name', 'classOutput', 'Classes', classes);
    end
end

function weights = effective_num_weights(counts, beta)
    counts = double(counts(:)');
    effectiveNum = (1 - beta.^counts) ./ max(1 - beta, eps);
    weights = 1 ./ max(effectiveNum, eps);
    weights = weights / mean(weights);
end

function [mu, sigma] = compute_nan_mean_std_dim5(X)
    % Memory-efficient mean/std over sample dimension (dim 5) while omitting NaNs.
    nSamples = size(X, 5);
    statSize = size(X(:, :, :, :, 1));
    sumX = zeros(statSize, 'double');
    sumX2 = zeros(statSize, 'double');
    count = zeros(statSize, 'double');

    for i = 1:nSamples
        Xi = double(X(:, :, :, :, i));
        valid = ~isnan(Xi);
        Xi(~valid) = 0;
        sumX = sumX + Xi;
        sumX2 = sumX2 + Xi.^2;
        count = count + double(valid);
    end

    muDouble = zeros(statSize, 'double');
    hasData = count > 0;
    muDouble(hasData) = sumX(hasData) ./ count(hasData);

    varDouble = zeros(statSize, 'double');
    enoughData = count > 1;
    varNumerator = sumX2(enoughData) - (sumX(enoughData).^2) ./ count(enoughData);
    varNumerator = max(varNumerator, 0);
    varDouble(enoughData) = varNumerator ./ (count(enoughData) - 1);

    sigmaDouble = sqrt(varDouble);
    sigmaDouble(~isfinite(sigmaDouble) | sigmaDouble == 0) = 1;

    mu = cast(muDouble, 'like', X);
    sigma = cast(sigmaDouble, 'like', X);
end
