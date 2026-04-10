function class = make_prediction_cnn_lstm(testData, model)
% make_prediction_cnn_lstm  Predict class using CNN+LSTM decoder model.

[X, nSamples] = normalize_input_shape(testData, model.inputSize);
if ~isfield(model, 'normalizeInNetwork') || ~model.normalizeInNetwork
    X = (X - model.mu) ./ model.sigma;
    X(~isfinite(X)) = 0;
end

seq = cell(nSamples, 1);
poolSize = 1;
if isfield(model, 'config') && isstruct(model.config) && isfield(model.config, 'temporal_pool_size') ...
        && ~isempty(model.config.temporal_pool_size)
    poolSize = max(1, round(double(model.config.temporal_pool_size)));
end
for i = 1:nSamples
    Xi = X(:, :, :, :, i);
    if poolSize > 1
        Xi = temporal_average_pool_volume(Xi, poolSize);
    end
    seq{i} = permute(Xi, [1 2 4 3]);
end

yPredStr = scores_to_class_names(predict(model.net, seq), model.classNames);

class = NaN(size(yPredStr));
for i = 1:numel(yPredStr)
    v = str2double(yPredStr(i));
    if ~isnan(v)
        class(i) = v;
    else
        idx = find(string(model.classNames) == yPredStr(i), 1, 'first');
        if isempty(idx)
            error('CNN+LSTM predicted an unknown class label: %s', yPredStr(i));
        end
        class(i) = model.classValues(idx);
    end
end

if nSamples == 1
    class = class(1);
end
end

function [X, nSamples] = normalize_input_shape(testData, inputSize)
    X = double(testData);
    switch ndims(X)
        case 2
            if size(X, 2) ~= prod(inputSize)
                error('测试输入特征数(%d)与模型 inputSize 乘积(%d)不匹配。', size(X, 2), prod(inputSize));
            end
            nSamples = size(X, 1);
            X = reshape(X', [inputSize 1 nSamples]);
        case 3
            if ~isequal(size(X), inputSize)
                error('测试输入尺寸与模型 inputSize 不匹配。');
            end
            nSamples = 1;
            X = reshape(X, [inputSize 1 1]);
        case 4
            if isequal([size(X, 1), size(X, 2), size(X, 3)], inputSize)
                nSamples = size(X, 4);
                X = reshape(X, [inputSize 1 nSamples]);
            else
                error('测试输入尺寸与模型 inputSize 不匹配。');
            end
        case 5
            if ~isequal([size(X, 1), size(X, 2), size(X, 3)], inputSize) || size(X, 4) ~= 1
                error('测试输入尺寸与模型 inputSize 不匹配。');
            end
            nSamples = size(X, 5);
        otherwise
            error('不支持的 CNN+LSTM 测试输入维度：ndims(testData) = %d', ndims(X));
    end
end

function yPredStr = scores_to_class_names(scores, classNames)
    scores = gather_numeric(scores);
    nClasses = numel(classNames);
    dims = size(scores);
    classDim = find(dims == nClasses, 1, 'last');
    if isempty(classDim)
        error('CNN+LSTM predict 输出尺寸与类别数不匹配。');
    end
    [~, idx] = max(scores, [], classDim);
    idx = reshape(gather_numeric(idx), [], 1);
    yPredStr = string(classNames(idx));
end

function value = gather_numeric(value)
    if isa(value, 'dlarray')
        value = extractdata(value);
    end
    if isa(value, 'gpuArray')
        value = gather(value);
    end
end

function Xpool = temporal_average_pool_volume(X, poolSize)
    nT = size(X, 3);
    nBlocks = ceil(nT / poolSize);
    Xpool = zeros(size(X, 1), size(X, 2), nBlocks, size(X, 4), 'like', X);
    for b = 1:nBlocks
        tStart = (b - 1) * poolSize + 1;
        tEnd = min(nT, b * poolSize);
        Xpool(:, :, b, :) = mean(X(:, :, tStart:tEnd, :), 3, 'omitnan');
    end
end
