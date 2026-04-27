function classes = make_prediction_cnn_lstm_twohead(testData, model)
% make_prediction_cnn_lstm_twohead  Predict horizontal and vertical classes.

[X, nSamples] = normalize_input_shape(testData, model.inputSize);
if ~isfield(model, 'normalizeInNetwork') || ~model.normalizeInNetwork
    X = (X - model.mu) ./ model.sigma;
    X(~isfinite(X)) = 0;
end

dlX = dlarray(permute(X, [1 2 4 3 5]), 'SSCTB');
if canUseGPU()
    try
        dlX = gpuArray(dlX);
    catch
    end
end

[scoreH, scoreV] = forward(model.dlnet, dlX, 'Outputs', {'fc_horz', 'fc_vert'});
classH = scores_to_class_values(scoreH, model.horzClassValues);
classV = scores_to_class_values(scoreV, model.vertClassValues);
classes = [classH(:), classV(:)];

if nSamples == 1
    classes = classes(1, :);
end
end

function [X, nSamples] = normalize_input_shape(testData, inputSize)
X = single(testData);
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

function classVals = scores_to_class_values(scores, classValues)
scores = gather_numeric(stripdims(scores));
[~, idx] = max(scores, [], 1);
idx = reshape(idx, [], 1);
classVals = classValues(idx);
end

function value = gather_numeric(value)
if isa(value, 'dlarray')
    value = extractdata(value);
end
if isa(value, 'gpuArray')
    value = gather(value);
end
end
