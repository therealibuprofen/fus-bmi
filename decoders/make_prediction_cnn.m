function class = make_prediction_cnn(testData, model)
% make_prediction_cnn  Predict class using CNN decoder model.
%
% INPUTS:
%   testData: [H x W x T], [H x W x T x N], [H x W x T x 1 x N], or
%             flattened [N x (H*W*T)] / [1 x (H*W*T)]
%   model:    output struct from train_classifier_cnn
%
% OUTPUT:
%   class:    numeric class labels, one per sample

[X, nSamples] = normalize_input_shape(testData, model.inputSize);
X = (X - model.mu) ./ model.sigma;
X(~isfinite(X)) = 0;

yPred = classify(model.net, X);
yPredStr = string(yPred);

class = NaN(size(yPredStr));
for i = 1:numel(yPredStr)
    v = str2double(yPredStr(i));
    if ~isnan(v)
        class(i) = v;
    else
        idx = find(string(model.classNames) == yPredStr(i), 1, 'first');
        if isempty(idx)
            error('CNN predicted an unknown class label: %s', yPredStr(i));
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
                error('测试输入特征数(%d)与模型 inputSize 乘积(%d)不匹配。', ...
                    size(X, 2), prod(inputSize));
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
            error('不支持的 CNN 测试输入维度：ndims(testData) = %d', ndims(X));
    end
end
