function class = make_prediction_fcnn(testData, model)
% make_prediction_fcnn  Predict class using FCNN decoder model.
%
% INPUTS:
%   testData: (nSamples x nFeatures) or (1 x nFeatures)
%   model:    output struct from train_classifier_fcnn
%
% OUTPUT:
%   class:    numeric class labels, one per sample

X = double(testData);
if isvector(X)
    X = reshape(X, 1, []);
end

X = (X - model.mu) ./ model.sigma;
X(~isfinite(X)) = 0;

yPredStr = scores_to_class_names(predict(model.net, X), model.classNames);

class = NaN(size(yPredStr));
for i = 1:numel(yPredStr)
    v = str2double(yPredStr(i));
    if ~isnan(v)
        class(i) = v;
    else
        idx = find(string(model.classNames) == yPredStr(i), 1, 'first');
        if isempty(idx)
            error('FCNN predicted an unknown class label: %s', yPredStr(i));
        end
        class(i) = model.classValues(idx);
    end
end

if isrow(testData) || (isvector(testData) && size(testData, 1) == 1)
    class = class(1);
end
end

function yPredStr = scores_to_class_names(scores, classNames)
    scores = gather_numeric(scores);
    nClasses = numel(classNames);
    dims = size(scores);
    classDim = find(dims == nClasses, 1, 'last');
    if isempty(classDim)
        error('FCNN predict 输出尺寸与类别数不匹配。');
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
