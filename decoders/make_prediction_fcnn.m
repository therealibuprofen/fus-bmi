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
            error('FCNN predicted an unknown class label: %s', yPredStr(i));
        end
        class(i) = model.classValues(idx);
    end
end

if isrow(testData) || (isvector(testData) && size(testData, 1) == 1)
    class = class(1);
end
end
