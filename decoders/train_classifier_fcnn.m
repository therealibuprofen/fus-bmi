function model = train_classifier_fcnn(trainData, trainLabels, varargin)
% train_classifier_fcnn  Train FCNN decoder for classification.
%
% INPUTS:
%   trainData:    (nSamples x nFeatures)
%   trainLabels:  (nSamples x 1) or (1 x nSamples)
%
% OPTIONAL NAME-VALUE:
%   'hidden_dims'         default [64 32]
%   'dropout'             default 0.2
%   'max_epochs'          default 120
%   'mini_batch_size'     default 64
%   'initial_learn_rate'  default 1e-3
%   'l2_regularization'   default 1e-4
%   'validation_ratio'    default 0.2
%   'verbose'             default false
%   'random_seed'         default 42

p = inputParser;
p.addParameter('hidden_dims', [64 32]);
p.addParameter('dropout', 0.2);
p.addParameter('max_epochs', 120);
p.addParameter('mini_batch_size', 64);
p.addParameter('initial_learn_rate', 1e-3);
p.addParameter('l2_regularization', 1e-4);
p.addParameter('validation_ratio', 0.2);
p.addParameter('verbose', false, @islogical);
p.addParameter('random_seed', 42);
p.KeepUnmatched = true;
p.parse(varargin{:});
cfg = p.Results;

rng(cfg.random_seed);

X = double(trainData);
y = trainLabels(:);
ok = all(isfinite(X), 2) & ~isnan(y);
X = X(ok, :);
y = y(ok);

classVals = unique(y, 'sorted');
classNames = string(classVals);
yCat = categorical(y, classVals, classNames);

% If dataset is too small to split, skip validation.
hasValidation = size(X, 1) >= 10 && cfg.validation_ratio > 0;
if hasValidation
    try
        cv = cvpartition(yCat, 'HoldOut', cfg.validation_ratio);
        idxTrain = training(cv);
        idxVal = test(cv);
        XTrain = X(idxTrain, :);
        yTrain = yCat(idxTrain);
        XVal = X(idxVal, :);
        yVal = yCat(idxVal);
    catch
        hasValidation = false;
        XTrain = X;
        yTrain = yCat;
    end
else
    XTrain = X;
    yTrain = yCat;
end

[XTrain, mu, sigma] = zscore(XTrain);
sigma(sigma == 0) = 1;
XTrain(~isfinite(XTrain)) = 0;

if hasValidation
    XVal = (XVal - mu) ./ sigma;
    XVal(~isfinite(XVal)) = 0;
end

numFeatures = size(XTrain, 2);
numClasses = numel(categories(yTrain));
hidden = cfg.hidden_dims(:)';
if isempty(hidden)
    error('hidden_dims must be non-empty.');
end

layers = [featureInputLayer(numFeatures, 'Name', 'input', 'Normalization', 'none')];
for i = 1:numel(hidden)
    layers = [layers
        fullyConnectedLayer(hidden(i), 'Name', sprintf('fc%d', i))
        reluLayer('Name', sprintf('relu%d', i))];
    if cfg.dropout > 0
        layers = [layers
            dropoutLayer(cfg.dropout, 'Name', sprintf('drop%d', i))];
    end
end
layers = [layers
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')];

opts = trainingOptions('adam', ...
    'MaxEpochs', cfg.max_epochs, ...
    'MiniBatchSize', cfg.mini_batch_size, ...
    'InitialLearnRate', cfg.initial_learn_rate, ...
    'L2Regularization', cfg.l2_regularization, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', cfg.verbose, ...
    'Plots', 'none');

if hasValidation
    opts.ValidationData = {XVal, yVal};
    opts.ValidationFrequency = max(1, floor(size(XTrain, 1) / cfg.mini_batch_size));
end

net = trainNetwork(XTrain, yTrain, layers, opts);

model = struct();
model.method = 'FCNN';
model.net = net;
model.mu = mu;
model.sigma = sigma;
model.classValues = classVals;
model.classNames = categories(yTrain);
model.config = cfg;
end
