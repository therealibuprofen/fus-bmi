%% train_fcnn_decoder
% 完整脚本：构建、训练并评估 FCNN 分类器，用于替代 PCA+LDA / CPCA+LDA decoder。
%
% 数据要求（.mat 文件中至少包含一组）:
%   1) data (nSamples x nFeatures), labels (nSamples x 1 or 1 x nSamples)
%   2) train_data (yPix x xPix x nTimepoints x nSamples), train_labels
%
% 运行后输出:
%   - 终端打印测试集 Accuracy / Macro-F1
%   - 混淆矩阵图
%   - 保存模型到 decoders/fcnn_decoder_model.mat

clear;
clc;

%% -------------------- 可配置参数 --------------------
cfg.dataFile = "decoder_training_data.mat";   % 修改为你的训练数据路径
cfg.testRatio = 0.20;                         % 测试集比例
cfg.valRatioWithinTrain = 0.20;               % 训练集内部再划分验证集比例
cfg.randomSeed = 42;

cfg.hiddenDims = [256, 128];                  % FCNN 隐层维度
cfg.dropout = 0.20;
cfg.maxEpochs = 120;
cfg.miniBatchSize = 64;
cfg.initialLearnRate = 1e-3;
cfg.l2Regularization = 1e-4;

cfg.modelOutFile = "decoders/fcnn_decoder_model.mat";

%% -------------------- 读取并整理数据 --------------------
rng(cfg.randomSeed);
s = load(cfg.dataFile);

[X, y] = parseInputData(s);
X = double(X);
y = y(:);

valid = all(isfinite(X), 2) & ~isnan(y);
X = X(valid, :);
y = y(valid);

if numel(unique(y)) < 2
    error("标签类别数不足（<2），无法进行分类训练。");
end

classVals = unique(y, "sorted");
classNames = string(classVals);
yCat = categorical(y, classVals, classNames);

%% -------------------- 分层划分 train/val/test --------------------
cvTest = cvpartition(yCat, "HoldOut", cfg.testRatio);
idxTrainVal = training(cvTest);
idxTest = test(cvTest);

XTrainVal = X(idxTrainVal, :);
yTrainVal = yCat(idxTrainVal);
XTest = X(idxTest, :);
yTest = yCat(idxTest);

cvVal = cvpartition(yTrainVal, "HoldOut", cfg.valRatioWithinTrain);
idxTrain = training(cvVal);
idxVal = test(cvVal);

XTrain = XTrainVal(idxTrain, :);
yTrain = yTrainVal(idxTrain);
XVal = XTrainVal(idxVal, :);
yVal = yTrainVal(idxVal);

%% -------------------- 标准化（只用训练集统计量） --------------------
[XTrain, mu, sigma] = zscore(XTrain);
sigma(sigma == 0) = 1;
XVal = (XVal - mu) ./ sigma;
XTest = (XTest - mu) ./ sigma;

XTrain(~isfinite(XTrain)) = 0;
XVal(~isfinite(XVal)) = 0;
XTest(~isfinite(XTest)) = 0;

%% -------------------- 定义 FCNN --------------------
numFeatures = size(XTrain, 2);
numClasses = numel(categories(yTrain));

layers = [
    featureInputLayer(numFeatures, "Name", "input", "Normalization", "none")
    fullyConnectedLayer(cfg.hiddenDims(1), "Name", "fc1")
    reluLayer("Name", "relu1")
    dropoutLayer(cfg.dropout, "Name", "drop1")
    fullyConnectedLayer(cfg.hiddenDims(2), "Name", "fc2")
    reluLayer("Name", "relu2")
    dropoutLayer(cfg.dropout, "Name", "drop2")
    fullyConnectedLayer(numClasses, "Name", "fc_out")
    softmaxLayer("Name", "softmax")
    classificationLayer("Name", "classOutput")
];

opts = trainingOptions("adam", ...
    "MaxEpochs", cfg.maxEpochs, ...
    "MiniBatchSize", cfg.miniBatchSize, ...
    "InitialLearnRate", cfg.initialLearnRate, ...
    "L2Regularization", cfg.l2Regularization, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", {XVal, yVal}, ...
    "ValidationFrequency", max(1, floor(size(XTrain, 1) / cfg.miniBatchSize)), ...
    "Verbose", true, ...
    "Plots", "training-progress");

%% -------------------- 训练 --------------------
net = trainNetwork(XTrain, yTrain, layers, opts);

%% -------------------- 测试集评估 --------------------
yPred = classify(net, XTest);
acc = mean(yPred == yTest);

cm = confusionmat(yTest, yPred, "Order", categories(yTrain));
macroF1 = computeMacroF1(cm);

fprintf("\n===== FCNN Decoder Test Result =====\n");
fprintf("Test samples   : %d\n", numel(yTest));
fprintf("Accuracy       : %.2f%%\n", acc * 100);
fprintf("Macro-F1       : %.4f\n", macroF1);

figure("Name", "FCNN Decoder - Confusion Matrix", "Color", "w");
confusionchart(yTest, yPred);
title(sprintf("FCNN Test Accuracy = %.2f%% | Macro-F1 = %.4f", acc * 100, macroF1));

%% -------------------- 保存为 decoder 结构体 --------------------
decoder = struct();
decoder.method = "FCNN";
decoder.net = net;
decoder.mu = mu;
decoder.sigma = sigma;
decoder.classValues = classVals;
decoder.classNames = categories(yTrain);
decoder.config = cfg;

save(cfg.modelOutFile, "decoder", "acc", "macroF1", "cm");
fprintf("模型已保存到: %s\n", cfg.modelOutFile);

%% -------------------- 使用示例 --------------------
% singleTestData = XTest(1, :);            % 1 x nFeatures
% predClass = make_prediction_fcnn(singleTestData, decoder)

%% ==================== 本脚本内函数 ====================
function [X, y] = parseInputData(s)
    hasDataLabels = isfield(s, "data") && isfield(s, "labels");
    hasTrainData = isfield(s, "train_data") && isfield(s, "train_labels");

    if hasDataLabels
        X = s.data;
        y = s.labels;
    elseif hasTrainData
        td = s.train_data;
        % train_data: yPix x xPix x nTimepoints x nSamples
        X = reshape(td, [], size(td, 4))';
        y = s.train_labels;
    else
        error(["未找到可识别的数据字段。需要以下之一: " + ...
               "(data, labels) 或 (train_data, train_labels)。"]);
    end

    if isrow(y)
        y = y';
    end
end

function macroF1 = computeMacroF1(cm)
    nClass = size(cm, 1);
    f1 = zeros(nClass, 1);
    for i = 1:nClass
        tp = cm(i, i);
        fp = sum(cm(:, i)) - tp;
        fn = sum(cm(i, :)) - tp;
        p = tp / max(tp + fp, eps);
        r = tp / max(tp + fn, eps);
        f1(i) = 2 * p * r / max(p + r, eps);
    end
    macroF1 = mean(f1);
end

function predClass = make_prediction_fcnn(testData, decoder)
    if ndims(testData) > 2
        testData = reshape(testData, 1, []);
    end
    x = double(testData);
    x = (x - decoder.mu) ./ decoder.sigma;
    x(~isfinite(x)) = 0;

    yPred = classify(decoder.net, x);
    predClass = str2double(string(yPred));

    if isnan(predClass)
        idx = find(decoder.classNames == string(yPred), 1, "first");
        predClass = decoder.classValues(idx);
    end
end
