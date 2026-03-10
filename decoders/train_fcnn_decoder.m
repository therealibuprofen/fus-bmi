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
% 数据输入模式：
%   'project'   使用项目统一接口 load_doppler_data + extract_training_data_and_labels
%   'matfile'   直接读取 cfg.dataFile
cfg.dataInputMode = "project";

% project 模式参数（推荐）
cfg.sessionRunList = [];                      % 例如 [12 3]；为空则弹窗选择
cfg.singleOrMultiple = "single";              % {'single','multiple'}
cfg.trainingSetSize = 3;                      % 与项目默认一致
cfg.bufferSize = 60;
cfg.preprocessZScore = true;
cfg.spatialFilter = {'', [], []};             % 例如 {'disk', 2, 0}

% matfile 模式参数（后备）
cfg.dataFile = "decoder_training_data.mat";   % 修改为你的训练数据路径
cfg.testRatio = 0.20;                         % 测试集比例
cfg.valRatioWithinTrain = 0.20;               % 训练集内部再划分验证集比例
cfg.randomSeed = 42;

cfg.hiddenDims = 3;                           % FCNN 隐层维度（固定为 3）
cfg.dropout = 0.0;                            % 不使用 dropout（固定）
cfg.maxEpochs = 120;
cfg.miniBatchSize = 64;
cfg.initialLearnRate = 1e-3;
cfg.l2Regularization = 1e-4;

cfg.modelOutFile = "decoders/fcnn_decoder_model.mat";

%% -------------------- 读取并整理数据 --------------------
rng(cfg.randomSeed);
[X, y, dataSource] = buildDecoderDataset(cfg);
fprintf("已识别输入数据来源: %s\n", dataSource);
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
    fullyConnectedLayer(3, "Name", "fc1")
    reluLayer("Name", "relu1")
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

%% -------------------- 可选：隐藏层 3D 可视化 --------------------
% 提取 3 个隐藏神经元的激活并绘图
% hiddenAct = activations(net, XTest, "relu1", "OutputAs", "rows");
% figure("Name", "Hidden Layer Activations", "Color", "w");
% scatter3(hiddenAct(:,1), hiddenAct(:,2), hiddenAct(:,3), 20, double(yTest), "filled");
% xlabel("Neuron 1"); ylabel("Neuron 2"); zlabel("Neuron 3");
% title("3D Hidden Activations (Test Set)");
% grid on;

%% ==================== 本脚本内函数 ====================
function [X, y, source] = parseInputData(s)
    [X, y, source] = parseXYFromStruct(s, "root");
    if ~isempty(X)
        if isrow(y), y = y'; end
        return;
    end

    % 若 .mat 里只有一个 struct 变量，尝试在其内部查找
    fn = fieldnames(s);
    if numel(fn) == 1 && isstruct(s.(fn{1}))
        [X, y, source] = parseXYFromStruct(s.(fn{1}), "root." + string(fn{1}));
        if ~isempty(X)
            if isrow(y), y = y'; end
            return;
        end
    end

    vars = strjoin(fieldnames(s), ", ");
    error(['未找到可识别的数据字段。可用变量: ' vars newline ...
        '支持格式: (data,labels), (X,y), (features,labels), ' ...
        '(train,trainLabels), (train_data,train_labels), ' ...
        '或包含 dop/behavior/timestamps 的原始数据 struct。']);
end

function [X, y, source] = parseXYFromStruct(st, prefix)
    X = [];
    y = [];
    source = "";

    pairs = {
        "data", "labels";
        "X", "y";
        "features", "labels";
        "train", "trainLabels"
    };
    for i = 1:size(pairs, 1)
        xf = pairs{i, 1};
        yf = pairs{i, 2};
        if isfield(st, xf) && isfield(st, yf)
            X = st.(xf);
            y = st.(yf);
            source = prefix + "." + xf + "/" + yf;
            return;
        end
    end

    if isfield(st, "train_data") && isfield(st, "train_labels")
        td = st.train_data;
        X = reshape(td, [], size(td, 4))';
        y = st.train_labels;
        source = prefix + ".train_data/train_labels";
        return;
    end

    % 兼容原始数据结构：调用项目已有提取函数
    if isfield(st, "dop") && isfield(st, "behavior") && isfield(st, "timestamps")
        if exist('extract_training_data_and_labels', 'file') ~= 2
            error(['检测到原始数据结构，但找不到 extract_training_data_and_labels.m。' ...
                '请先运行 setup.m。']);
        end
        [td, tl] = extract_training_data_and_labels(st);
        X = reshape(td, [], size(td, 4))';
        y = tl;
        source = prefix + ".(dop/behavior/timestamps)->extract_training_data_and_labels";
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

function [X, y, source] = buildDecoderDataset(cfg)
    mode = string(cfg.dataInputMode);
    switch lower(mode)
        case "project"
            loadArgs = { ...
                'single_or_multiple', char(cfg.singleOrMultiple), ...
                'verbose', true, ...
                'manual_alignment', false, ...
                'variables_to_load', {'dop', 'behavior', 'timestamps', 'actual_labels', 'session_run_list'}};

            if ~isempty(cfg.sessionRunList)
                loadArgs = [loadArgs, {'session_run_list', cfg.sessionRunList}];
            end

            data = load_doppler_data(loadArgs{:});
            [td, tl] = extract_training_data_and_labels(data, ...
                'zscore', cfg.preprocessZScore, ...
                'spatial_filter', cfg.spatialFilter, ...
                'training_set_size', cfg.trainingSetSize, ...
                'buffer_size', cfg.bufferSize);
            td = max_pool_2x2_stride2(td);
            X = reshape(td, [], size(td, 4))';
            y = tl(:);

            if isfield(data, 'session_run_list')
                source = "project_loader session_run_list=" + mat2str(data.session_run_list);
            else
                source = "project_loader";
            end

        case "matfile"
            s = load(cfg.dataFile);
            [X, y, source] = parseInputData(s);

        otherwise
            error("cfg.dataInputMode 只支持 'project' 或 'matfile'。");
    end
end

function td_pooled = max_pool_2x2_stride2(td)
    % td: [yPix x xPix x nFrames x nTrials]
    [yPix, xPix, nFrames, nTrials] = size(td);
    y2 = floor(yPix / 2);
    x2 = floor(xPix / 2);
    td = td(1:2*y2, 1:2*x2, :, :);

    td_pooled = zeros(y2, x2, nFrames, nTrials, 'like', td);
    for t = 1:nTrials
        for f = 1:nFrames
            A = td(:, :, f, t);
            A = reshape(A, 2, y2, 2, x2);
            td_pooled(:, :, f, t) = squeeze(max(max(A, [], 1), [], 3));
        end
    end
end
