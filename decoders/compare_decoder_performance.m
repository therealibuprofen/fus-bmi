%% compare_decoder_performance
% 在完全相同的交叉验证折分下，对比原解码器与 FCNN 的性能。
%
% 数据要求（.mat 文件中至少包含一组）:
%   1) data (nSamples x nFeatures), labels (nSamples x 1 or 1 x nSamples)
%   2) train_data (yPix x xPix x nTimepoints x nSamples), train_labels

clear;
clc;

%% -------------------- 配置 --------------------
% 数据输入模式：
%   'project'   使用项目统一接口 load_doppler_data + extract_training_data_and_labels
%   'matfile'   直接读取 cfg.dataFile
cfg.dataInputMode = "project";

% project 模式参数（推荐）
cfg.sessionRunList = [];                     % 例如 [12 3]；为空则弹窗选择
cfg.singleOrMultiple = "single";             % {'single','multiple'}
cfg.trainingSetSize = 3;
cfg.bufferSize = 60;
cfg.preprocessZScore = true;
cfg.spatialFilter = {'', [], []};

% matfile 模式参数（后备）
cfg.dataFile = "decoder_training_data.mat";

cfg.validationMethod = "kFold";              % "kFold" 或 "leaveOneOut"
cfg.K = 10;                                  % 仅在 kFold 生效
cfg.randomSeed = 42;
cfg.decoderVerbose = true;                   % 打印是否使用 FCNN

% 需要对比的方法（原解码器 + FCNN）
cfg.methods = {"CPCA+LDA", "PCA+LDA", "FCNN"};

%% -------------------- 读取数据 --------------------
rng(cfg.randomSeed);
[X, y, dataSource] = buildDecoderDataset(cfg);
fprintf("已识别输入数据来源: %s\n", dataSource);

X = double(X);
y = y(:);
ok = all(isfinite(X), 2) & ~isnan(y);
X = X(ok, :);
y = y(ok);

N = size(X, 1);
if strcmpi(cfg.validationMethod, "leaveOneOut")
    K = N;
else
    K = cfg.K;
end

%% -------------------- 固定折分，保证公平对比 --------------------
indices = crossvalind('Kfold', N, K);
supportsFoldIndices = true;

%% -------------------- 逐方法评估 --------------------
nMethods = numel(cfg.methods);
acc = NaN(nMethods, 1);
pval = NaN(nMethods, 1);
nCorrect = NaN(nMethods, 1);
nTotal = NaN(nMethods, 1);

for i = 1:nMethods
    method = cfg.methods{i};
    fprintf('\n===== Evaluating %s =====\n', method);

    % Use local CV to guarantee FCNN path is exercised.
    [acc(i), nCorrect(i), nTotal(i), pval(i)] = cross_validate_local( ...
        X, y, indices, method, cfg.decoderVerbose);
end

results = table(string(cfg.methods(:)), nCorrect, nTotal, acc, pval, ...
    'VariableNames', {'Method', 'Correct', 'Total', 'AccuracyPercent', 'PValue'});

disp(' ');
disp('===== Decoder Comparison (same CV splits) =====');
disp(results);

% 创新性展示：FCNN 相对最佳传统方法提升
isFCNN = strcmpi(results.Method, "FCNN");
if any(isFCNN)
    fcnnAcc = results.AccuracyPercent(isFCNN);
    baselineAcc = max(results.AccuracyPercent(~isFCNN));
    delta = fcnnAcc - baselineAcc;
    fprintf('\nFCNN vs best baseline delta: %+0.2f percentage points\n', delta);
end

figure('Name', 'Decoder Performance Comparison', 'Color', 'w');
bar(categorical(results.Method), results.AccuracyPercent);
ylabel('Accuracy (%)');
title('Decoder Comparison on Identical Cross-Validation Splits');
grid on;

%% -------------------- 本脚本内函数 --------------------
function [X, y, source] = parseInputData(s)
    [X, y, source] = parseXYFromStruct(s, "root");
    if ~isempty(X)
        if isrow(y), y = y'; end
        return;
    end

    fn = fieldnames(s);
    if numel(fn) == 1 && isstruct(s.(fn{1}))
        [X, y, source] = parseXYFromStruct(s.(fn{1}), "root." + string(fn{1}));
        if ~isempty(X)
            if isrow(y), y = y'; end
            return;
        end
    end

    vars = strjoin(fieldnames(s), ", ");
    error(['未找到可识别字段。可用变量: ' vars newline ...
        '支持格式: (data,labels), (X,y), (features,labels), ' ...
        '(train,trainLabels), (train_data,train_labels), ' ...
        '或包含 dop/behavior/timestamps 的原始数据 struct。']);
end

function [accPct, nCorrect, nTotal] = summarize_classperf(cp)
    % Robustly extract confusion matrix from different classperf classes.
    cm = [];
    if isprop(cp, 'CountingMatrix')
        cm = cp.CountingMatrix;
    elseif isprop(cp, 'ConfusionMatrix')
        cm = cp.ConfusionMatrix;
    elseif isfield(cp, 'CountingMatrix')
        cm = cp.CountingMatrix;
    elseif isfield(cp, 'ConfusionMatrix')
        cm = cp.ConfusionMatrix;
    end

    if isempty(cm)
        error('无法从 classperf 结果中提取混淆矩阵。');
    end

    nCorrect = sum(diag(cm));
    nTotal = sum(cm(:));
    accPct = (nCorrect / max(nTotal, 1)) * 100;
end

function [accPct, nCorrect, nTotal, p] = cross_validate_local(X, y, indices, method, decoderVerbose)
    % Manual CV loop to ensure train_decoder/predict_decoder are used.
    y = y(:);
    classes = unique(y, 'sorted');
    nClass = numel(classes);
    cm = zeros(nClass);

    K = max(indices);
    for k = 1:K
        test = (indices == k);
        train = ~test;

        XTrain = X(train, :);
        yTrain = y(train, :);
        XTest = X(test, :);
        yTest = y(test, :);

        % Fit zscore on train, apply to test
        [XTrain, mu, sigma] = zscore(XTrain);
        sigma(sigma == 0) = 1;
        XTest = (XTest - mu) ./ sigma;
        XTrain(~isfinite(XTrain)) = 0;
        XTest(~isfinite(XTest)) = 0;

        model = train_decoder(XTrain, yTrain, ...
            'method', method, ...
            'decoder_verbose', decoderVerbose);
        yPred = predict_decoder(XTest, model, ...
            'decoder_verbose', decoderVerbose);

        % Update confusion matrix
        for i = 1:numel(yTest)
            trueIdx = find(classes == yTest(i), 1, 'first');
            predIdx = find(classes == yPred(i), 1, 'first');
            if ~isempty(trueIdx) && ~isempty(predIdx)
                cm(predIdx, trueIdx) = cm(predIdx, trueIdx) + 1;
            end
        end
    end

    nCorrect = sum(diag(cm));
    nTotal = sum(cm(:));
    accPct = (nCorrect / max(nTotal, 1)) * 100;

    chance = 1 / nClass;
    p = binomialTest(nCorrect, nTotal, chance, 'one');
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
