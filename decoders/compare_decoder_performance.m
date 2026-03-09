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

%% -------------------- 逐方法评估 --------------------
nMethods = numel(cfg.methods);
acc = NaN(nMethods, 1);
pval = NaN(nMethods, 1);
nCorrect = NaN(nMethods, 1);
nTotal = NaN(nMethods, 1);

for i = 1:nMethods
    method = cfg.methods{i};
    fprintf('\n===== Evaluating %s =====\n', method);

    [cp, p] = cross_validate(X, y, ...
        'verbose', false, ...
        'validationMethod', char(cfg.validationMethod), ...
        'K', K, ...
        'classificationMethod', method, ...
        'fold_indices', indices, ...
        'decoder_verbose', cfg.decoderVerbose);

    acc(i) = cp.correctRate * 100;
    pval(i) = p;
    nCorrect(i) = sum(diag(cp.CountingMatrix));
    nTotal(i) = sum(cp.CountingMatrix(:));
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
