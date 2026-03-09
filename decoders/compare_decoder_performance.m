%% compare_decoder_performance
% 在完全相同的交叉验证折分下，对比原解码器与 FCNN 的性能。
%
% 数据要求（.mat 文件中至少包含一组）:
%   1) data (nSamples x nFeatures), labels (nSamples x 1 or 1 x nSamples)
%   2) train_data (yPix x xPix x nTimepoints x nSamples), train_labels

clear;
clc;

%% -------------------- 配置 --------------------
cfg.dataFile = "decoder_training_data.mat";  % 修改为你的数据
cfg.validationMethod = "kFold";              % "kFold" 或 "leaveOneOut"
cfg.K = 10;                                  % 仅在 kFold 生效
cfg.randomSeed = 42;
cfg.decoderVerbose = true;                   % 打印是否使用 FCNN

% 需要对比的方法（原解码器 + FCNN）
cfg.methods = {"CPCA+LDA", "PCA+LDA", "FCNN"};

%% -------------------- 读取数据 --------------------
rng(cfg.randomSeed);
s = load(cfg.dataFile);
[X, y] = parseInputData(s);

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
function [X, y] = parseInputData(s)
    hasDataLabels = isfield(s, "data") && isfield(s, "labels");
    hasTrainData = isfield(s, "train_data") && isfield(s, "train_labels");

    if hasDataLabels
        X = s.data;
        y = s.labels;
    elseif hasTrainData
        td = s.train_data;
        X = reshape(td, [], size(td, 4))';
        y = s.train_labels;
    else
        error(['未找到可识别字段。需要 (data, labels) 或 ' ...
               '(train_data, train_labels)。']);
    end

    if isrow(y)
        y = y';
    end
end
