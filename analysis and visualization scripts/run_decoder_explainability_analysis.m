function results = run_decoder_explainability_analysis(modelOrFile, varargin)
% run_decoder_explainability_analysis
% 对 CNN / CNN+LSTM 解码器做可解释性分析，并与 Searchlight 结果比较。
%
% 主要功能：
%   1. 对单样本或多样本生成 Grad-CAM / 遮挡敏感度热力图
%   2. 将热力图叠加到 fUS neurovascular map 上
%   3. 计算与 Searchlight 高亮区域的重合度（Dice / IoU / precision / recall）
%
% 典型用法：
%   results = run_decoder_explainability_analysis(model, ...
%       'searchlight_file', '/path/to/searchlight_results.mat');
%
%   results = run_decoder_explainability_analysis('/path/to/model.mat', ...
%       'session_run_list', [12 3], ...
%       'searchlight_file', '/path/to/searchlight_results.mat');

p = inputParser;
p.addOptional('modelOrFile', [], @(x) isstruct(x) || ischar(x) || isstring(x) || isempty(x));
p.addParameter('searchlight_file', "", @(x) ischar(x) || isstring(x));
p.addParameter('session_run_list', [], @(x) isnumeric(x) || isempty(x));
p.addParameter('sample_indices', [], @(x) isnumeric(x) || isempty(x));
p.addParameter('num_samples_per_class', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('sample_selection', 'correct_only', @(x) any(strcmpi(string(x), ["all", "correct_only", "misclassified_only"])));
p.addParameter('analysis_method', 'auto', @(x) any(strcmpi(string(x), ["auto", "gradcam", "occlusion", "both"])));
p.addParameter('target_class_source', 'predicted', @(x) any(strcmpi(string(x), ["predicted", "true"])));
p.addParameter('temporal_reduction', 'mean', @(x) any(strcmpi(string(x), ["mean", "max"])));
p.addParameter('heatmap_percentile', 90, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 100);
p.addParameter('occlusion_patch_size', [7 7 1], @(x) isnumeric(x) && numel(x) == 3);
p.addParameter('occlusion_stride', [3 3 1], @(x) isnumeric(x) && numel(x) == 3);
p.addParameter('occlusion_fill_value', 0, @isnumeric);
p.addParameter('searchlight_metric', 'accuracy', @(x) any(strcmpi(string(x), ["accuracy", "angular_error"])));
p.addParameter('searchlight_region_mode', 'top_voxels', @(x) any(strcmpi(string(x), ["top_voxels", "bottom_voxels", "pvalue_threshold"])));
p.addParameter('searchlight_percent', 0.1, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
p.addParameter('searchlight_pvalue_threshold', 1e-2, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('radius_index', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('save_results', true, @islogical);
p.addParameter('output_dir', "", @(x) ischar(x) || isstring(x));
p.parse(modelOrFile, varargin{:});
cfg = p.Results;

model = resolve_model_input(cfg.modelOrFile);
searchlight = resolve_searchlight_input(cfg.searchlight_file);

if isempty(cfg.session_run_list)
    if ~isempty(searchlight) && isfield(searchlight, 'session_run_list')
        cfg.session_run_list = searchlight.session_run_list;
    else
        error('请提供 session_run_list，或者提供包含 session_run_list 的 searchlight_file。');
    end
end

if ~isfield(model, 'inputSize') || numel(model.inputSize) ~= 3
    error('模型缺少 inputSize 信息，无法恢复 [H W T] 输入尺寸。');
end

fprintf('[run_decoder_explainability_analysis] Loading sessions/runs...\n');
data = load_doppler_data([], cfg.session_run_list, true, ...
    get_data_path('path_type', 'doppler'), true, 'multiple');

data.dop = preprocess_data(data.dop, false, {'disk', 2, 0});
[trainData, trainLabels] = extract_training_data_and_labels(data, true, ...
    {'', [], []}, model.inputSize(3));

if ~isequal(size(trainData, 1), model.inputSize(1)) || ~isequal(size(trainData, 2), model.inputSize(2))
    error('训练样本尺寸 [%d %d] 与模型输入尺寸 [%d %d] 不匹配。', ...
        size(trainData, 1), size(trainData, 2), model.inputSize(1), model.inputSize(2));
end

nSamples = size(trainData, 4);
[predictedLabels, predictedScores] = predict_labels_for_batch(model, trainData);
selectedIdx = choose_sample_indices(trainLabels, predictedLabels, cfg);
if isempty(selectedIdx)
    error('没有找到满足条件的样本，请调整 sample_selection 或 sample_indices。');
end

selectedData = trainData(:, :, :, selectedIdx);
selectedTrueLabels = trainLabels(selectedIdx);
selectedPredLabels = predictedLabels(selectedIdx);

analysisMethods = resolve_analysis_methods(model, cfg.analysis_method);
heatVolumes = struct();
heatMaps2D = struct();

for methodIdx = 1:numel(analysisMethods)
    methodName = analysisMethods{methodIdx};
    fprintf('[run_decoder_explainability_analysis] Running %s on %d samples...\n', methodName, numel(selectedIdx));
    maps = zeros([model.inputSize numel(selectedIdx)], 'single');
    validMap = false(1, numel(selectedIdx));
    for i = 1:numel(selectedIdx)
        sample = selectedData(:, :, :, i);
        targetClass = pick_target_class(selectedTrueLabels(i), selectedPredLabels(i), cfg.target_class_source);
        try
            maps(:, :, :, i) = single(compute_sample_explainability(model, sample, targetClass, methodName, cfg));
            validMap(i) = true;
        catch ME
            warning('%s failed for sample %d: %s', methodName, selectedIdx(i), ME.message);
        end
    end

    if ~any(validMap)
        warning('%s 未成功生成任何热力图。', methodName);
        continue;
    end

    maps = maps(:, :, :, validMap);
    heatVolumes.(methodName) = mean(maps, 4, 'omitnan');
    heatMaps2D.(methodName) = reduce_temporal_map(heatVolumes.(methodName), cfg.temporal_reduction);
end

searchlightMask = [];
searchlightMap = [];
searchlightPMap = [];
overlap = struct();
if ~isempty(searchlight)
    [searchlightMap, searchlightPMap] = build_searchlight_display_map(searchlight, cfg);
    searchlightMask = build_searchlight_roi_mask(searchlightMap, searchlightPMap, cfg);
    methodFields = fieldnames(heatMaps2D);
    for i = 1:numel(methodFields)
        methodName = methodFields{i};
        overlap.(methodName) = compute_overlap_metrics(heatMaps2D.(methodName), searchlightMap, searchlightMask, cfg.heatmap_percentile);
    end
end

fig = visualize_results(data, heatMaps2D, searchlightMap, searchlightMask, cfg);

results = struct();
results.config = cfg;
results.modelMethod = string(model.method);
results.selectedSampleIndices = selectedIdx(:);
results.selectedTrueLabels = selectedTrueLabels(:);
results.selectedPredictedLabels = selectedPredLabels(:);
results.selectedPredictedScores = predictedScores(selectedIdx, :);
results.heatVolumes = heatVolumes;
results.heatMaps2D = heatMaps2D;
results.searchlightMap = searchlightMap;
results.searchlightPMap = searchlightPMap;
results.searchlightMask = searchlightMask;
results.overlap = overlap;

if cfg.save_results
    save_outputs(results, fig, cfg, data, model);
end

print_overlap_summary(results.overlap);
end

function model = resolve_model_input(modelOrFile)
if isempty(modelOrFile)
    [fname, fpath] = uigetfile('*.mat', 'Select decoder model MAT file');
    if isequal(fname, 0)
        error('未选择模型文件。');
    end
    modelOrFile = fullfile(fpath, fname);
end

if isstruct(modelOrFile)
    model = modelOrFile;
    return;
end

loaded = load(modelOrFile);
candidateNames = {'model', 'decoder'};
for i = 1:numel(candidateNames)
    if isfield(loaded, candidateNames{i}) && isstruct(loaded.(candidateNames{i}))
        model = loaded.(candidateNames{i});
        return;
    end
end

fields = fieldnames(loaded);
for i = 1:numel(fields)
    candidate = loaded.(fields{i});
    if isstruct(candidate) && isfield(candidate, 'net') && isfield(candidate, 'method')
        model = candidate;
        return;
    end
end

error('在 MAT 文件中未找到模型结构体。');
end

function searchlight = resolve_searchlight_input(searchlightFile)
if nargin == 0 || isempty(searchlightFile) || strlength(string(searchlightFile)) == 0
    searchlight = [];
    return;
end

searchlight = load(searchlightFile);
end

function [labels, scores] = predict_labels_for_batch(model, X)
nSamples = size(X, 4);
scores = zeros(nSamples, numel(model.classNames), 'single');
labels = NaN(nSamples, 1);
for i = 1:nSamples
    [sampleScores, classNames] = predict_scores_single(model, X(:, :, :, i));
    scores(i, :) = sampleScores(:)';
    labels(i) = class_name_to_value(classNames(argmax(sampleScores)), model);
end
end

function idx = choose_sample_indices(trueLabels, predLabels, cfg)
if ~isempty(cfg.sample_indices)
    idx = unique(cfg.sample_indices(:))';
    idx = idx(idx >= 1 & idx <= numel(trueLabels));
    return;
end

switch lower(cfg.sample_selection)
    case 'all'
        candidate = 1:numel(trueLabels);
    case 'correct_only'
        candidate = find(predLabels(:) == trueLabels(:))';
    case 'misclassified_only'
        candidate = find(predLabels(:) ~= trueLabels(:))';
    otherwise
        error('Unknown sample_selection: %s', cfg.sample_selection);
end

classes = unique(trueLabels(candidate), 'stable')';
idx = [];
for c = classes
    classIdx = candidate(trueLabels(candidate) == c);
    idx = [idx, classIdx(1:min(cfg.num_samples_per_class, numel(classIdx)))]; %#ok<AGROW>
end
idx = unique(idx, 'stable');
end

function methods = resolve_analysis_methods(model, requestedMethod)
requestedMethod = lower(string(requestedMethod));
modelMethod = lower(string(model.method));

switch requestedMethod
    case "auto"
        if strcmp(modelMethod, "cnn")
            methods = {'gradcam', 'occlusion'};
        else
            methods = {'occlusion'};
        end
    case "both"
        if strcmp(modelMethod, "cnn")
            methods = {'gradcam', 'occlusion'};
        else
            warning('CNN+LSTM 当前默认只运行 occlusion；gradCAM 将被跳过。');
            methods = {'occlusion'};
        end
    case "gradcam"
        if ~strcmp(modelMethod, "cnn")
            error('gradCAM 当前仅对 CNN 模型启用。CNN+LSTM 请使用 occlusion。');
        end
        methods = {'gradcam'};
    case "occlusion"
        methods = {'occlusion'};
    otherwise
        error('Unknown analysis_method: %s', requestedMethod);
end
end

function targetClass = pick_target_class(trueLabel, predLabel, source)
switch lower(string(source))
    case "predicted"
        targetClass = predLabel;
    case "true"
        targetClass = trueLabel;
    otherwise
        error('Unknown target_class_source: %s', source);
end
end

function heat = compute_sample_explainability(model, sample, targetClass, methodName, cfg)
switch lower(methodName)
    case 'gradcam'
        heat = compute_gradcam_volume(model, sample, targetClass);
    case 'occlusion'
        heat = compute_occlusion_volume(model, sample, targetClass, cfg);
    otherwise
        error('Unknown method: %s', methodName);
end
end

function heat = compute_gradcam_volume(model, sample, targetClass)
if exist('gradCAM', 'file') ~= 2
    error('当前 MATLAB 环境中没有 gradCAM 函数。');
end

[X, ~] = normalize_input_for_model(model, sample);
targetClassName = value_to_class_name(targetClass, model);
featureLayer = pick_last_conv_layer(model.net);

try
    scoreMap = gradCAM(model.net, X, char(targetClassName), 'FeatureLayer', featureLayer);
catch
    scoreMap = gradCAM(model.net, X, char(targetClassName));
end

scoreMap = squeeze(gather_numeric(scoreMap));
if ismatrix(scoreMap)
    heat = repmat(scoreMap, 1, 1, model.inputSize(3));
else
    heat = scoreMap;
end

heat = imresize3(single(heat), model.inputSize, 'linear');
heat = normalize_positive_map(heat);
end

function heat = compute_occlusion_volume(model, sample, targetClass, cfg)
[baselineScores, classNames] = predict_scores_single(model, sample);
targetIdx = find(classNames == value_to_class_name(targetClass, model), 1, 'first');
if isempty(targetIdx)
    error('目标类别不在模型类别列表中。');
end

[h, w, t] = size(sample);
patch = min(double(cfg.occlusion_patch_size(:)'), [h w t]);
stride = max(ones(1, 3), double(cfg.occlusion_stride(:)'));
importance = zeros(h, w, t, 'single');
counts = zeros(h, w, t, 'single');

rowStarts = compute_starts(h, patch(1), stride(1));
colStarts = compute_starts(w, patch(2), stride(2));
timeStarts = compute_starts(t, patch(3), stride(3));

for r = rowStarts
    rIdx = r:min(h, r + patch(1) - 1);
    for c = colStarts
        cIdx = c:min(w, c + patch(2) - 1);
        for tt = timeStarts
            tIdx = tt:min(t, tt + patch(3) - 1);
            occluded = sample;
            occluded(rIdx, cIdx, tIdx) = cfg.occlusion_fill_value;
            occScores = predict_scores_single(model, occluded);
            dropScore = baselineScores(targetIdx) - occScores(targetIdx);
            importance(rIdx, cIdx, tIdx) = importance(rIdx, cIdx, tIdx) + single(dropScore);
            counts(rIdx, cIdx, tIdx) = counts(rIdx, cIdx, tIdx) + 1;
        end
    end
end

counts(counts == 0) = 1;
heat = importance ./ counts;
heat = normalize_positive_map(heat);
end

function starts = compute_starts(len, patch, stride)
starts = 1:stride:len;
if isempty(starts) || starts(end) ~= max(1, len - patch + 1)
    starts = unique([starts, max(1, len - patch + 1)]);
end
end

function map2D = reduce_temporal_map(map3D, reductionName)
switch lower(string(reductionName))
    case "mean"
        map2D = mean(map3D, 3, 'omitnan');
    case "max"
        map2D = max(map3D, [], 3);
    otherwise
        error('Unknown temporal_reduction: %s', reductionName);
end
end

function [scores, classNames] = predict_scores_single(model, sample)
[X, nSamples] = normalize_input_for_model(model, sample);
if nSamples ~= 1
    error('predict_scores_single expects exactly one sample.');
end

method = lower(string(model.method));
classNames = string(model.classNames(:));

switch method
    case "cnn"
        rawScores = predict(model.net, X);
    case {"cnn+lstm", "cnn_lstm", "cnnlstm"}
        seq = {permute(X(:, :, :, :, 1), [1 2 4 3])};
        rawScores = predict(model.net, seq);
    otherwise
        error('当前 explainability 只支持 CNN / CNN+LSTM。');
end

scores = squeeze(gather_numeric(rawScores));
scores = reshape(scores, 1, []);
end

function [X, nSamples] = normalize_input_for_model(model, sample)
X = single(sample);
inputSize = double(model.inputSize(:)');

switch ndims(X)
    case 2
        if size(X, 2) ~= prod(inputSize)
            error('输入特征数与模型 inputSize 不匹配。');
        end
        nSamples = size(X, 1);
        X = reshape(X', [inputSize 1 nSamples]);
    case 3
        if ~isequal(size(X), inputSize)
            error('输入尺寸与模型 inputSize 不匹配。');
        end
        nSamples = 1;
        X = reshape(X, [inputSize 1 1]);
    case 4
        if isequal(size(X, 1), inputSize(1)) && isequal(size(X, 2), inputSize(2)) && isequal(size(X, 3), inputSize(3))
            nSamples = size(X, 4);
            X = reshape(X, [inputSize 1 nSamples]);
        else
            error('4D 输入尺寸与模型 inputSize 不匹配。');
        end
    case 5
        if ~isequal([size(X, 1), size(X, 2), size(X, 3)], inputSize) || size(X, 4) ~= 1
            error('5D 输入尺寸与模型 inputSize 不匹配。');
        end
        nSamples = size(X, 5);
    otherwise
        error('不支持的输入维度：%d', ndims(X));
end

if ~isfield(model, 'normalizeInNetwork') || ~model.normalizeInNetwork
    X = (X - model.mu) ./ model.sigma;
    X(~isfinite(X)) = 0;
end
end

function layerName = pick_last_conv_layer(net)
layerName = "";
for i = numel(net.Layers):-1:1
    layer = net.Layers(i);
    if isa(layer, 'nnet.cnn.layer.Convolution3DLayer') || isa(layer, 'nnet.cnn.layer.Convolution2DLayer')
        layerName = string(layer.Name);
        return;
    end
end
if strlength(layerName) == 0
    error('未找到卷积层，无法运行 Grad-CAM。');
end
end

function name = value_to_class_name(classValue, model)
classNames = string(model.classNames(:));
classValues = model.classValues(:);
idx = find(classValues == classValue, 1, 'first');
if isempty(idx)
    error('无法将类别值 %g 映射到 classNames。', classValue);
end
name = classNames(idx);
end

function value = class_name_to_value(className, model)
classNames = string(model.classNames(:));
classValues = model.classValues(:);
idx = find(classNames == string(className), 1, 'first');
if isempty(idx)
    numericValue = str2double(string(className));
    if isnan(numericValue)
        error('无法将类别名 %s 映射到 classValues。', className);
    end
    value = numericValue;
else
    value = classValues(idx);
end
end

function idx = argmax(x)
[~, idx] = max(x, [], 2);
idx = idx(1);
end

function map = normalize_positive_map(map)
map = gather_numeric(map);
map(~isfinite(map)) = 0;
map = max(map, 0);
mx = max(map(:));
if mx > 0
    map = map ./ mx;
end
end

function [metricMap, pMap] = build_searchlight_display_map(searchlight, cfg)
radiusIndex = min(cfg.radius_index, size(searchlight.sl_percentCorrect_combined, 3));
pMap = [];

switch lower(cfg.searchlight_metric)
    case 'accuracy'
        metricMap = searchlight.sl_percentCorrect_combined(:, :, radiusIndex);
        if isfield(searchlight, 'sl_pvalue_combined')
            pMap = maybe_fdr_correct(searchlight.sl_pvalue_combined(:, :, radiusIndex));
        end
    case 'angular_error'
        metricMap = searchlight.sl_angularError_combined(:, :, radiusIndex) * 180 / pi;
        if isfield(searchlight, 'sl_angularError_pvalue_combined')
            pMap = maybe_fdr_correct(searchlight.sl_angularError_pvalue_combined(:, :, radiusIndex));
        end
    otherwise
        error('Unknown searchlight_metric: %s', cfg.searchlight_metric);
end
end

function mask = build_searchlight_roi_mask(metricMap, pMap, cfg)
valid = isfinite(metricMap);
values = metricMap(valid);
mask = false(size(metricMap));
if isempty(values)
    return;
end

switch lower(cfg.searchlight_region_mode)
    case 'top_voxels'
        thresh = quantile(values, 1 - cfg.searchlight_percent);
        mask = metricMap >= thresh;
    case 'bottom_voxels'
        thresh = quantile(values, cfg.searchlight_percent);
        mask = metricMap <= thresh;
    case 'pvalue_threshold'
        if isempty(pMap)
            error('Searchlight 文件中没有 p-value，无法使用 pvalue_threshold。');
        end
        mask = pMap <= cfg.searchlight_pvalue_threshold;
    otherwise
        error('Unknown searchlight_region_mode: %s', cfg.searchlight_region_mode);
end

mask = mask & valid;
end

function corrected = maybe_fdr_correct(pMap)
corrected = pMap;
if exist('mafdr', 'file') ~= 2
    return;
end

p = pMap(:);
valid = isfinite(p);
if ~any(valid)
    return;
end

q = mafdr(p(valid), 'BHFDR', true);
corrected = NaN(size(p));
corrected(valid) = q;
corrected = reshape(corrected, size(pMap));
end

function metrics = compute_overlap_metrics(heatMap, searchlightMap, searchlightMask, heatPercentile)
metrics = struct();
valid = isfinite(heatMap);
if ~isempty(searchlightMap)
    valid = valid & isfinite(searchlightMap);
end

heatValid = heatMap(valid);
if isempty(heatValid)
    return;
end

heatThresh = quantile(heatValid, heatPercentile / 100);
heatMask = heatMap >= heatThresh;
heatMask = heatMask & valid;

inter = nnz(heatMask & searchlightMask);
unionCount = nnz(heatMask | searchlightMask);
heatCount = nnz(heatMask);
searchCount = nnz(searchlightMask);

metrics.heatThreshold = heatThresh;
metrics.heatMask = heatMask;
metrics.dice = safe_divide(2 * inter, heatCount + searchCount);
metrics.iou = safe_divide(inter, unionCount);
metrics.precision = safe_divide(inter, heatCount);
metrics.recall = safe_divide(inter, searchCount);
metrics.intersectionPixels = inter;
metrics.heatPixels = heatCount;
metrics.searchlightPixels = searchCount;

if ~isempty(searchlightMap)
    heatVec = heatMap(valid);
    searchVec = searchlightMap(valid);
    if numel(heatVec) >= 3
        metrics.pearson = corr(heatVec(:), searchVec(:), 'rows', 'complete', 'type', 'Pearson');
        metrics.spearman = corr(heatVec(:), searchVec(:), 'rows', 'complete', 'type', 'Spearman');
    end
end
end

function out = safe_divide(a, b)
if b == 0
    out = NaN;
else
    out = a / b;
end
end

function fig = visualize_results(data, heatMaps2D, searchlightMap, searchlightMask, cfg)
methodFields = fieldnames(heatMaps2D);
nTiles = numel(methodFields) + ~isempty(searchlightMap);
if ~isempty(searchlightMask)
    nTiles = nTiles + 1;
end

fig = figure('Name', 'Decoder Explainability', 'Color', 'w');
tiledlayout('flow');

pixelsize = 0.1;
[yPix, xPix] = size(data.neurovascular_map);
X_img_mm = pixelsize/2 + (0:xPix-1)*pixelsize;
Z_img_mm = pixelsize/2 + (0:yPix-1)*pixelsize + data.UF.Depth(1);

for i = 1:numel(methodFields)
    nexttile;
    methodName = methodFields{i};
    plotDuplexImage(X_img_mm, Z_img_mm, heatMaps2D.(methodName), data.neurovascular_map, ...
        'colormap2use', inferno, ...
        'nonlinear_bg', 2, ...
        'showColorbar', true, ...
        'ColorBarTitle', sprintf('%s importance', upper(methodName)));
    title(sprintf('%s on anatomy', upper(methodName)));
end

if ~isempty(searchlightMap)
    nexttile;
    colorscale = inferno;
    if strcmpi(cfg.searchlight_metric, 'accuracy')
        colorscale = viridis;
    end
    plotDuplexImage(X_img_mm, Z_img_mm, searchlightMap, data.neurovascular_map, ...
        'colormap2use', colorscale, ...
        'nonlinear_bg', 2, ...
        'showColorbar', true, ...
        'ColorBarTitle', sprintf('Searchlight %s', cfg.searchlight_metric));
    title('Searchlight map');
end

if ~isempty(searchlightMask) && ~isempty(methodFields)
    nexttile;
    combined = double(searchlightMask);
    refHeatMap = heatMaps2D.(methodFields{1});
    refHeatVals = refHeatMap(isfinite(refHeatMap));
    heatMask = refHeatMap >= quantile(refHeatVals, cfg.heatmap_percentile / 100);
    combined(heatMask & ~searchlightMask) = 2;
    combined(searchlightMask & ~heatMask) = 1;
    combined(searchlightMask & heatMask) = 3;
    imagesc(X_img_mm, Z_img_mm, combined);
    axis image;
    colormap(gca, [0 0 0; 0.2 0.7 1.0; 1.0 0.4 0.1; 1.0 0.9 0.1]);
    colorbar('Ticks', [0 1 2 3], 'TickLabels', {'bg', 'searchlight', 'heat only', 'overlap'});
    title(sprintf('Overlap (%s)', upper(methodFields{1})));
    xlabel('mm');
    ylabel('mm');
end
end

function save_outputs(results, fig, cfg, data, model)
if strlength(string(cfg.output_dir)) == 0
    outputDir = get_data_path('path_type', 'output');
else
    outputDir = char(cfg.output_dir);
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
if size(data.session_run_list, 1) == 1
    tag = sprintf('S%dR%d', data.session_run_list(1, 1), data.session_run_list(1, 2));
else
    tag = 'multipleSessions';
end

baseName = sprintf('decoder_explainability_%s_%s_%s', strrep(char(model.method), '+', '_'), tag, timestamp);
save(fullfile(outputDir, [baseName '.mat']), 'results');
saveas(fig, fullfile(outputDir, [baseName '.png']));
fprintf('[run_decoder_explainability_analysis] Saved results to %s\n', outputDir);
end

function print_overlap_summary(overlap)
methodFields = fieldnames(overlap);
if isempty(methodFields)
    return;
end

fprintf('\n[run_decoder_explainability_analysis] Overlap summary\n');
for i = 1:numel(methodFields)
    m = overlap.(methodFields{i});
    fprintf('  %s: Dice=%.3f, IoU=%.3f, Precision=%.3f, Recall=%.3f', ...
        upper(methodFields{i}), m.dice, m.iou, m.precision, m.recall);
    if isfield(m, 'pearson') && ~isempty(m.pearson)
        fprintf(', Pearson=%.3f, Spearman=%.3f', m.pearson, m.spearman);
    end
    fprintf('\n');
end
fprintf('\n');
end

function value = gather_numeric(value)
if isa(value, 'dlarray')
    value = extractdata(value);
end
if isa(value, 'gpuArray')
    value = gather(value);
end
end
