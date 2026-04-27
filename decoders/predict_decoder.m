function class = predict_decoder(testData, model, varargin)
% predict_decoder  Dispatcher for decoder inference.
% Uses FCNN/CNN/CNN+LSTM model when available, otherwise delegates to make_prediction.

persistent printedInferenceMethods
if isempty(printedInferenceMethods)
    printedInferenceMethods = string.empty(1, 0);
end

decoderVerbose = true;
requestedHead = "";
for i = 1:2:numel(varargin)
    if i + 1 <= numel(varargin) && (ischar(varargin{i}) || isstring(varargin{i}))
        if strcmpi(string(varargin{i}), "decoder_verbose")
            decoderVerbose = logical(varargin{i + 1});
        elseif strcmpi(string(varargin{i}), "head")
            requestedHead = string(varargin{i + 1});
        end
    end
end

if isstruct(model) && isfield(model, "method")
    method = string(model.method);
else
    method = "UNKNOWN";
end

if decoderVerbose && ~any(strcmpi(printedInferenceMethods, method))
    fprintf('[predict_decoder] Inference decoder method: %s\n', method);
    printedInferenceMethods(end + 1) = method;
end

if isstruct(model) && isfield(model, "method") && strcmpi(method, "FCNN")
    class = make_prediction_fcnn(testData, model);
elseif isstruct(model) && isfield(model, "method") && strcmpi(method, "CNN")
    class = make_prediction_cnn(testData, model);
elseif isstruct(model) && isfield(model, "method") && strcmpi(method, "CNN_TWOHEAD")
    class = select_requested_head(make_prediction_cnn_twohead(testData, model), requestedHead);
elseif isstruct(model) && isfield(model, "method") && ...
        (strcmpi(method, "CNN+LSTM") || strcmpi(method, "CNN_LSTM") || strcmpi(method, "CNNLSTM"))
    class = make_prediction_cnn_lstm(testData, model);
elseif isstruct(model) && isfield(model, "method") && strcmpi(method, "CNN_LSTM_TWOHEAD")
    class = select_requested_head(make_prediction_cnn_lstm_twohead(testData, model), requestedHead);
else
    class = make_prediction(testData, model);
end
end

function class = select_requested_head(prediction, requestedHead)
if isempty(requestedHead) || (isstring(requestedHead) && all(strlength(requestedHead) == 0))
    class = prediction;
    return;
end

switch lower(strtrim(requestedHead))
    case {"horizontal", "horz", "x"}
        class = prediction(:, 1);
    case {"vertical", "vert", "y"}
        class = prediction(:, 2);
    otherwise
        error('Unknown two-head decoder output "%s".', requestedHead);
end

if isrow(prediction) && isvector(class)
    class = class(1);
end
end
