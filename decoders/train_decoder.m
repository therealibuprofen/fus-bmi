function model = train_decoder(trainData, trainLabels, varargin)
% train_decoder  Dispatcher for decoder training.
% Uses FCNN when method='FCNN', otherwise delegates to train_classifier.

persistent printedMethods
if isempty(printedMethods)
    printedMethods = string.empty(1, 0);
end

method = "CPCA+LDA";
decoderVerbose = true;

forwardArgs = {};
i = 1;
while i <= numel(varargin)
    key = varargin{i};
    if (ischar(key) || isstring(key)) && i + 1 <= numel(varargin)
        keyStr = string(key);
        val = varargin{i + 1};
        if strcmpi(keyStr, "method")
            method = string(val);
            forwardArgs = [forwardArgs, varargin(i:i+1)]; %#ok<AGROW>
        elseif strcmpi(keyStr, "decoder_verbose")
            decoderVerbose = logical(val);
        else
            forwardArgs = [forwardArgs, varargin(i:i+1)]; %#ok<AGROW>
        end
        i = i + 2;
    else
        forwardArgs = [forwardArgs, varargin(i)]; %#ok<AGROW>
        i = i + 1;
    end
end

if decoderVerbose && ~any(strcmpi(printedMethods, method))
    fprintf('[train_decoder] Using decoder method: %s\n', method);
    printedMethods(end + 1) = method;
end

if strcmpi(method, "FCNN")
    model = train_classifier_fcnn(trainData, trainLabels, forwardArgs{:});
else
    model = train_classifier(trainData, trainLabels, forwardArgs{:});
end
end
