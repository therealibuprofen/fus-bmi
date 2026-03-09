function model = train_decoder(trainData, trainLabels, varargin)
% train_decoder  Dispatcher for decoder training.
% Uses FCNN when method='FCNN', otherwise delegates to train_classifier.

method = "CPCA+LDA";
for i = 1:2:numel(varargin)
    if ischar(varargin{i}) || isstring(varargin{i})
        if strcmpi(string(varargin{i}), "method") && i + 1 <= numel(varargin)
            method = string(varargin{i + 1});
            break;
        end
    end
end

if strcmpi(method, "FCNN")
    model = train_classifier_fcnn(trainData, trainLabels, varargin{:});
else
    model = train_classifier(trainData, trainLabels, varargin{:});
end
end
