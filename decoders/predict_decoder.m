function class = predict_decoder(testData, model)
% predict_decoder  Dispatcher for decoder inference.
% Uses FCNN model when available, otherwise delegates to make_prediction.

if isstruct(model) && isfield(model, "method") && strcmpi(string(model.method), "FCNN")
    class = make_prediction_fcnn(testData, model);
else
    class = make_prediction(testData, model);
end
end
