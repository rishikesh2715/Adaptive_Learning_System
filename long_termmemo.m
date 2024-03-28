% Sine Wave
data = sin(linspace(0, 4 * pi, 1000))';
numTimeStepsTrain = floor(0.9*numel(data));
dataTrain = data(1:numTimeStepsTrain);
dataTest = data(numTimeStepsTrain+1:end);

% Normalize Training Data
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;

% Prepare Data for LSTM
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
XTrain = num2cell(XTrain', 1);
YTrain = num2cell(YTrain', 1);

% Define LSTM Network Architecture
numFeatures = 1;
numHiddenUnits = 200;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(1)
    regressionLayer];

% Specify Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 250, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 125, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% Train LSTM Network
net = trainNetwork(XTrain, YTrain, layers, options);