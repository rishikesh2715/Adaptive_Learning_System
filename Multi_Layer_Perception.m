% Define the input and target data
inputs = [0 0; 0 1; 1 0; 1 1]';  % Transposed to match MATLAB's format
targets = [0 1 1 0]; % Logical XOR output

% Create a feedforward network with one hidden layer containing 2 neurons
% Using 'tansig' as transfer function for the hidden layer and 'purelin' for the output layer
hiddenLayerSize = 4;
net = feedforwardnet(hiddenLayerSize, 'trainlm');

% Set the transfer function for the hidden layer and the output layer
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';

% Configure the neural network for this dataset
net = configure(net, inputs, targets);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the network
[net, tr] = train(net, inputs, targets);

% Test the network
outputs = net(inputs);

% Calculate errors
errors = gsubtract(targets, outputs);

% View the network
view(net);

% Assess the performance: mean squared error
performance = perform(net, targets, outputs);

% Convert continuous outputs to binary to compare with the binary targets
binaryOutput = outputs > 0.5;

% Check if outputs match targets
isCorrect = binaryOutput == targets;
accuracy = sum(isCorrect) / length(isCorrect);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Display the outputs with the corresponding inputs
disp('Input 1 | Input 2 | Output | Target');
for i = 1:size(inputs, 2)
    fprintf('    %d    |    %d    |  %.4f  |   %d\n', inputs(1,i), inputs(2,i), outputs(i), targets(i));
end

% Check if the output is correctly classifying the input
correctness = binaryOutput == targets;
disp('Correctness of each input:');
disp(correctness);
