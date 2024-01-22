% XOR Table
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0 1 1 0];

% Rx2 matrix of min and max values for R input elements.
pr = [0 1; 0 1];

% Create a feedforward backpropagation network 
net = newff(pr, [2,1], {'tansig', 'logsig'}, 'trainlm');

% Configure the neural network for this dataset
net = configure(net, inputs, targets);

% Train the network
net.trainParam.epochs = 100;
net = train(net, inputs, targets);

% Test the network
outputs = sim(net, inputs);

% View the network
view(net);

% Convert the decimal output to 1's and 0's 
binaryOutput = outputs > 0.5;

% Display the outputs with the corresponding inputs
disp('Input 1 | Input 2 | Output | Target');
for i = 1:size(inputs, 2)
    fprintf('    %d    |    %d    |  %.4f  |   %d\n', inputs(1,i), inputs(2,i), outputs(i), targets(i));
end

% Check if the output is correctly classifying the input
correctness = binaryOutput == targets;
disp('Correctness of each input:');
disp(correctness);