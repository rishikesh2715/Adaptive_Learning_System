function mlp_xor

    % XOR input and output
    X = [0 0; 0 1; 1 0; 1 1];
    Y = [0; 1; 1; 0];

    % Network configuration
    inputSize = 2; % number of features
    hiddenLayerSizes = [2, 2]; % number of nodes in hidden layers
    outputSize = 1; % number of output nodes
    learnRate = 0.1; % learning rate
    maxEpochs = 20000; % max number of epochs
    epsilon = 1e-3; % error threshold

    % Weights initialization
    W1 = rand(hiddenLayerSizes(1), inputSize) - 0.5;
    b1 = rand(hiddenLayerSizes(1), 1) - 0.5;
    W2 = [];
    b2 = [];
    if length(hiddenLayerSizes) > 1
        W2 = rand(hiddenLayerSizes(2), hiddenLayerSizes(1)) - 0.5;
        b2 = rand(hiddenLayerSizes(2), 1) - 0.5;
    end
    W3 = rand(outputSize, hiddenLayerSizes(end)) - 0.5;
    b3 = rand(outputSize, 1) - 0.5;

    % Training loop
    for epoch = 1:maxEpochs
        for i = 1:size(X, 1)
            % Forward pass
            a1 = sigmoid(W1 * X(i, :)' + b1);
            a2 = [];
            if ~isempty(W2)
                a2 = sigmoid(W2 * a1 + b2);
            end
            if isempty(a2)
                output = sigmoid(W3 * a1 + b3);
            else
                output = sigmoid(W3 * a2 + b3);
            end

            % Error calculation
            error = Y(i) - output;

            % Backpropagation
            delta3 = error .* sigmoid_prime(output);
            delta2 = [];
            if ~isempty(W2)
                delta2 = (W3' * delta3) .* sigmoid_prime(a2);
            end
            delta1 = [];
            if isempty(delta2)
                delta1 = (W3' * delta3) .* sigmoid_prime(a1);
            else
                delta1 = (W2' * delta2) .* sigmoid_prime(a1);
            end

            % Weights update
            if ~isempty(delta2)
                W3 = W3 + learnRate * delta3 * a2';
                b3 = b3 + learnRate * delta3;
                W2 = W2 + learnRate * delta2 * a1';
                b2 = b2 + learnRate * delta2;
            else
                W3 = W3 + learnRate * delta3 * a1';
                b3 = b3 + learnRate * delta3;
            end
            W1 = W1 + learnRate * delta1 * X(i, :);
            b1 = b1 + learnRate * delta1;

        end

        % Display training progress
        if mod(epoch, 1000) == 0
            disp(['Epoch: ' num2str(epoch) ' Error: ' num2str(sum(abs(error)))]);
        end

        % Check convergence
        if sum(abs(error)) < epsilon
            disp(['Converged at epoch: ' num2str(epoch)]);
            break;
        end
    end

    % Testing
    for i = 1:size(X, 1)
        a1 = sigmoid(W1 * X(i, :)' + b1);
        a2 = [];
        if ~isempty(W2)
            a2 = sigmoid(W2 * a1 + b2);
        end
        if isempty(a2)
            output = sigmoid(W3 * a1 + b3);
        else
            output = sigmoid(W3 * a2 + b3);
        end
        disp(['Input: ' num2str(X(i, :)) ' Expected: ' num2str(Y(i)) ' Output: ' num2str(round(output))]);
    end
end

% Sigmoid activation function
function s = sigmoid(z)
    s = 1 ./ (1 + exp(-z));
end

% Derivative of the sigmoid function
function s_prime = sigmoid_prime(z)
    s_prime = z .* (1 - z);
end




