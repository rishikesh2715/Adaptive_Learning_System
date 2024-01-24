% Clear workspace and close all figures
clear; 
close all;

% Generate the sine wave data
x = linspace(0, 2*pi, 100); % Input data (creates 100 points from 0 to 2*pi)
t = sin(x); % Target data (sine of each point)

% Reshape the data to match the expected format for the newrb function
% Inputs (P) should have the size R-by-Q, and Targets (T) should have the size S-by-Q,
% where R is the number of input features, S is the number of target variables, and Q is the number of samples
P = x';
T = t';

% Define the parameters for the RBF network
goal = 0.0; % Mean squared error goal (set to 0 for a perfect fit)
spread = 1.0; % Spread of radial basis functions
MN = 20; % Maximum number of neurons in the hidden layer
DF = 1; % Number of neurons to add between displays

% Create the RBF network
net = newrb(P, T, goal, spread, MN, DF);

% Simulate the network
Y = net(P);

% Plot the original sine wave against the RBF network output
figure;
plot(P, T, 'bo-', P, Y, 'rx-');
legend('Sine wave (Target)', 'RBF Network Output');
title('Comparison of the original sine wave and RBF network output');
xlabel('Input X');
ylabel('Sine and RBF Output');

% Checking the network performance
perf = perform(net, T, Y);
fprintf('Network performance (MSE): %f\n', perf);