%clear all and close all window
clear; 
close all;

%choose which dataset to use
%datasetName = 'iris';
datasetName = 'simplecluster';

% Load the dataset
if strcmp(datasetName, 'iris')
    [x, ~] = iris_dataset;
else 
    [x, ~] = simplecluster_dataset;
end

% SOM parameters
dimensions = [8 8];
coverSteps = 100;
initNeighbor = 3;
topologyFcn = 'hextop';
distanceFcn = 'linkdist';

% Create SOM
selfOrgMap = selforgmap(dimensions, coverSteps, initNeighbor, topologyFcn, distanceFcn);

% Train SOM
selfOrgMap.trainParam.epochs = 500;
selfOrgMap = train(selfOrgMap, x);

% Visualization based on selected dataset
figure, plotsompos(selfOrgMap, x), title([datasetName ' Dataset - Neuron Positions']);
figure, plotsomhits(selfOrgMap, x), title([datasetName ' Dataset - SOM Hit Map']);
figure, plotsomnd(selfOrgMap), title([datasetName ' Dataset - Neuron Distances']);