%clear all and close all window
clear; 
close all;

%dataset to use
[x, ~] = iris_dataset;
%[x, ~] = simplecluster_dataset;

% SOM params
dimensions = [10 10];
coverSteps = 100;
initNeighbor = 3;
topologyFcn = 'hextop';
distanceFcn = 'linkdist';

% create the network
selfOrgMap = selforgmap(dimensions, coverSteps, initNeighbor, topologyFcn, distanceFcn);

% train the network
selfOrgMap = train(selfOrgMap, x);

% test the network
y = selfOrgMap(x);

% view the network
view(selfOrgMap)

% visualization
figure, plotsomtop(selfOrgMap);
figure, plotsomnc(selfOrgMap);
figure, plotsomnd(selfOrgMap);
figure, plotsomplanes(selfOrgMap);
figure, plotsomhits(selfOrgMap,x);
figure, plotsompos(selfOrgMap,x);