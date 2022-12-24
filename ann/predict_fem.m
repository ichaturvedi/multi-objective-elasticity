function rmse = predict_fem(inputfile, outputfile)

data = importdata(inputfile);
data = data(:,[1:5,end])';

numTimeStepsTrain = floor(0.8*size(data,2));

dataTrain = data(:,1:numTimeStepsTrain+1);
dataTest = data(:,numTimeStepsTrain+1:end);

mu = mean(mean(dataTrain(1:end-1,:)));
sig = std(std(dataTrain(1:end-1,:)));

dataTrainStandardized = (dataTrain(1:end-1,:) - mu) / sig;

XTrain = dataTrainStandardized;
YTrain = dataTrain(end,:);

%XTrain = normalize(XTrain, 'range', [-1 1]);
%YTrain = normalize(YTrain, 'range', [0 1]);

numFeatures = 5;
numResponses = 1;
numHiddenUnits = 5;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

dataTestStandardized = (dataTest(1:end-1,:) - mu) / sig;
XTest = dataTestStandardized;

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,XTrain(:,end));

for i = 1:size(XTest,2)
    [net,YPred(i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

YTest = dataTest(end,:);
rmse = sqrt(mean((YPred-YTest).^2))

save outputfile net
end