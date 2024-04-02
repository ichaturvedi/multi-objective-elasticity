function ext_features(inputs, targets)

%inputs = load('cafedatas.txt');
%targets = load('cafelabels.txt');

nclass = 2;

[n1 n2] = size(inputs);
ind = floor(0.7*n1);
trainx = inputs(1:ind,:);
trainy = targets(1:ind,:);
testx = inputs(ind+1:end,:);
testy = targets(ind+1:end,:);
trainy2 = trainy;
trainy = transformtarget(trainy,nclass);

%load happy.mat;
net = feedforwardnet(5);
%net.trainParam.epochs = 4;
[net,tr] = train(net,trainx',trainy');
act = inputs*net.IW{1,1}';
dlmwrite('cafes.txt',act);
save cafes.mat net act;
Ieval = net(testx');

[M, I] = max(Ieval);
cm2 = confusionmat(I,testy)
%tp = sum(diag(cm))

% Calculate F-measure
for x=1:nclass

tp = cm2(x,x);
tn = cm2(1,1);
for y=2:nclass
tn = tn+cm2(y,y);
end
tn = tn-cm2(x,x);

fp = sum(cm2(:, x))-cm2(x, x);
fn = sum(cm2(x, :), 2)-cm2(x, x);
pre(x)=tp/(tp+fp+0.01);
rec(x)=tp/(tp+fn+0.01);
fmea(x) = 2*pre(x)*rec(x)/(pre(x)+rec(x)+0.01);
acc1(x) = (tp + tn)/(tp + fp + fn + tn);

end

save cafe_results.mat Ieval fmea acc1 cm2

end