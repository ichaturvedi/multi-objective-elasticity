function fuzzy_nn(inputfile,labels)

%clear
%inputfile = 'cafes.txt';
%labels = 'cafelabels.txt';

close all
delete(allchild(groot))

tic

wa = -1;
wb = 1;
wc = 1;
ba = -1;
bb = -1;
bc = 1;

emotions = ["angry","happy"];

nclass = 2;

data = importdata(inputfile); % 22 pw. 23 bw. conc mv
x = data;
mv = importdata(labels);
mv2 = mv;
mv3 = transformtarget(mv2,nclass);

x = rescale(x,-1,1);
x2 = x;
x3 = x2;

% train and test
n1 = floor(0.7*size(x3,1));
trainx = x3(1:n1,:);
testx = x3(n1+1:end,:);

trainy = mv3(1:n1,:);
trainy2 = mv2(1:n1,:);
testy = mv2(n1+1:end,:);

edgeFIS = mamfis('Name','Emotion');

for i=1:size(x2,2)
  landid = sprintf('ln%d',i);
  edgeFIS = addInput(edgeFIS,[-1 1],'Name',landid);
  edgeFIS = addMF(edgeFIS,landid,'trimf',[wa wb wc],'Name','low');
  edgeFIS = addMF(edgeFIS,landid,'trimf',[ba bb bc],'Name','high');
end


wa = 0;
wb = 1;
wc = 1;
ba = 0;
bb = 0;
bc = 1;

for i=1:nclass
  outid = sprintf('cl%d',i);
  edgeFIS = addOutput(edgeFIS,[0 1],'Name',outid);
  edgeFIS = addMF(edgeFIS,outid,'trimf',[wa wb wc],'Name',emotions(i));
  edgeFIS = addMF(edgeFIS,outid,'trimf',[ba bb bc],'Name','low');
end

% prior rules using a decision tree
r1 = " ln2 is high and ln1 is low and ln3 is low then cl1 is angry";
r2 = " ln2 is low and ln4 is low and ln1 is high then cl2 is happy";
r3 = " ln2 is low and ln4 is low and ln1 is low then cl2 is happy";
r4 = " ln2 is low and ln4 is high and ln4 is low then cl1 is angry";
r5 = " ln2 is high and ln1 is high and ln4 is low then cl2 is happy";
r6 = " ln2 is low and ln4 is high and ln4 is high then cl1 is angry";
r7 = " ln2 is high and ln1 is high and ln4 is high then cl1 is angry";
r8 = " ln2 is high and ln1 is low and ln3 is high then cl2 is happy";

edgeFIS = addRule(edgeFIS, [r1 r2 r3 r4 r5 r6 r7 r8]);  
[in,out,rule] = getTunableSettings(edgeFIS);

opt = tunefisOptions("Method","ga");
opt.OptimizationType = "learning";
opt.NumMaxRules = 40;
opt.MethodOptions.MaxGenerations = 200;
opt.MethodOptions.PopulationSize = 50;
opt.UseParallel = true;
opt.MethodOptions.UseVectorized = false;
opt.MethodOptions.CrossoverFraction = 0.2;
opt.MethodOptions.CrossoverFcn = @crossoverscattered;
opt.MethodOptions.MutationFcn = @mutationadaptfeasible;

%load fisouts.mat;
fisout = tunefis1(edgeFIS,[in;out;rule],trainx,trainy,opt);
%fisout = tunefis1(fisout,[in;out;rule],trainx,trainy,opt);
save fisouts.mat fisout;

Ieval2 = zeros(size(testy,1),nclass);
for ii = 1:size(testy,1)
    Ieval2(ii,:) = evalfis(fisout,testx(ii,:));
end

[M, I] = max(Ieval2');
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

%fmeaavg = mean(fmea);
%accavg = mean(acc1);

end

save cafes_result_fuz.mat Ieval2 fmea acc1 cm2


end
