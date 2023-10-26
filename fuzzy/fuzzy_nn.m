function fuzzy_nn(inputfile, labels)

close all
delete(allchild(groot))

tic

wa = -1;
wb = 1;
wc = 1;
ba = -1;
bb = -1;
bc = 1;

emotions = ["happy","neutral","angry","surprise"];

x = importdata(inputfile); % 22 pw. 23 bw. conc mv
mv = importdata(labels);

mv2 = mv(mv==1 | mv==2 | mv==6 | mv==7);
mv2(mv2==6) = 3;
mv2(mv2==7) = 4;

mv3 = transformtarget(mv2,4);

x = normalize(x, 'range', [-1 1]);
x2 = x(:,[38 40 48 98 114 120 121 128 129 132 134]);

x3 = x2(mv==1 | mv==2 | mv==6 | mv==7,:);

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

for i=1:4
  outid = sprintf('cl%d',i);
  edgeFIS = addOutput(edgeFIS,[0 1],'Name',outid);
  edgeFIS = addMF(edgeFIS,outid,'trimf',[wa wb wc],'Name',emotions(i));
  edgeFIS = addMF(edgeFIS,outid,'trimf',[ba bb bc],'Name','low');
end

r1 = "If ln8 is high and ln6 is low then cl1 is happy";
r2 = "If ln8 is low and ln1 is high and ln2 is low and ln7 is high and ln6 is high then cl2 is neutral";
r3 = "If ln8 is high and ln6 is high and ln3 is high and ln10 is high then cl3 is angry";
r4 = "If ln9 is high and ln7 is high and ln3 is low and ln4 is low then cl4 is surprise";

edgeFIS = addRule(edgeFIS,[r1 r2 r3 r4]);
[in,out,rule] = getTunableSettings(edgeFIS);

opt = tunefisOptions("Method","ga")
opt.MethodOptions.MaxGenerations = 20;
opt.MethodOptions.PopulationSize = 200;
opt.MethodOptions.UseParallel = true;
opt.MethodOptions.UseVectorized = false;
opt.MethodOptions.CrossoverFraction = 0.1;
opt.MethodOptions.CrossoverFcn = @crossoverscattered;
opt.MethodOptions.MutationFcn = @mutationadaptfeasible;

fisout = tunefis1(edgeFIS,[in;out;rule],x3,mv3,opt);
save fisout.mat fisout;

Ieval2 = zeros(size(mv3));
for ii = 1:size(x3,1)
    Ieval2(ii,:) = evalfis(fisout,x2(ii,:));
end

[M, I] = max(Ieval2');
cm = confusionmat(I,mv2)
tp = sum(diag(cm))

toc

end
