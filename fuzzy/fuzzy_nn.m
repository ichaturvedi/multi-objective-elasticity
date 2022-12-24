function fuzzy_nn(inputfile, outputfile)

close all
delete(allchild(groot))

data = importdata("../spatialica/happyica.csv"); % 22 pw. 23 bw. conc mv
num = 136;
x = data(:,1:num);
mv = data(:,end);
mv(mv>2)=1;

x = normalize(x, 'range', [-1 1]);
mv = normalize(mv, 'range', [0 1]);

edgeFIS = mamfis('Name','Emotion');
sx = 0.1;

for i=1:num
  wellname = sprintf('ln%d',i);
  edgeFIS = addInput(edgeFIS,[-1 1],'Name',wellname);
  edgeFIS = addMF(edgeFIS,wellname,'gaussmf',[sx 0],'Name','zero');
end

edgeFIS = addOutput(edgeFIS,[0 1],'Name','Iout');

wa = 0.1;
wb = 1;
wc = 1;
ba = 0;
bb = 0;
bc = 0.7;
edgeFIS = addMF(edgeFIS,'Iout','trimf',[wa wb wc],'Name','happy');
edgeFIS = addMF(edgeFIS,'Iout','trimf',[ba bb bc],'Name','neutral');

figure
subplot(2,1,1)
plotmf(edgeFIS,'input',1)
title('x')
subplot(2,1,2)
plotmf(edgeFIS,'output',1)
title('Iout')

% happy
r1 = "If ln109 is zero and ln99 is zero then Iout is happy";
r2 = "If ln109 is zero and ln99 is not zero then Iout is neutral";
r3 = "If ln109 is not zero and ln6 is zero then Iout is happy";
r4 = "If ln109 is not zero and ln6 is not zero then Iout is neutral";

edgeFIS = addRule(edgeFIS,[r1 r2 r3 r4]);
edgeFIS.Rules
[in,out,rule] = getTunableSettings(edgeFIS);
opt = tunefisOptions("Method","ga")
opt.MethodOptions.MaxGenerations = 5;
fisout = tunefis(edgeFIS,[in;out;rule],x,mv,opt);

Ieval = zeros(size(mv));
for ii = 1:size(x,1)
    Ieval(ii,:) = evalfis(edgeFIS,x(ii,:));
end

Ieval2 = zeros(size(mv));
for ii = 1:size(x,1)
    Ieval2(ii,:) = evalfis(fisout,x(ii,:));
end

sqrt(mse(mv,Ieval))
sqrt(mse(mv,Ieval2))

save happy.mat fisout;

end