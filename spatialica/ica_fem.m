function ica_fem(inputfile, labelfile, outputfile, outputlabel)

data = importdata(inputfile);
label = importdata(labelfile);

num_pca = 30; %max(n-1,m)
num_ica = 10;
[coeff,score,latent]=pca(data','NumComponents',num_pca);

data_spa = score;
data_tem = coeff;

ica_spa = rica(data_spa, num_ica);
data_spa2 = ica_spa.TransformWeights;

ica_tem = rica(data_tem,num_ica);
data_tem2 = ica_tem.TransformWeights;

% compute barrier
C = 5;
k1 = ones(1,num_ica)*0.01;
M = 200;
sumg = data_spa2 + data_tem2;
deno = log(2*C*C)/(C*C-latent'*latent);
gaint = k1*(data_spa2 + data_tem2)';
num2 = (data_spa2+data_tem2)/(C*C-latent'*latent);
num1 = M*latent(1:num_pca)'*data_spa2;

barrier = num1*num2';
barrier2 = (barrier - gaint)/deno;

%datax2 = data_spa*diag(barrier2)*data_tem';

barrier2 = rescale(barrier2, 0.01, 1);

cnt1 = floor(size(data,2)/num_pca);
barrier3 = repmat(barrier2,1,cnt1+1);

datax2 = data + 0.3*data*diag(barrier3(1,1:size(data,2)));
data_new = [datax2 label];

%idx = randperm(size(data_new, 1));
idx = 1:size(data_new,1);
data_new2 = data_new(idx, :);

datax = data_new2(:,1:end-1);
datay = data_new2(:,end);

dlmwrite(outputfile,datax,",");
dlmwrite(outputlabel,datay,",");

end
