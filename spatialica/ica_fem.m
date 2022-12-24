function ica_fem(inputfile, outputfile)

data = importdata(inputfile);
label = data(:,end)';
datax = data(:,1:end-1)';

num_pca = 5;
num_ica = 10;
[coeff,score,latent]=pca(datax','NumComponents',num_pca);

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

datax2 = data_spa*diag(barrier2)*data_tem';

data_new = [datax2';label];
dlmwrite(outputfile,data_new',",");

end


