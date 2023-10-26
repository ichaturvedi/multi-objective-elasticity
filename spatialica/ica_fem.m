function ica_fem(inputfile, outputfile)

data = importdata(inputfile);
label = data(:,end);
datax = data(:,1:end-1)';

num_pca = 30; %max(n-1,m)
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

%datax2 = data_spa*diag(barrier2)*data_tem';

barrier2 = rescale(barrier2, 0.01, 1);

cnt1 = floor(size(datax,2)/num_pca);
barrier3 = repmat(barrier2,1,cnt1+1);

datax2 = datax*diag(barrier3(1,1:size(datax,2)));

for i=1:num_pca

   landx = datax(1:2:136,i);
   landy = datax(2:2:136,i);

   landx2 = datax2(1:2:136,i);
   landy2 = datax2(2:2:136,i);

   minx = min(landx);
   maxx = max(landx);
   miny = min(landy);
   maxy = max(landy);

   landx2 = rescale(landx2,minx,maxy);
   landy2 = rescale(landy2,miny,maxy);

   landnew = [landx2 landy2]';
   datax2(:,i)=landnew(:);

end

datax2 = datax2';

data_new = [datax2 label];
dlmwrite(outputfile,data_new',",");

end
