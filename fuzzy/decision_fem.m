function decision_fem(inputfile)

close all
delete(allchild(groot))

data = importdata("../spatialica/happyica.csv"); % 22 pw. 23 bw. conc mv
num = 136;
x = data(:,1:num);
x = data(:,[randi(136) randi(136)]);

mv = data(:,end);
x = normalize(x, 'range', [-1 1]);
mv = normalize(mv, 'range', [0 1]);

RMdl = fitctree(x,mv,"MaxNumSplits",4);
view(RMdl,'mode','graph') % graphic description

x1range = min(x(:,1)):.01:max(x(:,1));
x2range = min(x(:,2)):.01:max(x(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];
predictedspecies = predict(RMdl,XGrid);
gscatter(xx1(:), xx2(:), predictedspecies,'rgb');

end