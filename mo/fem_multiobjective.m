function y = fem_multiobjective(x)
%SIMPLE_MULTIOBJECTIVE is a simple multi-objective fitness function.
%
% The multi-objective genetic algorithm solver assumes the fitness function
% will take one input x where x is a row vector with as many elements as
% number of variables in the problem. The fitness function computes the
% value of each objective function and returns the vector value in its one
% return argument y.

%   Copyright 2007 The MathWorks, Inc. 

%y(1) = (x+2)^2 - 10;
%y(2) = (x-2)^2 + 20;x
%size(x)
load '../fuzzy/happy.mat';

% index smile 
%pwx = 49:68; pwy = 49*2+1:49*2+20;
% index eye 1 and 2
bwx1 = 37:42; bwy1 = 37*2+1:37*2+6;
bwx2 = 43:48; bwy2 = 43*2+1:43*2+6;
% index nose
pwx = 29:36; pwy = 29*2+1:29*2+8;

out = [];
for i=1:size(x,1) 
       
    out(i,1)=evalfis(fisout,x(i,:));
    
    data1 = [x(i,pwx);x(i,pwy)];
    [idx1,C1,sumd1,D1] = kmeans(data1,1);

    data2a = [x(i,bwx1);x(i,bwy1)];
    data2b = [x(i,bwx2);x(i,bwy2)];
    [idx2a,C2a,sumd2a,D2a] = kmeans(data2a',1);
    [idx2b,C2b,sumd2b,D2b] = kmeans(data2b',1);

    %default minimize
    % maximise -3
    out(i,2)= (mean(D2a)+mean(D2b)); 
    out(i,3) = 3-mean(D1);        
    
end
y = out;
