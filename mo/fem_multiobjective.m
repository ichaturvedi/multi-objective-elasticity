function y = fem_multiobjective(x)
%SIMPLE_MULTIOBJECTIVE is a simple multi-objective fitness function.
%
% The multi-objective genetic algorithm solver assumes the fitness function
% will take one input x where x is a row vector with as many elements as
% number of variables in the problem. The fitness function computes the
% value of each objective function and returns the vector value in its one
% return argument y.

%   Copyright 2007 The MathWorks, Inc. 

load salt.mat; %load net
act = x*net.IW{1,1}';

x2 = rescale(x,0,1);
pw = x2(1:23);
bw = x2(24:45);
y(1) = 50-sum(pw);
y(2) = sum(bw);

load fisouto.mat;
act = rescale(act,-1,1);
fuz = evalfis(fisout,act)
[M I] = max(fuz);
if M == 0.5
    y(3) = 2;
end
y(3) = I;

end