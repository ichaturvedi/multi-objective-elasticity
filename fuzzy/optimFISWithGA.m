function varargout = optimFISWithGA(pdata,fitnessFcn,options,kFoldData)
%

%

%  Copyright 2018-2023 The MathWorks, Inc.

%% Define an optimization problem
p = fuzzy.tuning.internal.createGAProblemStructure;
p.nvars = pdata.nvars;
p.lb = pdata.lbdiscset;
p.ub = pdata.ubdiscset;
p.options = options.MethodOptions;
if isempty(p.options.InitialPopulationMatrix)
    p.options.InitialPopulationMatrix = pdata.x0;
else
    if size(p.options.InitialPopulationMatrix,2) ~= pdata.nvars
        error(message("fuzzy:general:errTunefis_invalidGAPopulation",pdata.nvars))
    end
    for i = 1:size(p.options.InitialPopulationMatrix,1)
        p.options.InitialPopulationMatrix(i,:) = ...
            fuzzy.tuning.internal.mapMFIndicesToIds(...
            p.options.InitialPopulationMatrix(i,:),pdata.discset);
    end
    p.lb = min([p.lb;p.options.InitialPopulationMatrix]);
    p.ub = max([p.ub;p.options.InitialPopulationMatrix]);
end
if isempty(p.options.InitialPopulationRange)
    p.options.InitialPopulationRange = [p.lb;p.ub];
else
    if size(p.options.InitialPopulationRange,2) ~= pdata.nvars
        error(message("fuzzy:general:errTunefis_invalidGAPopulationRange",pdata.nvars))
    end
    for i = 1:2
        p.options.InitialPopulationRange(i,:) = ...
            fuzzy.tuning.internal.mapMFIndicesToIds(...
            p.options.InitialPopulationRange(i,:),pdata.discset);
    end
end

%% Start parallel pool if specified.
fuzzy.tuning.internal.startParallelPool(options.UseParallel)

%% Optimize parameters
% Initialize variables.
optionalOutput = [];
totalRuntime = 0;
totalFcnCount = 0;
errorMessage = [];
if options.KFoldValue > 1 || ~isempty(options.TuningOutputSinks)
    % Create data handle to save the optimization results for the best
    % validation cost.
    data = fuzzy.tuning.internal.FISTuningData;
else
    data = fuzzy.tuning.internal.FISTuningData.empty;
end
preData = fuzzy.tuning.internal.FISTuningData.empty;

% Run the k-fold iterations.
for k = 1:options.KFoldValue
    
    % Set fitness function.
    if isempty(fitnessFcn)
        evalFcn = @(fis)fuzzy.tuning.internal.evaluateFISFitness(fis, ...
            kFoldData.input(k).training,kFoldData.output(k).training, ...
            options.DistanceMetric);
    else
        evalFcn = fitnessFcn;
    end
    p.fitnessfcn = @(x)fuzzy.tuning.internal.fisFitnessFcnWithoutIntCon(...
        x,pdata.fis,pdata.spec,evalFcn,pdata.discset, ...
        options.IgnoreInvalidParameters,false,options.TuningOutputSinks);
    
    % Set output function.
    if options.KFoldValue>1 || ~isempty(options.TuningOutputSinks)
        data.K = k;
        evalFcn = @(fis)fuzzy.tuning.internal.evaluateFISFitness(fis, ...
            kFoldData.input(k).validation,kFoldData.output(k).validation, ...
            options.DistanceMetric);
        ftnFcn = @(x)fuzzy.tuning.internal.fisFitnessFcnWithoutIntCon(...
            x,pdata.fis,pdata.spec,evalFcn,pdata.discset, ...
            options.IgnoreInvalidParameters,false,[]);
        p.options.OutputFcn = [options.MethodOptions.OutputFcn; ...
            @(x,v,s)fuzzy.tuning.internal.gaOutputFcn(x,v,s,ftnFcn,options,data)];
    end
    
    % Invoke solver.
    startTic = tic;
    load bestpop.mat;
    %im = zeros(200,150);
    %im(1:100,:) = population;
    p.options.InitialPopulationMatrix = population;
    [params,fval,exitflag,output,population,scores] = ga(p);
    save bestpop.mat population;
    totalRuntime = totalRuntime + toc(startTic);
    totalFcnCount = totalFcnCount + output.funccount;
    fuzzy.tuning.internal.addExitMessageToSink(data,options.TuningOutputSinks,output.message)
    
    % Display the minimum validation cost.
    if options.KFoldValue > 1
        if isempty(data.MethodData)
            if isempty(preData)
                valCost = Inf;
                trnCost = Inf;
            else
                valCost = preData.BestScore;
                trnCost = preData.MethodData.Best(end);
            end
        else
            valCost = data.BestScore;
            trnCost = data.MethodData.Best(end);
        end
        if any(options.Display == ["all" "validationonly"])
            fprintf(['\n' getString(message('fuzzy:general:msgTunefis_displayFormat')) '\n'],...
                k,valCost,trnCost);
        end
        if ~isempty(options.TuningOutputSinks)
            msg = sprintf(['\n' getString(message('fuzzy:general:msgTunefis_displayFormat')) '\n'],...
                k,valCost,trnCost);
            options.TuningOutputSinks.ResultsDocument.append(msg)
            pause(0.1)
        end
    end
    
    % Construct outputs.
    if options.KFoldValue > 1
        % Use the optimization results corresponding to the best validation
        % cost.
        if isempty(data.MethodData)
            if ~isempty(preData)
                params = preData.BestParams;
                fval = preData.MethodData.Best(end);
                population = preData.MethodData.Population;
                scores = preData.MethodData.Score;
            end
        else
            params = data.BestParams;
            fval = data.MethodData.Best(end);
            population = data.MethodData.Population;
            scores = data.MethodData.Score;
        end
    end
    mparams = fuzzy.tuning.internal.mapIdsToMFIndices(params,pdata.discset);
    mpopulation = population;
    for i = 1:size(mpopulation,1)
        mpopulation(i,:) = fuzzy.tuning.internal.mapIdsToMFIndices(...
            mpopulation(i,:),pdata.discset);
    end
    tuningOutputs = struct('x',mparams,'fval',fval,'exitflag',exitflag, ...
        'output',output,'population',mpopulation,'scores',scores);
    
    [varargout{1},me] = constructOutputFIS(pdata.fis,pdata.spec,mparams, ...
        options.IgnoreInvalidParameters);

    if ~isempty(options.TuningOutputSinks) && options.TuningOutputSinks.Stop
        if isempty(varargout{1})
            varargout{1} = pdata.fis;
        end
    end

    if ~isempty(me)
        if nargout<=1
            throwAsCaller(me)
        else
            errorMessage = me.message;
        end
    end
    
    if nargout > 1
        optionalOutput = [optionalOutput tuningOutputs]; %#ok<AGROW>
    end
        
    if options.KFoldValue > 1
        % Use current FIS output to start the next iteration.
        pdata.fis = varargout{1};
        p.options.InitialPopulationMatrix = population;
        
        % Reset the data handle so that the minimum validation cost of the
        % current iteratin does not affect the next iterations. If we don't
        % reset the data handle, the optimization results of the next
        % iterations might be ignored.
        if ~isempty(data.MethodData)
            if isempty(preData)
                preData = fuzzy.tuning.internal.FISTuningData;
            end
            copy(data,preData)
        end
        reset(data)
    end
    
    if ~isempty(options.TuningOutputSinks) && options.TuningOutputSinks.Stop
        break
    end
end

% Remove duplicate rules from the output FIS.
if ~isempty(varargout{1})
    varargout{1} = fuzzy.tuning.internal.removeRulesWithSameDescription(varargout{1});
    varargout{1} = fuzzy.tuning.internal.removeRulesWithSameAntecedent(varargout{1});
end

% Provide the second output if requested by the user.
if nargout > 1
    varargout{2} = struct(...
        'tuningOutputs',optionalOutput, ...
        'totalFcnCount',totalFcnCount, ...
        'totalRuntime',totalRuntime, ...
        'errorMessage',errorMessage);
end

end
%% Helper functions
function [fis,me] = constructOutputFIS(fis,spec,params,ignoreInvalidParams)

me = [];
try
    fis = setPVec(fis,spec,params,ignoreInvalidParams,true);
catch me
    fis = [];
end

end