function varargout = tunefis(fisin,spec,varargin)
% TUNEFIS Tunes a fuzzy inference system or a tree of fuzzy inference systems
%
%   FISOUT = TUNEFIS(FISIN,TUNESET,IN,OUT) tunes FISIN using TUNESET, IN,
%   and OUT; and returns FISOUT. FISIN is a scalar 
%     - mamfis or sugfis object, or
%     - fistree object.
%   TUNESET is an array of tunable parameter settings of inputs, outputs,
%   and rules in FISIN. IN and OUT are input and output training data,
%   respectively. FISOUT is the optimized version of FISIN. In this case,
%   TUNEFIS uses default tunefisOptions.
%   
%   FISOUT = TUNEFIS(FISIN,TUNESET,IN,OUT,OPTIONS) tunes FISIN using
%   TUNESET, IN, OUT, and OPTIONS; and returns FISOUT. OPTIONS is created
%   using tunefisOptions.
%
%   FISOUT = TUNEFIS(FISIN,TUNESET,CUSTCOSTFCN,OPTIONS) tunes FISIN using
%   TUNESET, CUSTCOSTFCN, and OPTIONS; and returns FISOUT. CUSTCOSTFCN is a
%   function handle to a custom cost function. The custom cost function
%   evaluates FISOUT to calculate its cost with respect to an evaluation
%   criteria, such as input/output data.
%   CUSTCOSTFCN must accept at least one input argument for FISOUT and 
%   returns a cost value. You can provide an anonymous function handle to
%   attach additional data for fitness calculation, e.g.
%
%     function fitness = CUSTCOST(fis,trainingData)
%         ...
%     end
%     CUSTCOSTFCN = @(fis)CUSTCOST(fis,trainingData);
%   
%   [~,OPTOUTPUTS] = TUNEFIS(...) returns optional outputs OPTOUTPUTS,
%   which is a structure containing the following fields:
%     - tuningOutputs: Structure containing algorithm-specific tuning
%                      outputs. For k-fold cross validation, tuningOutputs
%                      is a structure array of length k.
%     - totalFcnCount: Total number of evaluations of the cost function.
%     - totalRuntime : Total execution time of the tuning process.
%     - errorMessage : Error message generated when updating FISIN with new
%                      parameter values.
%   To find tuning algorithm specific outputs, see the help for the
%   specific tuning method; for example,help ga. To specify the tuning
%   method, use tunefiOptions/Method. The default method is "ga".
%
%   Example:
%
%     %% Create initial FIS.
%     x = (0:0.1:10)';
%     y = sin(2*x)./exp(x/5);
%     options = genfisOptions('GridPartition');
%     options.NumMembershipFunctions = 5;
%     fisin  = genfis(x,y,options);            
%     [in,out,rule] = getTunableSettings(fisin);
%
%     %% Tune MF parameters with "anfis"
%     fisout = tunefis(fisin,[in;out],x,y,tunefisOptions("Method","anfis"));
%
%     %% Tune only input MF parameters with "ga"
%     fisout = tunefis(fisin,in,x,y,tunefisOptions("Method","ga"));
%
%     %% Tune only output MF parameters with "particleswarm"
%     fisout = tunefis(fisin,out,x,y,tunefisOptions("Method","particleswarm"));
%
%     %% Tune only rule parameters with "patternsearch"
%     fisout = tunefis(fisin,out,x,y,tunefisOptions("Method","patternsearch"));
%
%     %% Tune input, output, and rule parameters with "simulannealbnd"
%     fisout = tunefis(fisin,[in;out;rule],x,y,tunefisOptions("Method","simulannealbnd"));
%
%     %% Tune with custom parameter settings.
%     % Do not tune input 1
%     in(1) = setTunable(in(1),false);
%     % Do not tune MFs 1 and 2 of output 1
%     out(1).MembershipFunctions(1:2) = setTunable(out(1).MembershipFunctions(1:2),false);
%     out(1).MembershipFunctions(3).Parameters.Free = false;
%     out(1).MembershipFunctions(4).Parameters.Minimum = -2;
%     out(1).MembershipFunctions(5).Parameters.Maximum = 2;
%     % Do not tune rules 1 and 2.
%     rule(1:2) = setTunable(rule(1:2),false);
%     rule(3).Antecedent.Free = false;
%     rule(4).Antecedent.AllowNot = true;
%     rule(3).Consequent.AllowEmpty = false;
%     fisout = tunefis(fisin,[in;out;rule],x,y,tunefisOptions("Method","ga"));
%
%     %% Tune type-2 FIS.
%     fisin = mamfistype2;
%     fisin = addInput(fisin,[0 10],'NumMFs',5);
%     fisin = addOutput(fisin,[0 1],'NumMFs',5);
%     [in,out] = getTunableSettings(fisin);
%     rng('default')
%     fisout = tunefis(fisin,[in;out],x,y,tunefisOptions("OptimizationType","learning"));
%
%     %% Tune rules with k-fold cross validation.
%     fisin = sugfis;
%     fisin = addInput(fisin,[0 10],'NumMFs',5);
%     fisin = addOutput(fisin,[0 1],'NumMFs',5);
%     fisin = addRule(fisin,repmat([1 1 1 1],[5,1]));
%     [~,~,rule] = getTunableSettings(fisin);
%     options = tunefisOptions('KFoldValue',2,'ValidationTolerance',0, ...
%         'ValidationWindowSize',1);
%     rng('default')
%     [fisout,optOutputs] = tunefis(fisin,rule,x,y,options);
%
%   See also
%     mamfis sugfis mamfistype2 sugfistype2 fistree getTunableSettings tunefisOptions

%  Copyright 2018-2019 The MathWorks, Inc.

%% Input validation
narginchk(3,5)

% Check FIS input.
type = validateFISInput(fisin);

numInputs = numel(fisin.Inputs);
numOutputs = numel(fisin.Outputs);

enableStructuralChecks = false;
if ~fisin.DisableStructuralChecks
    fisin.DisableStructuralChecks = true;
    enableStructuralChecks = true;
end

% Check parameter setting input.
specIsEmpty = isempty(spec);
if ~specIsEmpty && ~isa(spec,'fuzzy.tuning.internal.ParameterSettings')
    error(message("fuzzy:general:errTunefis_invalidParameterSettings"))
end

% Check training data or user specified function handle.
validNumericInput = @(in)~isempty(in) && ...
    (isnumeric(in) && ismatrix(in) && all(isfinite(in(:))) && isreal(in));
validOptions = @(o)~isempty(o) && isscalar(o) && isa(o,'tunefisOptions');
trainingDataIsSpecified = false;
x = [];
y = [];
if isnumeric(varargin{1})
    if ~validNumericInput(varargin{1})
        error(message("fuzzy:general:errTunefis_invalidInputData"))
    end
    
    %% Case 1: User provides training data.
    % Check the 2nd input argument. If it is numeric (i.e. input training
    % data), user must provide the 3rd numeric input argument (i.e. output
    % training data) with same row number. In this case, the 4th input
    % argument is optional, which is an option object.
    if numel(varargin) < 2
        error(message("fuzzy:general:errTunefis_missingOutputTrainingData"))
    end
    
    x = varargin{1};
    if isvector(x)
        n = length(x);
        if n == numInputs
            % Single data set.
            if iscolumn(x)
                x = x';
            end        
        else
            % Multiple data sets.
            if mod(n,numInputs) == 0
                row = n/numInputs;
                x = reshape(x,[numInputs row])';
            else            
                error(message("fuzzy:general:errTunefis_invalidInputDataVector"))
            end
        end
    else
        % Transpose data if data sets are specified column-wise.
        if size(x,2)~=numInputs && size(x,1)==numInputs
            x = x';
        end
    end
    
    numCols = size(x,2);
    
    if numCols ~= numInputs
        error(message("fuzzy:general:errTunefis_invalidSizeOfInputData"))
    end
    
    y = varargin{2};
    if ~validNumericInput(y)
        error(message("fuzzy:general:errTunefis_invalidOutputData"))
    end
    if isvector(y)
        n = length(y);
        if n == numOutputs
            % Single data set.
            if iscolumn(y)
                y = y';
            end        
        else
            % Multiple data sets.
            if mod(n,numOutputs) == 0
                row = n/numOutputs;
                y = reshape(y,[numOutputs row])';
            else            
                error(message("fuzzy:general:errTunefis_invalidOutputDataVector"))
            end
        end
    end
    
    if size(y,2) ~= numOutputs
        error(message("fuzzy:general:errTunefis_invalidSizeOfOutputData"))
    end
    
    if size(x,1) ~= size(y,1)
        error(message("fuzzy:general:errTunefis_mismatchedIODataPoints"))
    end
    
    if numel(varargin) > 2
        if ~validOptions(varargin{3})
            error(message("fuzzy:general:errTunefis_invalidOptions"))
        end
        options = varargin{3};
    else
        options = tunefisOptions("Method","ga");
    end
    
    fitnessFcn = [];
    
    trainingDataIsSpecified = true;
    
    % Number of k-fold cross validations must be less than or equal to
    % the size of input/output data set.
    if options.KFoldValue > size(x,1)
        error(message("fuzzy:general:errTunefis_kFoldValueGtDataSize"))
    end
    
else
    %% Case 2: User provides custom function handle for FIS evaluation.
    % In this case, the 3rd input argument must be a function handle having
    % at least one input argument and one output argument. The 4th input
    % argument, if specified, must be an option object.
    
    if ~isa(varargin{1},'function_handle')
        error(message("fuzzy:general:errTunefis_invalidThirdInput"))
    end
    
    fh = varargin{1};
    if isempty(fh) || ~isscalar(fh)
        error(message("fuzzy:general:errTunefis_emptyOrNonscalarFcnHandle"))
    end    
    
    try
        numFcnIn = abs(nargin(fh));
        numFcnOut = abs(nargout(fh));
    catch me
        error(message("fuzzy:general:errTunefis_emptyOrNonscalarFcnHandle"))
    end

    if numFcnIn<1 || numFcnOut<1
        error(message("fuzzy:general:errTunefis_emptyOrNonscalarFcnHandle"))
    end
    
    if numel(varargin) > 1
        if ~validOptions(varargin{2})
            error(message("fuzzy:general:errTunefis_invalidOptions"))
        end
        options = varargin{2};
    else
        options = tunefisOptions("Method","ga");
    end
    
    fitnessFcn = varargin{1};
    
    if options.KFoldValue >= 2
        error(message("fuzzy:general:errTunefis_kFoldNotSupportedForCustomFitnessFcn"))
    end
    
end
if isa(fisin,'fistree')
    fisin.IgnoreValidation = true;
end

% Update method display according to tunefis options.
options = fuzzy.tuning.internal.updateMethodDisplay(options);

if options.Method == "anfis"
    if type ~= "FIS"
        error(message("fuzzy:general:errTunefis_fistreeWithANFIS"))
    end
    if isa(fisin,'fuzzy.internal.fis.Type2FuzzyInferenceSystem')
        error(message("fuzzy:general:errTunefis_T2FISNotSupported"))
    end
    try
        options.MethodOptions.InitialFIS = fisin;
    catch me
        throw(me)
    end
    if ~specContainAllIOParams(fisin,spec)
        error(message("fuzzy:general:errTunefis_notAllIOParamsWithANFIS"))
    end
    if ~trainingDataIsSpecified
        error(message("fuzzy:general:errTunefis_fcnHandleWithANFIS"))
    end
    inputData = [varargin{1} varargin{2}];
    [varargout{1:nargout}] = fuzzy.tuning.internal.optimFISWithANFIS(inputData,options.MethodOptions);
else
    if any(strcmp(properties(options.MethodOptions),'UseParallel'))
        options.MethodOptions.UseParallel = options.UseParallel;
        if options.UseParallel
            options.MethodOptions.UseVectorized = false;
        end
    end    
    if specIsEmpty
        if options.OptimizationType == "tuning"
            error(message("fuzzy:general:errTunefis_emptySettings"))
        else
            spec = fuzzy.tuning.VariableSettings.empty;
        end
    end
        
    % Get FIS problem data.
    pdata = fuzzy.tuning.internal.createFISProblemData(fisin,spec,options);
    
    % Get k-fold data.
    kFoldData = fuzzy.tuning.internal.getKFoldData(x,y,options);
    
    % Saturate KFoldvalue if less than 2.
    if options.KFoldValue < 2
        options.KFoldValue = 1;
    end
    
    % Invoke the specified optimization method.
    if options.Method == "ga"
        [varargout{1:nargout}] = optimFISWithGA(pdata,fitnessFcn,options,kFoldData);
    elseif options.Method == "particleswarm"
        [varargout{1:nargout}] = fuzzy.tuning.internal.optimFISWithParticleSwarm(pdata,fitnessFcn,options,kFoldData);
    elseif options.Method == "patternsearch"
        [varargout{1:nargout}] = fuzzy.tuning.internal.optimFISWithPatternSearch(pdata,fitnessFcn,options,kFoldData);
    else % "simulannealbnd"
        [varargout{1:nargout}] = fuzzy.tuning.internal.optimFISWithSimulatedAnnealing(pdata,fitnessFcn,options,kFoldData);
    end
end

if isempty(varargout) || isempty(varargout{1})
    return
end

if enableStructuralChecks
    varargout{1}.DisableStructuralChecks = false;
end
if isa(varargout{1},'fistree')
    varargout{1}.IgnoreValidation = false;
end

end
%% Helper functions -------------------------------------------------------
function type = validateFISInput(fisin)
if isempty(fisin) || ~isscalar(fisin) || ~(isa(fisin,'FuzzyInferenceSystem') || ...
        isa(fisin,'fistree'))
    error(message("fuzzy:general:errTunefis_invalidFISInput"))
end

if isa(fisin,'FuzzyInferenceSystem')
    validateFISConsistency(fisin)
    type = "FIS";
else
    validateFISTreeConsistency(fisin)
    type = "FIS tree";
end
end

function validateFISConsistency(fisin)
% If DisableStructuralChecks is enabled, ensure specified FIS is
% consistent.
if fisin.DisableStructuralChecks
    try
        fuzzy.internal.utility.createFromStruct(fisin.convertToStruct);
    catch me
        error(message("fuzzy:general:errTunefis_inconsistentFISState", ...
            fisin.Name))
    end
end

if numel(fisin.Inputs) < 1
    error(message("fuzzy:general:errTunefis_fisHasNoInput", ...
        fisin.Name))
end
numInputMFs = fuzzy.internal.utility.getVarMFs(fisin.Inputs);
id = find(numInputMFs==0,1);
if ~isempty(id)
    error(message("fuzzy:general:errTunefis_fisInputHasNoMF", ...
        id,fisin.Name))
end

if numel(fisin.Outputs) < 1
    error(message("fuzzy:general:errTunefis_fisHasNoOutput", ...
        fisin.Name))
end
numOutputMFs = fuzzy.internal.utility.getVarMFs(fisin.Outputs);
id = find(numOutputMFs==0,1);
if ~isempty(id)
    error(message("fuzzy:general:errTunefis_fisOutputHasNoMF", ...
        id,fisin.Name))
end
end

function validateFISTreeConsistency(fisin)
% If DisableStructuralChecks is enabled, ensure specified FISTREE is
% consistent.
if fisin.DisableStructuralChecks
    temp = removeInconsistency(fisin);
    if ~isequal(fisin.Connections,temp.Connections)
        error(message("fuzzy:general:errTunefis_inconsistentFISTreeConnections"))
    end
    if ~isequal(fisin.Inputs,temp.Inputs)
        error(message("fuzzy:general:errTunefis_inconsistentFISTreeInputs"))
    end
    if ~isequal(fisin.Outputs,temp.Outputs)
        error(message("fuzzy:general:errTunefis_inconsistentFISTreeOutputs"))
    end
end

for i = 1:numel(fisin.FIS)
    validateFISConsistency(fisin.FIS(i))
end
end

function yes = specContainAllIOParams(fisin,spec)
[in,out] = getTunableSettings(fisin);
expectedPVec = getTunableValues(fisin,[in;out]);
actualPVec = getTunableValues(fisin,spec);
yes = isequal(actualPVec,expectedPVec);
end