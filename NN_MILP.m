clear
addpath('functions')
%% Load
%Load the weights and biases of the neural net.
e = 0;
basePath = 'Nano-Photonics/MILP/Experiments/E%i/';
experimentPath = sprintf(basePath,e);
run([experimentPath 'params.m'])

load([experimentPath NET_WEIGHTS])
% Define number of workers for parallelization (arg) (default = 128)
% delete(gcp('nocreate'))
% parpool(POOLSIZE)
%%
load([experimentPath NET_WEIGHTS])
% Target to be optimized for
load([experimentPath OPTIMIZATION_TARGET])
%Upper bounds and lower bounds
load([experimentPath BOUNDS_OUTPUT_FILENAME])

%% Test vals

l_h = lower_bound_all;
u_h = upper_bound_all;
w_b_net = w_numpy;

% Determine the size and depth of the net
net_ind = size(w_b_net,2);
depth_ind = net_ind/2;

% Number of neurons in the first layer
lay_neuron_size(1) = size(double(w_b_net{1})',1);

% Initializing the MILP reformulation of the neural network
for lay =1:depth_ind
    w{lay} = double(w_b_net{2*lay-1})';
    b{lay} = double(w_b_net{2*lay});
    lay_neuron_size(lay+1) = size(w{lay},2);
end


time_all = [];
optimal_integer_design_all = [];
optimal_integer_spec_all = [];
gt_spec_all = [];

n_target=1:6:100;
for n_cnt=1:size(n_target,2)
%     spec_test = double(selected_coreset(2,:));
    constraints =[];
    target = myspects(:,n_target(n_cnt))';
    
    %coreset size arg
    input_size = size(target,1);
    Z = sdpvar(input_size,201);
    % Define the variables and set the always active and always deactive relus.
    
    

    if EXPERIMENT_TYPE == 'Material_Selection'
        x{1} = sdpvar(input_size,lay_neuron_size(1));
    elseif EXPERIMENT_TYPE == 'Nano-Photonics_Integer'
        x_in = intvar(input_size,lay_neuron_size(1));
        x{1} = x_in/7;
    elseif EXPERIMENT_TYPE == 'Nano-Photonics_Rounded'
        x{1} = sdpvar(input_size,lay_neuron_size(1));
    end
    
    % Initializing the MILP reformulation of the neural network
    for lay =2:depth_ind
        x{lay} = sdpvar(input_size,lay_neuron_size(lay),'full');
        z{lay-1} = binvar(input_size,lay_neuron_size(lay),'full');
        out{lay-1} = x{lay-1} * w{lay-1} + repmat(b{lay-1},[input_size 1]);
    end

    % last layer does not have relu!
    y = x{end} * w{end} + repmat(b{end},[input_size 1]);
    
    % Define the piecewise linear Relu constraints.
    for layer =1:depth_ind-1
        constraints = [constraints,
            x{layer+1} >= out{layer},...
            x{layer+1} >= 0, ...
            x{layer+1} <= repmat(u_h{layer+1},[input_size 1]).*z{layer}, ...
            x{layer+1} <= out{layer}-repmat(l_h{layer+1},[input_size 1]).*(1-z{layer}), ...
            ];
    end

    % Define the selection constraints arg
    if EXPERIMENT_TYPE == 'Material_Selection'
        constraints = [constraints, 0 <= x{1} , x{1} <= repmat(x_b,[input_size 1]), sum(x_b)<=2, Z >= (y-target), Z >= -(y-target)];
    elseif EXPERIMENT_TYPE == 'Nano-Photonics_Integer'
        constraints = [constraints, 0 <= x_in , x_in <= 7, Z >= (y-target), Z >= -(y-target)];
    elseif EXPERIMENT_TYPE == 'Nano-Photonics_Rounded'
        constraints = [constraints, 0 <= x{1} , x{1} <= 1, Z >= (y-target), Z >= -(y-target)];
    end

    % Define the objective
    objective =   sum(sum(Z));
    %Optimization params (arg)
    options = sdpsettings('solver','gurobi', 'gurobi.Threads', 0,  'gurobi.MIPFocus', 0);
    tic
    % Stack the outputs
    sol = optimize(constraints, objective, options);
    time_all(n_cnt) = toc;
    optimal_integer_design_all{n_cnt} = double(x_in);
    optimal_integer_spec_all{n_cnt} = double(y);
    obj_all{n_cnt} = double(objective);
    gt_spec_all{n_cnt} = target;

end
% save name (arg)
save([experimentPath INTEGER_INVERSION_OUTPUT_FILENAME], 'time_all','optimal_integer_design_all','optimal_integer_spec_all','obj_all','gt_spec_all')
