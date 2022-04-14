clear
addpath('functions')
%% Load
%Load the weights and biases of the neural net.
e = 0;
basePath = 'Nano-Photonics/MILP/Experiments/E%i';
experimentPath = sprintf(basePath,e);
run([experimentPath '/params.m'])

load([experimentPath '/' NET_WEIGHTS])
% Define number of workers for parallelization (arg) (default = 128)
parpool(POOLSIZE)
%% The optimization loop
tic
for n_layer= 2:size(w_numpy,2)/2
    % Define the subnet for bound computation
    w_b_net = w_numpy(1:2*(n_layer));
    % Determine the size and depth of the subnets
    net_ind = size(w_b_net,2);
    depth_ind = net_ind/2;
    
    % Define the range of the inputs (arg)
    lower_bound_all{1} = INPUT_LOWER_BOUND;
    upper_bound_all{1} = INPUT_UPPER_BOUND;
    
    % Create placeholder
    lower_bound = zeros(1,size(w_numpy{2*n_layer-1},2));
    upper_bound = zeros(1,size(w_numpy{2*n_layer-1},2)); 
    
    % Parallel for loop
    if EXPERIMENT_TYPE == 1
        parfor n_neuron = 1:size(w_numpy{2*n_layer-1},2)
            disp(n_neuron)
            [lower_bound(n_neuron), upper_bound(n_neuron)] = loop_main(n_neuron, lower_bound_all, upper_bound_all, depth_ind, w_b_net);
        end
%     elseif EXPERIMENT_TYPE == 2
%         parfor n_neuron = 1:size(w_numpy{2*n_layer-1},2)
%             disp(n_neuron);
%             [lower_bound(n_neuron), upper_bound(n_neuron)] = loop_main2(n_neuron, lower_bound_all, upper_bound_all, depth_ind, w_b_net);
% 
%         end 
    end
    % Stack calculated bounds
    lower_bound_all{n_layer} = lower_bound;
    upper_bound_all{n_layer} = upper_bound;
    
    
end
time = toc;
save([experimentPath '/' BOUNDS_OUTPUT_FILENAME],'lower_bound_all','upper_bound_all','time')
delete(gcp('nocreate'))

