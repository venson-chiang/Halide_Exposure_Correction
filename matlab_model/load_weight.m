clear;
clc
close all;

modelName = fullfile('matlab_model','model.mat');
model = load(modelName);
k = size(model.net.Learnables, 1);

for k = 1 : 2 : k-1
    
    %output weight
    weight = model.net.Learnables.Value{k, 1};
    weight_ex = extractdata(weight);
    weight_double = double(weight_ex);
    weight_doublet = permute(weight_double, [2 1 3 4]);
    
    weight_name = 'ml_variables/' + model.net.Learnables{k, 1} + '_weight.data';
    wid = fopen(weight_name, 'w');
    fwrite(wid, weight_doublet, 'double');
    fclose(wid);
    
    w_shape = size(weight_doublet);
    
    w_shape_name = 'ml_variables/' + model.net.Learnables{k, 1} + '_wshape.data';
    wsid = fopen(w_shape_name, 'w');
    fwrite(wsid, size(w_shape, 2), 'int');
    for n = 1 : 4
        fwrite(wsid, w_shape(n), 'int');
    end
    fclose(wsid);
    
    % output bias
    bias = model.net.Learnables.Value{k+1, 1};
    bias_ex = extractdata(bias);
    bias_double = double(bias_ex);
    
    bias_name = 'ml_variables/' + model.net.Learnables{k, 1} + '_bias.data';
    bid = fopen(bias_name, 'w');
    fwrite(bid, bias_double, 'double');
    fclose(bid);
    
    b_shape = size(bias_double);
    b_shape_name = 'ml_variables/' + model.net.Learnables{k, 1} + '_bshape.data';
    bsid = fopen(b_shape_name, 'w');
    fwrite(bsid, 1, 'int');
    fwrite(bsid, b_shape(3), 'int');
    fclose(wsid);
   
end

disp("Success!");


