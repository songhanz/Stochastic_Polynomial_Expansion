clear all;
close all


for netname = ["resnet321", "resnet3103", "resnet4253"]
    for nonlin = ["relu", "gelu", "tanh"]
        netname = char(netname);
        nonlin  = char(nonlin);
        modeldir = 'models/';
        savedir = 'convW/';
        load([modeldir netname '_' nonlin '.mat'], 'net');
        
        
        %% 
        I      = imread("sherlock.jpg");
        sz     = net.Layers(1).InputSize;
        I      = imresize(I,sz(1:2));
        figure();
        imshow(I)
        I      = im2single(I);
        % %% 
        % counter = 0;
        % for i = 1:numel(net.Layers)
        %     layer = net.Layers(i);
        %     if isa(layer, 'nnet.cnn.layer.Convolution2DLayer')
        %         counter = counter + 1;
        %         disp(layer.PaddingSize);
        %     end
        % end
        %% 
        counter = 0;
        for i = 1:numel(net.Layers)
            layer = net.Layers(i);
            if isa(layer, 'nnet.cnn.layer.ReLULayer')
                counter = counter + 1;
                
            end
        end
        %%
        
        reshaped_convW = cell(numel(net.Layers), 1);
        for i = 1:numel(net.Layers)
            layer = net.Layers(i);
            if isa(layer, 'nnet.cnn.layer.Convolution2DLayer')
                weight_reshape = reshape_conv2_weigths(net, I, layer);
                reshaped_convW{i} = weight_reshape;
                clear weight_reshape
            end
        end
   
        save([savedir netname nonlin '_convW.mat'], 'reshaped_convW', '-v7.3');
    end
end




%%
function weight_reshape = reshape_conv2_weigths(net, image, layer)

% need to feed an image to get dimensions of input
% since 2024a, can use info = analyzeNetwork(net)
rows       = matches(net.Connections.Destination,layer.Name);
source     = net.Connections.Source(rows);
input      = activations(net, image, source{:});

% info of the conv2d layer
filtersize = layer.FilterSize;
% numfilter  = layer.NumFilters;
weights    = layer.Weights;
% bias       = layer.Bias;
stride     = layer.Stride(1);  % only consider the symmetric stride
padding    = layer.PaddingSize;

padinput = padarray_tblr(input, padding);
[a,b,~]  = size(padinput);            % spatial dimension after padding
imageidx = reshape([1:(a*b)], [a,b]); % make each pixel is an index of padded input                   
imageidx_col = im2col(imageidx, filtersize); 


if stride > 1
    % output dimension if stride is 1
    H_output = (floor((a - filtersize(1))) + 1); 
    W_output = (floor((b - filtersize(2))) + 1); 
    idx      = reshape([1: H_output * W_output], H_output, W_output);
    
    % consider stride size, and remove some cols
    idx      = idx(1:stride:H_output, 1:stride:W_output);
    idx      = reshape(idx, 1, []);
    imageidx_col = imageidx_col(:, idx);
end

n_filtered = size(imageidx_col, 2); % H*W after convolution
imageidx_col = imageidx_col + [0:(a*b):(a*b*(n_filtered-1))];

weight_reshape = rearrange_weights(weights, imageidx_col, [a*b, n_filtered]);
weight_reshape = sparse(cell2mat(weight_reshape));
end

% 
% function output = func1(I, weight_reshape, n_filtered, numfilter, bias, padding)
% padI     = padarray(I, padding);
% flatten_padI = reshape(padI, 1, []);
% output = full(double(flatten_padI) * weight_reshape);
% output = reshape(output, sqrt(n_filtered), sqrt(n_filtered), numfilter) + bias;
% 
% end
% 
% 
% function original = func2(I, weights, bias, padding)
% padI     = padarray(I, padding);
% original = dlconv(dlarray(padI, 'SSC'),weights,bias);
% original = extractdata(original);
% end


function weight_reshape = rearrange_weights(weights, imageidx_col, weights_size)
[H,W,C,F]      = size(weights);
n_filtered     = size(imageidx_col, 2);
weights        = reshape(weights, [], C, F);  % flatten spatial dimensions
weights_cell   = squeeze(mat2cell(weights, H*W, ones(1,C), ones(1,F)));
weights_cell   = cellfun(@(X) repmat(X, 1, n_filtered), weights_cell, 'UniformOutput',false);

weight_canvas  = zeros(weights_size);
weight_canvas  = repmat({weight_canvas}, C, F);

weight_reshape = cellfun(@(X, Y) myfunc(X, Y, imageidx_col), weights_cell, weight_canvas, 'UniformOutput',false);

end

function Y = myfunc(X, Y, idx)
Y(idx) = X;
end

function padinput = padarray_tblr(input, padding)
% padarray() function that takes 4 d padding size vector
% namely top bottom left and right padding size
padding_pre  = padding([1,3]);
padding_post = padding([2,4]);

padinput     = padarray(input, padding_pre, 'pre');
padinput     = padarray(padinput, padding_post, 'post');
end