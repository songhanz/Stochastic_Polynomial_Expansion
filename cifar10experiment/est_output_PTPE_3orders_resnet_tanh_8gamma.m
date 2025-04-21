for in_var  = [1 10 100 1000]
depth = [3 2 1; 3 10 3; 4 25 3];
for modeldepth = 1:3
% clear all;
close all;
reset(gpuDevice);

stacks  = depth(modeldepth, :);
netname = ['resnet' sprintf('%d', stacks)];
nonlin  = 'tanh';
savedir = '/scratch1/fs1/shinung/songhan.z/TMLR/';
load([savedir 'models/' netname '_' nonlin '.mat'], 'net');
load([savedir 'convW/' netname nonlin '_convW.mat']);


% put reshaped convolution weights on gpu
reshaped_convW = cellfun(@gpuArray, reshaped_convW, 'UniformOutput', false);
%%
datadir = tempdir; 
downloadCIFARData(datadir);
[~,~,XValidation,~] = loadCIFARData(datadir);
I       = double(XValidation(:,:,:,2));
sz      = net.Layers(1).InputSize;


%% Stochastic Taylor 3rd order

sigma2     = in_var * [1 1 1]; % iid gaussian noises to each color channel
pixel_chan = prod(sz([1,2]));
input_cov  = gpuArray(diag([repmat(sigma2(1), 1, pixel_chan) ...
                            repmat(sigma2(2), 1, pixel_chan) ...
                            repmat(sigma2(3), 1, pixel_chan) ]));
input_mean = gpuArray(I);
% propagate variance
[modelO3_mean, modelO3_cov] = PropCovResNet_Tanh('ST3', stacks, net.Layers, input_mean, input_cov, reshaped_convW);

modelO3_mean = gather(modelO3_mean);
modelO3_cov  = gather(modelO3_cov);

%% Stochastic Taylor 2nd order
[modelO2_mean, modelO2_cov] = PropCovResNet_Tanh('ST2', stacks, net.Layers, input_mean, input_cov, reshaped_convW);

modelO2_mean = gather(modelO2_mean);
modelO2_cov  = gather(modelO2_cov);

%% Stochastic Taylor 1st order (Stochastic linearization)
[modelO1_mean, modelO1_cov] = PropCovResNet_Tanh('ST1', stacks, net.Layers, input_mean, input_cov, reshaped_convW);

modelO1_mean = gather(modelO1_mean);
modelO1_cov  = gather(modelO1_cov);




save([savedir 'regularized/' netname '_' nonlin '_8gamma_preddist_var' num2str(in_var) '.mat'], ...
     'modelO3_mean', 'modelO3_cov', ...
     'modelO2_mean', 'modelO2_cov', ...
     'modelO1_mean', 'modelO1_cov');


end
end

%%

function [new_mean, new_cov] = PropCovResNet_Tanh(type, stacks, layers, input_mean, input_cov, reshaped_convWs)


% ========== initial layers ==========
[new_mean, new_cov]     = ImageInput(layers(1).Mean, input_mean, input_cov);
[new_mean, new_cov, ~]  = CBR(type, layers(2:3), new_mean, new_cov, [], reshaped_convWs{2});      

% ============= stack 1 ==============
start_stack1            = 4;
for i = 1:stacks(1)
    layer_num           = start_stack1 + [1:7];
    [new_mean, new_cov] = ResPackage_standard(type, layers(layer_num), new_mean, new_cov, reshaped_convWs(layer_num));
    start_stack1        = start_stack1 + 7;
end
% ============= stack 2 ==============
start_stack2            = start_stack1;  
[new_mean, new_cov]     = ResPackage_downsample(type, layers(start_stack2 + [1:9]), new_mean, new_cov, reshaped_convWs(start_stack2 + [1:9]));
start_stack2            = start_stack2 + 9;
for i = 1:(stacks(2)-1)
    layer_num           = start_stack2 + [1:7];
    [new_mean, new_cov] = ResPackage_standard(type, layers(layer_num), new_mean, new_cov, reshaped_convWs(layer_num));
    start_stack2        = start_stack2 + 7;
end
% ============= stack 3 ==============
start_stack3            = start_stack2;  
[new_mean, new_cov]     = ResPackage_downsample(type, layers(start_stack3 + [1:9]), new_mean, new_cov, reshaped_convWs(start_stack3 + [1:9]));
start_stack3            = start_stack3 + 9;
for i = 1:(stacks(3)-1)
    layer_num           = start_stack3 + [1:7];
    [new_mean, new_cov] = ResPackage_standard(type, layers(layer_num), new_mean, new_cov, reshaped_convWs(layer_num));
    start_stack3        = start_stack3 + 7;
end
% =========== final layers ===========
n                       = start_stack3 + 1;
[new_mean, new_cov]     = FullyConnect(layers(n), new_mean, new_cov);

end


function [new_mean, new_cov] = ResPackage_standard(type, layers, new_mean_res, new_cov_res, reshaped_convWs)

[new_mean, new_cov, new_xcov] = CBR(type, layers(1:2), new_mean_res, new_cov_res, new_cov_res, reshaped_convWs{1});       
[new_mean, new_cov, new_xcov] = CB(layers(4:5), new_mean, new_cov, new_xcov, 'left', reshaped_convWs{4});   
[new_mean, new_cov]           = Residual(new_mean, new_cov, new_mean_res, new_cov_res, new_xcov);
[new_mean, new_cov, ~]        = Tanh(type, new_mean, new_cov, []);

end

function [new_mean, new_cov] = ResPackage_downsample(type, layers, new_mean_res, new_cov_res, reshaped_convWs)

new_mean = new_mean_res;
new_cov  = new_cov_res;
new_xcov = new_cov_res;
[new_mean, new_cov, new_xcov] = CBR(type, layers(1:2), new_mean, new_cov, new_xcov, reshaped_convWs{1});       
[new_mean, new_cov, new_xcov] = CB(layers(4:5), new_mean, new_cov, new_xcov, 'left', reshaped_convWs{4});  
[new_mean_res, new_cov_res, new_xcov] = CB(layers(8:9), new_mean_res, new_cov_res, new_xcov, 'right', reshaped_convWs{8});   
[new_mean, new_cov]           = Residual(new_mean, new_cov, new_mean_res, new_cov_res, new_xcov);
[new_mean, new_cov, ~]        = Tanh(type, new_mean, new_cov, []);

end



function [new_mean, new_cov, new_xcov] = CBR(type, layers, new_mean, new_cov, new_xcov, reshaped_convW)
% Conv2d + Batchnorm + Tanh
[new_mean, new_cov, new_xcov] = Conv2D(layers(1), new_mean, new_cov, new_xcov, 'left', reshaped_convW);           %if isequal(layers(1).Name, 'conv3'); ii = 8; how_good_is_mean_var_esti(ii, new_cov, new_mean, layers(1), Var_all{ii}, Mean_all{ii}); end
[new_mean, new_cov, new_xcov] = BatchNorm2D(layers(2), new_mean, new_cov, new_xcov, 'left');             %if isequal(layers(2).Name, 'batchnorm3');ii = 9; how_good_is_mean_var_esti(ii, new_cov, new_mean, layers(2), Var_all{ii}, Mean_all{ii}); end
[new_mean, new_cov, new_xcov] = Tanh(type, new_mean, new_cov, new_xcov);        % if isequal(layers(3).Name, 'leakyrelu3');ii = 10; how_good_is_mean_var_esti(ii, new_cov, new_mean, layers(3), Var_all{ii}, Mean_all{ii}); end

end

function [new_mean, new_cov, new_xcov] = CB(layers, new_mean, new_cov, new_xcov, direction, reshaped_convW)
% Conv2d + Batchnorm
[new_mean, new_cov, new_xcov] = Conv2D(layers(1), new_mean, new_cov, new_xcov, direction, reshaped_convW);           %if isequal(layers(1).Name, 'conv3'); ii = 8; how_good_is_mean_var_esti(ii, new_cov, new_mean, layers(1), Var_all{ii}, Mean_all{ii}); end
[new_mean, new_cov, new_xcov] = BatchNorm2D(layers(2), new_mean, new_cov, new_xcov, direction);             %if isequal(layers(2).Name, 'batchnorm3');ii = 9; how_good_is_mean_var_esti(ii, new_cov, new_mean, layers(2), Var_all{ii}, Mean_all{ii}); end

end



% =========================================================================
% ========================== Individual Layers ============================
% =========================================================================


function [new_mean, new_cov]  = ImageInput(layer_mean, new_mean, new_cov)
    % zerocenter
    new_mean = new_mean - double(layer_mean);
end


function [new_mean, new_cov, new_xcov] = Conv2D(layer, new_mean, new_cov, new_xcov, prod_direction, weight_reshape)

    bias         = gpuArray(double(layer.Bias)); 
    padding      = layer.PaddingSize;
    numfilter    = layer.NumFilters;
    stride       = layer.Stride;
    filtersize   = layer.FilterSize;
    [Hi, Wi, Ci]  = size(new_mean);
    H_output     = (floor((Hi - filtersize(1) + sum(padding([1,2])))/stride(1)) + 1); % spatial dimension after filtered
    W_output     = (floor((Wi - filtersize(2) + sum(padding([3,4])))/stride(2)) + 1); % or output dimension  

    padI         = padarray_tblr(new_mean, padding);
    flatten_padI = reshape(padI, 1, []);
    new_mean     = full(flatten_padI * weight_reshape);
    % spatial_dim  = numel(new_mean)/numfilter;  % number of pixels per channel
    % new_mean     = new_mean + repelem(bias, spatial_dim); % keep the output as flattened, all other layers are adjusted accordingly
    new_mean     = reshape(new_mean, H_output, W_output, numfilter) + bias;

    if sum(padding)
        % zero pad the covariance and cross-covariance matrices
        [padded_cov, padded_xcov] = pad_cov_xcov(new_cov, new_xcov, prod_direction, [Hi, Wi, Ci], padding);
        
        % Adding zeros into the covariance matrix will run the positive
        % definiteness. Add a small number to avoid that. Another way to 
        % think of this is a constant has a normal distribution with
        % infinitesimally small variance.
        padded_cov   = padded_cov + eye(size(padded_cov)) * 1e-12;


        new_cov      = full(weight_reshape' * padded_cov * weight_reshape);
        % new_cov      = weight_reshape' * padded_cov * weight_reshape; % sparse matrix
    
        if ~isempty(new_xcov)
            if isequal(prod_direction, 'left')
                new_xcov = full(weight_reshape' * padded_xcov);
                % new_xcov = weight_reshape' * padded_xcov;
            elseif isequal(prod_direction, 'right')
                new_xcov = full(padded_xcov * weight_reshape);
                % new_xcov = padded_xcov * weight_reshape;
            else
                error("Product direction must be one of 'left' and 'right'.");
            end
        else
            % no need to keep track of cross-covariance this time
            % new_xcov  = [];
        end
    else
        new_cov      = full(weight_reshape' * new_cov * weight_reshape);
        % new_cov      = weight_reshape' * new_cov * weight_reshape; % sparse matrix
    
        if ~isempty(new_xcov)
            if isequal(prod_direction, 'left')
                new_xcov = full(weight_reshape' * new_xcov);
                % new_xcov = weight_reshape' * new_xcov;
            elseif isequal(prod_direction, 'right')
                new_xcov = full(new_xcov * weight_reshape);
                % new_xcov = new_xcov * weight_reshape;
            else
                error("Product direction must be one of 'left' and 'right'.");
            end
        else
            % no need to keep track of cross-covariance this time
            % new_xcov  = [];
        end
    end
    
    new_cov = set_min_var(new_cov, 1e-12);
end






function [new_mean, new_cov, new_xcov] = BatchNorm2D(layer, new_mean, new_cov, new_xcov, prod_direction)

    TrainedMean      = gpuArray(double(layer.TrainedMean));
    TrainedVariance  = gpuArray(double(layer.TrainedVariance));
    Offset           = gpuArray(double(layer.Offset));
    Scale            = gpuArray(double(layer.Scale));
    Epsilon          = gpuArray(double(layer.Epsilon));
    nchannel         = numel(Offset);
    spatial_dim      = numel(new_mean)/nchannel; % number of pixels in each channel
    
    w                = Scale ./ sqrt(TrainedVariance + Epsilon);
    new_mean         = (new_mean - TrainedMean) .* w + Offset;

    w                = repelem(squeeze(w), spatial_dim)';
    % w                = double(w);
    new_cov          = w' .* new_cov .* w;

    if ~isempty(new_xcov)
        if isequal(prod_direction, 'left')
            new_xcov = w' .* new_xcov;
        elseif isequal(prod_direction, 'right')
            new_xcov = new_xcov .* w;
        else
            error("Product direction must be one of 'left' and 'right'.");
        end
    else
        % new_xcov  = [];
    end

    new_cov = set_min_var(new_cov, 1e-12);
    
end

function [new_mean, new_cov, new_xcov] = Tanh(type, new_mean, new_cov, new_xcov)   
    % gamma = [0.558267543233651,0.859623142094298,0.859623146088812,1.261245008299549];
    gamma = [0.887009116985261,0.737843341418961,0.737843362544275,0.498641493615980,1.398268710513909,1.045215117420952,1.045215791001011,0.737843252604683];

    disp(max(diag(new_cov)));
    % occasionally, there're small negative values for variance, due to
    % non-positive definiteness of covariance matrix. 
    new_cov = set_min_var(new_cov, 1e-12);

    % Jacobian linearization
    if isequal(type, 'JL')
        z            = reshape(new_mean, [], 1);
        w            = 1 - tanh(z).^2; 
    
        new_mean     = tanh(new_mean);
        new_cov      = w .* new_cov .* w' ;
    
        if ~isempty(new_xcov)
            % note that in our example resnets, there's no nonlinearity in the 
            % "right branch" of the networks, so we didn't implement right
            % product of xcovariance propagation through nonlinearities.
            new_xcov = w .* new_xcov;
        else
            % new_xcov = [];
        end

    elseif isequal(type,'wang')
        % derived by Wang 2016, inspired by MacKay 1992
        z            = reshape(new_mean, [], 1);
        s2           = max(diag(new_cov), 0);
        mu           = 2*sigmoid(z ./ sqrt(0.25 + pi / 8 *s2)) -1 ;
        new_mean     = reshape(mu, size(new_mean)); 
        
        alpha        = 8 - 4*sqrt(2);
        beta         = -0.5 * log(sqrt(2) +1);
        new_var      = 4*sigmoid(alpha * (z+beta) ./ sqrt(1 + pi / 8 * alpha^2 * s2)) - (mu+1).^2;
        new_cov      = diag(new_var); % only propagate variance

        if ~isempty(new_xcov)
            new_xcov = zeros(size(new_xcov)); % only propagate variance
        else
            % new_xcov = [];
        end

    else
        % only upto the third order Taylor expansion are implemented
        z            = double(reshape(new_mean, [], 1));    
        input_var    = double(full(diag(new_cov)));
        s2           = input_var + 1./ (2*gamma.^2);
        s            = sqrt(s2); 
        
        % factors of Taylor polynomial  
        pdf          = 1 ./s .* normpdf(z ./ s);
        cdf          = 1/2 * erfc(-z /sqrt(2) ./ s);
        if any(isnan(pdf))
            disp('pdf has nan');
        end
        if any(isnan(cdf))
            disp('cdf has nan');
        end
        
        A0           = mean(2 * cdf - 1, 2);
        A1           = 2 * mean(pdf, 2);
        if isequal(type, 'ST1')
            new_mean = reshape(A0, size(new_mean));            
            new_cov  = A1 .* new_cov .* A1';
      
            if ~isempty(new_xcov)
                % note that in our example resnets, there's no nonlinearity in the 
                % "right branch" of the networks, so we didn't implement right
                % product of xcovariance propagation through nonlinearities.
                new_xcov = A1 .* new_xcov;
            else
                % new_xcov = [];
            end

        elseif isequal(type, 'ST2')
            A2           = mean(-z ./ s2 .* pdf, 2);
            new_mean     = reshape(A0, size(new_mean)); 
            new_cov      = A1 .* new_cov .* A1'+ ...
                           A2 .* (2 * new_cov.^2) .* A2';
                       
            if ~isempty(new_xcov)
                new_xcov = A1 .* new_xcov;
            else
                % new_xcov = [];
            end

        elseif isequal(type, 'ST3')

            A2           = mean(-z ./ s2 .* pdf, 2);
            A3           = 2/factorial(3) * mean((z.^2 - s2) ./ s2.^2 .* pdf , 2); 
            
            new_mean     = reshape(A0, size(new_mean)); 
    
            new_cov      = A1 .* new_cov .* A1' + ...
                           A2 .* (2 * new_cov.^2) .* A2'+ ...
                           A3 .* (6 * new_cov.^3 + 9 * input_var .* new_cov .* input_var') .* A3';
        
        
            if ~isempty(new_xcov)
                % note that in our example resnets, there's no nonlinearity in the 
                % "right branch" of the networks, so we didn't implement right
                % product of xcovariance propagation through nonlinearities.
        
                new_xcov = A1 .* new_xcov +...
                           3 * A3 .* input_var .* new_xcov;
            else
                % new_xcov = [];
            end
        end
    end
    if any(isnan(new_cov))
        disp('new_cov has nan, and that is because');
        if any(isnan(A1))
            disp('A1 has nan');
        end
        if any(isnan(A2))
            disp('A2 has nan');
        end
        if any(isnan(A3))
            disp('A3 has nan');
        end
        
    end
    if any(isnan(new_xcov))
        disp('new_xcov has nan');
    end
    new_cov          = set_min_var(new_cov, 1e-12);
end






function [new_mean, new_cov] = FullyConnect(layer, new_mean, new_cov)

    W                = layer.Weights;
    B                = layer.Bias;
    new_mean         = reshape(new_mean, [], 1);
    new_mean         = W * new_mean + B;
    new_cov          = W * new_cov * W';
    new_cov          = set_min_var(new_cov, 1e-12);
end



function [new_mean, new_cov] = Residual(new_mean, new_cov, new_mean_res, new_cov_res, input_xcov)
    new_mean         = new_mean + new_mean_res;
    new_cov          = new_cov + new_cov_res + input_xcov + input_xcov';
    new_cov          = set_min_var(new_cov, 1e-12);
end


function [padded_cov, padded_xcov] = pad_cov_xcov(covariance, cross_cov, direction, image_dim, padding)

% construct a binary array to find where to pad zeros
% this is to avoid padding at the location where pixel values happen to be
% zero, e.g. got ReLUed
padimg           = padarray_tblr(ones(image_dim, "gpuArray"), padding); 
flatpadI         = reshape(padimg, 1, []);
pointer          = find(flatpadI);

% create a "canvas" of the padded covariance
padded_cov       = zeros(numel(flatpadI), numel(flatpadI), "gpuArray");

pointer_master   = 1:numel(flatpadI);
p                = ismember(pointer_master,pointer); 
M                = bsxfun(@and,p,p');
padded_cov(M~=0) = covariance;
% padded_cov       = sparse(padded_cov);

if ~isempty(cross_cov)
    if isequal(direction, 'left')
        padded_xcov          = zeros(numel(flatpadI), size(cross_cov,2), "gpuArray");
        padded_xcov(p~=0, :) = cross_cov;
        % padded_xcov          = sparse(padded_xcov);
    elseif isequal(direction, 'right') 
        padded_xcov          = zeros(size(cross_cov,1), numel(flatpadI), "gpuArray");
        padded_xcov(:, p~=0) = cross_cov;
        % padded_xcov          = sparse(padded_xcov);
    else
        error("can only be 'left' or 'right");
    end
else
    % no need to keep track of cross-covariance this time
    padded_xcov = []; 
end

end


function padinput = padarray_tblr(input, padding)
% padarray() function that takes 4 d padding size vector
% namely top bottom left and right padding size
padding_pre  = padding([1,3]);
padding_post = padding([2,4]);

padinput     = padarray(input, padding_pre, 'pre');
padinput     = padarray(padinput, padding_post, 'post');
end




function covariance = set_min_var(covariance, small_val)
% remove the negative variance.
% replace the minimal variance by a small value, typically 1e-12
% this is usually caused by covariance matrix being 
% not strictly positive definite
diagonals    = diag(covariance);
covariance   = covariance - diag(diagonals);
diagonals    = max(small_val, diagonals);
covariance   = covariance + diag(diagonals);

end

function y = sigmoid(x)
% have to redefine sigmoid because MATLAB doesn't take gpuArray
y = 1./ (1+exp(-x));
end
