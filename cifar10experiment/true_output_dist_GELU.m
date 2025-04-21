reset(gpuDevice);
clear all;
close all

%%
datadir = tempdir; 
downloadCIFARData(datadir);
[XTrain,TTrain,XValidation,TValidation] = loadCIFARData(datadir);
pic = XValidation(:,:,:,2); % take the second image as input, because the first is hard to tell what it is
sz = size(pic);

nonlin  = 'gelu';
savedir = '/scratch1/fs1/shinung/songhan.z/TMLR/';

for netname = ["resnet321", "resnet3103", "resnet4253"]
    netname = char(netname);
    load([savedir 'models/' netname '_' nonlin '.mat'], 'net');

    %% input variance
    %for var = [1 10]
    for var = [1 10 100 1000]    
        
    
        % Define the number of noise samples
        numSamples  = 1e6;
        batch_num   = 2;
        batchSize   = numSamples / batch_num; % Split into 10 batches
        trials      = 1;
        
        % Preallocate space for activations
        act_all = zeros(10, numSamples, 'double'); % 10 is activation (category) size 
        
        true_cov = zeros(10, 10, trials);
        true_mean = zeros(10, trials);
        % Loop through 10 trials
        for trial = 1:trials
            % Loop through 10 batches
            for batchIdx = 1:batch_num
                disp(batchIdx);
                I = gpuArray(single(pic));
    
                % Generate Gaussian noise directly on GPU for the current batch
                noise = gpuArray.randn(sz(1), sz(2), sz(3), batchSize, 'single');
            
                % Add noise to the input image
                noisyImages = I + noise * sqrt(var);
            
                % Get activations for the noisy images
                act_batch = activations(net, noisyImages, 'fc', "ExecutionEnvironment", "gpu");
                act_batch = squeeze(act_batch);
            
                % Store the batch of activations
                act_all(:, (batchIdx-1)*batchSize+1:batchIdx*batchSize) = double(gather(act_batch));
                reset(gpuDevice);
            end
        
            % Concatenate all activations and calculate mean and covariance
            true_cov(:,:,trial) = cov(act_all.');
            true_mean(:,trial) = mean(act_all, 2);
        end
    
        % Save the results
        save([savedir 'true1e6/' netname '_' nonlin '_result_var' num2str(var) '.mat'], 'true_mean', 'true_cov');
    
        
    end
end




