clear all;
close all;
data = zeros(3, 3, 4, 15);
savedir = './';
nonlin_counter = 1;
for nonlin  = ["relu", "tanh", "gelu"]
    nonlin = char(nonlin);
    figure('Position', [0 0 1000 1000]);
    t = tiledlayout(3,3);
    
    colormap    = ['4A5E65'; 'E29957'; 'B95A58'; '86B5A1'; 'C19A6B'];
    colormap    = hex2rgb(colormap);
    depth = [3 2 1; 3 10 3; 4 25 3];
    for modeldepth = 1:3
        stacks  = depth(modeldepth, :);
        netname = ['resnet' sprintf('%d', stacks)];
        
        % Initialize arrays to store data for plotting
        in_var_values = [1, 10, 100, 1000];
        
        % Initialize arrays for metrics and their standard errors
        W2_vals = zeros(5, length(in_var_values));  % 5 models, 4 variance values
        L2_vals = zeros(5, length(in_var_values));
        Fro_vals = zeros(5, length(in_var_values));
        
        W2_std_errs = zeros(5, length(in_var_values));
        L2_std_errs = zeros(5, length(in_var_values));
        Fro_std_errs = zeros(5, length(in_var_values));
        
        for var_idx = 1:length(in_var_values)
            in_var = in_var_values(var_idx);
            
            % Load data
            load([savedir 'true1e6/' netname '_' nonlin '_result_var' num2str(in_var) '.mat']);
            load([savedir 'predoutput/' netname '_' nonlin '_preddist_var' num2str(in_var) '.mat']);
            
            % Get number of repeat samples
            num_samples = size(true_mean, 2);  % Assuming true_mean is 10×N
            
            % Initialize arrays for storing metrics for each sample
            W2_samples = zeros(5, num_samples);  % 5 models, N samples
            L2_samples = zeros(5, num_samples);
            Fro_samples = zeros(5, num_samples);
            
            % Compute metrics for each sample
            for sample = 1:num_samples
                % Get the current sample's true mean and covariance
                curr_true_mean = true_mean(:, sample);
                curr_true_cov = true_cov(:, :, sample);
                
                % Compute Wasserstein distances for each model
                W2_samples(1, sample) = gaussianWassersteinDistance(curr_true_mean, curr_true_cov, modelJL_mean, modelJL_cov);
                W2_samples(2, sample) = gaussianWassersteinDistance(curr_true_mean, curr_true_cov, modelO1_mean, modelO1_cov);
                W2_samples(3, sample) = gaussianWassersteinDistance(curr_true_mean, curr_true_cov, modelO2_mean, modelO2_cov);
                W2_samples(4, sample) = gaussianWassersteinDistance(curr_true_mean, curr_true_cov, modelO3_mean, modelO3_cov);
                
                % Add Wang or Frey model if applicable
                if isequal(nonlin, 'tanh')
                    W2_samples(5, sample) = gaussianWassersteinDistance(curr_true_mean, curr_true_cov, modelwang_mean, modelwang_cov);
                elseif isequal(nonlin, 'relu')
                    W2_samples(5, sample) = gaussianWassersteinDistance(curr_true_mean, curr_true_cov, modelfrey_mean, modelfrey_cov);
                else
                    W2_samples(5, sample) = NaN;  % No 5th model for GELU
                end
                
                % Compute L2 norms
                L2_samples(1, sample) = norm(curr_true_mean - modelJL_mean, 2);
                L2_samples(2, sample) = norm(curr_true_mean - modelO1_mean, 2);
                L2_samples(3, sample) = norm(curr_true_mean - modelO2_mean, 2);
                L2_samples(4, sample) = norm(curr_true_mean - modelO3_mean, 2);
                
                if isequal(nonlin, 'tanh')
                    L2_samples(5, sample) = norm(curr_true_mean - modelwang_mean, 2);
                elseif isequal(nonlin, 'relu')
                    L2_samples(5, sample) = norm(curr_true_mean - modelfrey_mean, 2);
                else
                    L2_samples(5, sample) = NaN;
                end
                
                % Compute Frobenius norms
                Fro_samples(1, sample) = norm(curr_true_cov - modelJL_cov, 'fro');
                Fro_samples(2, sample) = norm(curr_true_cov - modelO1_cov, 'fro');
                Fro_samples(3, sample) = norm(curr_true_cov - modelO2_cov, 'fro');
                Fro_samples(4, sample) = norm(curr_true_cov - modelO3_cov, 'fro');
                
                if isequal(nonlin, 'tanh')
                    Fro_samples(5, sample) = norm(curr_true_cov - modelwang_cov, 'fro');
                elseif isequal(nonlin, 'relu')
                    Fro_samples(5, sample) = norm(curr_true_cov - modelfrey_cov, 'fro');
                else
                    Fro_samples(5, sample) = NaN;
                end
            end
            
            % Calculate mean and standard error for each metric
            W2_vals(:, var_idx) = mean(W2_samples, 2);
            L2_vals(:, var_idx) = mean(L2_samples, 2);
            Fro_vals(:, var_idx) = mean(Fro_samples, 2);
            
            % Calculate standard error (standard deviation / sqrt(n))
            W2_std_errs(:, var_idx) = std(W2_samples, 0, 2) / sqrt(num_samples);
            L2_std_errs(:, var_idx) = std(L2_samples, 0, 2) / sqrt(num_samples);
            Fro_std_errs(:, var_idx) = std(Fro_samples, 0, 2) / sqrt(num_samples);
        end
        
        % Extract values for each model for easier plotting
        W2_JL_vals = W2_vals(1, :);
        W2_O1_vals = W2_vals(2, :);
        W2_O2_vals = W2_vals(3, :);
        W2_O3_vals = W2_vals(4, :);
        
        L2_JL_vals = L2_vals(1, :);
        L2_O1_vals = L2_vals(2, :);
        L2_O2_vals = L2_vals(3, :);
        L2_O3_vals = L2_vals(4, :);
        
        Fro_JL_vals = Fro_vals(1, :);
        Fro_O1_vals = Fro_vals(2, :);
        Fro_O2_vals = Fro_vals(3, :);
        Fro_O3_vals = Fro_vals(4, :);
        
        % Extract standard errors
        W2_JL_errs = W2_std_errs(1, :);
        W2_O1_errs = W2_std_errs(2, :);
        W2_O2_errs = W2_std_errs(3, :);
        W2_O3_errs = W2_std_errs(4, :);
        
        L2_JL_errs = L2_std_errs(1, :);
        L2_O1_errs = L2_std_errs(2, :);
        L2_O2_errs = L2_std_errs(3, :);
        L2_O3_errs = L2_std_errs(4, :);
        
        Fro_JL_errs = Fro_std_errs(1, :);
        Fro_O1_errs = Fro_std_errs(2, :);
        Fro_O2_errs = Fro_std_errs(3, :);
        Fro_O3_errs = Fro_std_errs(4, :);
        
        % For special models (Wang or Frey)
        if isequal(nonlin, 'tanh')
            W2_wang_vals = W2_vals(5, :);
            L2_wang_vals = L2_vals(5, :);
            Fro_wang_vals = Fro_vals(5, :);
            
            W2_wang_errs = W2_std_errs(5, :);
            L2_wang_errs = L2_std_errs(5, :);
            Fro_wang_errs = Fro_std_errs(5, :);
        elseif isequal(nonlin, 'relu')
            W2_frey_vals = W2_vals(5, :);
            L2_frey_vals = L2_vals(5, :);
            Fro_frey_vals = Fro_vals(5, :);
            
            W2_frey_errs = W2_std_errs(5, :);
            L2_frey_errs = L2_std_errs(5, :);
            Fro_frey_errs = Fro_std_errs(5, :);
        end
        
        % Store data for later use if needed
        if isequal(nonlin, 'tanh')
            data(modeldepth, nonlin_counter, :,:) = vertcat(...
                W2_O3_vals, L2_O3_vals, Fro_O3_vals, ...
                W2_O2_vals, L2_O2_vals, Fro_O2_vals, ...
                W2_O1_vals, L2_O1_vals, Fro_O1_vals, ...
                W2_JL_vals, L2_JL_vals, Fro_JL_vals, ...
                W2_wang_vals, L2_wang_vals, Fro_wang_vals)';
        elseif isequal(nonlin, 'relu')
            data(modeldepth, nonlin_counter, :,:) = vertcat(...
                W2_O3_vals, L2_O3_vals, Fro_O3_vals, ...
                W2_O2_vals, L2_O2_vals, Fro_O2_vals, ...
                W2_O1_vals, L2_O1_vals, Fro_O1_vals, ...
                W2_JL_vals, L2_JL_vals, Fro_JL_vals, ...
                W2_frey_vals, L2_frey_vals, Fro_frey_vals)';
        else
            data(modeldepth, nonlin_counter, :,1:12) = vertcat(...
                W2_O3_vals, L2_O3_vals, Fro_O3_vals, ...
                W2_O2_vals, L2_O2_vals, Fro_O2_vals, ...
                W2_O1_vals, L2_O1_vals, Fro_O1_vals, ...
                W2_JL_vals, L2_JL_vals, Fro_JL_vals)';
        end
        
        % PLOT 1: Plot Wasserstein distances with error bars
        nexttile;
        axis square;
        % Plot with error bars
        errorbar(in_var_values, W2_JL_vals, W2_JL_errs, '-d', ...
            'LineWidth', 2, 'Color', colormap(1,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(1,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        hold on;

        if isequal(nonlin, 'tanh')
            errorbar(in_var_values, W2_wang_vals, W2_wang_errs, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        elseif isequal(nonlin, 'relu')
            errorbar(in_var_values, W2_frey_vals, W2_frey_errs, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        end
        
        errorbar(in_var_values, W2_O1_vals, W2_O1_errs, '-o', ...
            'LineWidth', 2, 'Color', colormap(2,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(2,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        errorbar(in_var_values, W2_O2_vals, W2_O2_errs, '-^', ...
            'LineWidth', 2, 'Color', colormap(3,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(3,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        errorbar(in_var_values, W2_O3_vals, W2_O3_errs, '-s', ...
            'LineWidth', 2, 'Color', colormap(4,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(4,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        
        set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize', 15);

        if modeldepth == 1
            % if isequal(nonlin, 'tanh')
            %     legend('JL', 'Wang et al. (2016)', 'PTPE 1st', 'PTPE 2nd', 'PTPE 3rd', 'Location', 'northwest', 'Box', 'off');
            if isequal(nonlin, 'relu')
                legend('JL', 'Frey et al. (1999)', 'PTPE 1st', 'PTPE 2nd', 'PTPE 3rd', 'Location', 'northwest', 'Box', 'off');
            elseif isequal(nonlin, 'gelu')
                legend('JL', 'PTPE 1st', 'PTPE 2nd', 'PTPE 3rd', 'Location', 'northwest', 'Box', 'off');
            end
        elseif modeldepth == 3
            xlabel("Input Variance");
        end
        ylabel("Wasserstein2 distance");
        xticks([1e0, 1e1, 1e2, 1e3]);
        
        
        % PLOT 2: Plot L2 norms with error bars
        nexttile;
        axis square;
        errorbar(in_var_values, L2_JL_vals, L2_JL_errs, '-d', ...
            'LineWidth', 2, 'Color', colormap(1,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(1,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        hold on;
        
        if isequal(nonlin, 'tanh')
            errorbar(in_var_values, L2_wang_vals, L2_wang_errs, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        elseif isequal(nonlin, 'relu')
            errorbar(in_var_values, L2_frey_vals, L2_frey_errs, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        end
        
        errorbar(in_var_values, L2_O1_vals, L2_O1_errs, '-o', ...
            'LineWidth', 2, 'Color', colormap(2,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(2,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        errorbar(in_var_values, L2_O2_vals, L2_O2_errs, '-^', ...
            'LineWidth', 2, 'Color', colormap(3,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(3,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        errorbar(in_var_values, L2_O3_vals, L2_O3_errs, '-s', ...
            'LineWidth', 2, 'Color', colormap(4,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(4,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        
        set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize', 15);
        if numel(gca().YTick) < 3
            gca().YTick= logspace(log10(gca().YLim(1)), log10(gca().YLim(2)), 3);
        end
        if modeldepth == 1
            title(['ResNet13 - ' nonlin]);
        elseif modeldepth == 2
            title(['ResNet33 - ' nonlin]);
        elseif modeldepth == 3
            xlabel("Input Variance");
            title(['ResNet65 - ' nonlin]);
        end
        ylabel("L2 norm");
        xticks([1e0, 1e1, 1e2, 1e3]);
        
        % PLOT 3: Plot Frobenius norms with error bars
        nexttile;
        axis square;
        errorbar(in_var_values, Fro_JL_vals, Fro_JL_errs, '-d', ...
            'LineWidth', 2, 'Color', colormap(1,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(1,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        hold on;
        
        if isequal(nonlin, 'tanh')
            errorbar(in_var_values, Fro_wang_vals, Fro_wang_errs, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        elseif isequal(nonlin, 'relu')
            errorbar(in_var_values, Fro_frey_vals, Fro_frey_errs, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        end
        
        errorbar(in_var_values, Fro_O1_vals, Fro_O1_errs, '-o', ...
            'LineWidth', 2, 'Color', colormap(2,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(2,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        errorbar(in_var_values, Fro_O2_vals, Fro_O2_errs, '-^', ...
            'LineWidth', 2, 'Color', colormap(3,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(3,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        errorbar(in_var_values, Fro_O3_vals, Fro_O3_errs, '-s', ...
            'LineWidth', 2, 'Color', colormap(4,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(4,:), 'MarkerEdgeColor', 'None', 'CapSize', 10);
        
        if modeldepth == 1
            if isequal(nonlin, 'tanh')
                legend('JL', 'Wang et al. (2016)', 'PTPE 1st', 'PTPE 2nd', 'PTPE 3rd', 'Location', 'northwest', 'Box', 'off');
            end
        end
        set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize', 15);

        if modeldepth == 3
            xlabel("Input Variance");
        end
        ylabel("Frobenius norm");
        xticks([1e0, 1e1, 1e2, 1e3]);
    end
    
    % Get all axes handles
    axesHandles = findall(t, 'Type', 'Axes');
    
    % Synchronize limits for each column
    numColumns = t.GridSize(2); % Number of columns in the layout
    
    for col = 1:numColumns
        % Find axes in the same column
        colAxes = axesHandles(mod(0:numel(axesHandles)-1, numColumns) == col-1);
    
        % Link axes for the column
        linkaxes(colAxes, 'xy'); % Link both x and y axes
    end
    nonlin_counter = nonlin_counter+1;
end

%% Keep the original functions
function w2_distance = gaussianWassersteinDistance(ref_mean, ref_cov, pred_mean, pred_cov)
    % Ensure that all inputs are double precision
    ref_mean = double(ref_mean);
    ref_cov  = double(ref_cov);
    pred_mean = double(pred_mean);
    pred_cov  = double(pred_cov);

    % 1) Compute the squared norm of the mean difference
    mean_diff = ref_mean - pred_mean;
    mean_term = mean_diff' * mean_diff;  % ||\u03bc1 - \u03bc2||^2

    % 2) Compute (S2^(1/2) * S1 * S2^(1/2))^(1/2)
    pred_cov_sqrt = sqrtm(pred_cov);
    middle = pred_cov_sqrt * ref_cov * pred_cov_sqrt;
    middle_sqrt = sqrtm(middle);

    % 3) Trace part: Tr(S1 + S2 - 2 * middle_sqrt)
    trace_term = trace(ref_cov + pred_cov - 2 * middle_sqrt);

    % 4) Combine for the squared 2-Wasserstein distance
    w2_squared = mean_term + trace_term;

    % 5) Wasserstein distance = sqrt(W2^2)
    w2_distance = sqrt(max(w2_squared, 0));
    
    % return only the real parts, because the pred_cov may be not positive
    % definite
    w2_distance = real(w2_distance);
end

function rgb = hex2rgb(hex)
    rgb = sscanf(hex', '%2x%2x%2x', [3, size(hex, 1)])' / 255;
end