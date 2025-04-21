clear all;
% close all;
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
        W2_JL_vals = [];
        W2_O1_vals = [];
        W2_O2_vals = [];
        W2_O3_vals = [];
        L2_JL_vals = [];
        L2_O1_vals = [];
        L2_O2_vals = [];
        L2_O3_vals = [];
        Fro_JL_vals = [];
        Fro_O1_vals = [];
        Fro_O2_vals = [];
        Fro_O3_vals = [];
        if isequal(nonlin, 'tanh')
            W2_wang_vals = [];
            L2_wang_vals = [];
            Fro_wang_vals= [];
        elseif isequal(nonlin, 'relu')
            W2_frey_vals = [];
            L2_frey_vals = [];
            Fro_frey_vals= [];
        end
        
        nexttile;
        for in_var = in_var_values
            % Load data
            load([savedir 'true1e7/' netname '_' nonlin '_result_var' num2str(in_var) '.mat']);
            load([savedir 'predoutput/' netname '_' nonlin '_preddist_var' num2str(in_var) '.mat']);
            
    
            % Compute Wasserstein distances
            W2_O3 = gaussianWassersteinDistance(true_mean, true_cov, modelO3_mean, modelO3_cov);
            W2_O2 = gaussianWassersteinDistance(true_mean, true_cov, modelO2_mean, modelO2_cov);
            W2_O1 = gaussianWassersteinDistance(true_mean, true_cov, modelO1_mean, modelO1_cov);
            W2_JL = gaussianWassersteinDistance(true_mean, true_cov, modelJL_mean, modelJL_cov);

            if isequal(nonlin, 'tanh')
                W2_wang = gaussianWassersteinDistance(true_mean, true_cov, modelwang_mean, modelwang_cov);
            elseif isequal(nonlin, 'relu')
                W2_frey = gaussianWassersteinDistance(true_mean, true_cov, modelfrey_mean, modelfrey_cov);
            end

            % Compute L2 norms
            L2_O3 = norm(true_mean - modelO3_mean, 2);
            L2_O2 = norm(true_mean - modelO2_mean, 2);
            L2_O1 = norm(true_mean - modelO1_mean, 2);
            L2_JL = norm(true_mean - modelJL_mean, 2);
            if isequal(nonlin, 'tanh')
                L2_wang = norm(true_mean - modelwang_mean, 2);
            elseif isequal(nonlin, 'relu')
                L2_frey = norm(true_mean - modelfrey_mean, 2);
            end
    
            % Compute Frobenius norms
            Fro_O3 = norm(true_cov - modelO3_cov, 'fro');
            Fro_O2 = norm(true_cov - modelO2_cov, 'fro');
            Fro_O1 = norm(true_cov - modelO1_cov, 'fro');
            Fro_JL = norm(true_cov - modelJL_cov, 'fro');
            if isequal(nonlin, 'tanh')
                Fro_wang = norm(true_cov - modelwang_cov, 'fro');
            elseif isequal(nonlin, 'relu')
                Fro_frey = norm(true_cov - modelfrey_cov, 'fro');
            end

    
            % Store values for plotting later
            W2_JL_vals = [W2_JL_vals, W2_JL];
            W2_O1_vals = [W2_O1_vals, W2_O1];
            W2_O2_vals = [W2_O2_vals, W2_O2];
            W2_O3_vals = [W2_O3_vals, W2_O3];
            
    
            L2_JL_vals = [L2_JL_vals, L2_JL];
            L2_O1_vals = [L2_O1_vals, L2_O1];
            L2_O2_vals = [L2_O2_vals, L2_O2];
            L2_O3_vals = [L2_O3_vals, L2_O3];
            
    
            Fro_JL_vals = [Fro_JL_vals, Fro_JL];
            Fro_O1_vals = [Fro_O1_vals, Fro_O1];
            Fro_O2_vals = [Fro_O2_vals, Fro_O2];
            Fro_O3_vals = [Fro_O3_vals, Fro_O3];

            
    
            if isequal(nonlin, 'tanh')
                W2_wang_vals  = [W2_wang_vals, W2_wang];
                L2_wang_vals  = [L2_wang_vals, L2_wang];
                Fro_wang_vals = [Fro_wang_vals, Fro_wang];
            elseif isequal(nonlin, 'relu')
                W2_frey_vals  = [W2_frey_vals, W2_frey];
                L2_frey_vals  = [L2_frey_vals, L2_frey];
                Fro_frey_vals = [Fro_frey_vals, Fro_frey];
            end
            
        end

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
    
        % Plot Wasserstein distances
        plot(in_var_values, W2_JL_vals, '-d', ...
            'LineWidth', 2, 'Color', colormap(1,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(1,:), 'MarkerEdgeColor', 'None');
        hold on;
        if isequal(nonlin, 'tanh')
            plot(in_var_values, W2_wang_vals, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None');
        elseif isequal(nonlin, 'relu')
            plot(in_var_values, W2_frey_vals, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None');
        end
        plot(in_var_values, W2_O1_vals, '-o', ...
            'LineWidth', 2, 'Color', colormap(2,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(2,:), 'MarkerEdgeColor', 'None');
        plot(in_var_values, W2_O2_vals, '-^', ...
            'LineWidth', 2, 'Color', colormap(3,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(3,:), 'MarkerEdgeColor', 'None');
        plot(in_var_values, W2_O3_vals, '-s', ...
            'LineWidth', 2, 'Color', colormap(4,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(4,:), 'MarkerEdgeColor', 'None');
        
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
        axis square;
    
        % Plot L2 norms
        nexttile;
        plot(in_var_values, L2_JL_vals, '-d', ...
            'LineWidth', 2, 'Color', colormap(1,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(1,:), 'MarkerEdgeColor', 'None');
        hold on;
        if isequal(nonlin, 'tanh')
            plot(in_var_values, L2_wang_vals, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None');
        elseif isequal(nonlin, 'relu')
            plot(in_var_values, L2_frey_vals, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None');
        end
        plot(in_var_values, L2_O1_vals, '-o', ...
            'LineWidth', 2, 'Color', colormap(2,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(2,:), 'MarkerEdgeColor', 'None');
        plot(in_var_values, L2_O2_vals, '-^', ...
            'LineWidth', 2, 'Color', colormap(3,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(3,:), 'MarkerEdgeColor', 'None');
        plot(in_var_values, L2_O3_vals, '-s', ...
            'LineWidth', 2, 'Color', colormap(4,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(4,:), 'MarkerEdgeColor', 'None');
        
        set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize', 15);
        if modeldepth == 1
            title(['ResNet13 - ' nonlin]);
        elseif modeldepth == 2
            title(['ResNet33 - ' nonlin]);
        elseif modeldepth == 3
            xlabel("Input Variance");
            title(['ResNet65 - ' nonlin]);
        end
        ylabel("L2 norm");
        axis square;
    
        % Plot Frobenius norms
        nexttile;
        plot(in_var_values, Fro_JL_vals, '-d', ...
            'LineWidth', 2, 'Color', colormap(1,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(1,:), 'MarkerEdgeColor', 'None');
        hold on;
        if isequal(nonlin, 'tanh')
            plot(in_var_values, Fro_wang_vals, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None');
        elseif isequal(nonlin, 'relu')
            plot(in_var_values, Fro_frey_vals, '-pentagram', ...
                'LineWidth', 2, 'Color', colormap(5,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(5,:), 'MarkerEdgeColor', 'None');
        end
        plot(in_var_values, Fro_O1_vals, '-o', ...
            'LineWidth', 2, 'Color', colormap(2,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(2,:), 'MarkerEdgeColor', 'None');
        plot(in_var_values, Fro_O2_vals, '-^', ...
            'LineWidth', 2, 'Color', colormap(3,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(3,:), 'MarkerEdgeColor', 'None');
        plot(in_var_values, Fro_O3_vals, '-s', ...
            'LineWidth', 2, 'Color', colormap(4,:), 'MarkerSize', 15, 'MarkerFaceColor', colormap(4,:), 'MarkerEdgeColor', 'None');
        
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
        axis square;
        
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

%% 
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

