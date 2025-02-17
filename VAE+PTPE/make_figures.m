close all
addpath("/Users/songhanz/Documents/WashU/SideProject/TMLR")
colormap    = ["4A5E65"; "E29957"; "B95A58"; "86B5A1"];
figure();

for dim = [2 5 10 20]

    T = readtable(['elbo_data_' num2str(dim) 'D.csv']);
    
    idx_train_vanilla = strcmp(T.Label, "train (vanilla)");
    idx_test_vanilla = strcmp(T.Label, "test (vanilla)");
    train_vanilla = T.Value(idx_train_vanilla)';
    test_vanilla = T.Value(idx_test_vanilla)';
    
    T = readtable(['elbo_data_' num2str(dim) 'D_EP.csv']);

    idx_train_PTPE = strcmp(T.Label, "train (EP)");
    idx_test_PTPE = strcmp(T.Label, "test (EP)");
    train_PTPE = T.Value(idx_train_PTPE)';
    test_PTPE = T.Value(idx_test_PTPE)';

    if dim == 2
        subplot(2,3,1);
        title("2D latent space");
    elseif dim == 5
        subplot(2,3,2);
        title("5D latent space");
    elseif dim == 10
        subplot(2,3,4);
        title("10D latent space");
    elseif dim == 20
        subplot(2,3,5);
        title("20D latent space");
    end
    plot([1:200], train_vanilla, ":", "Color", hex2rgb(colormap(1)), "LineWidth", 3.5);
    hold on;
    plot([1:200], test_vanilla, ":", "Color", hex2rgb(colormap(2)), "LineWidth", 3.5);
    plot([1:200], train_PTPE, "-", "Color", hex2rgb(colormap(1)), "LineWidth", 3);
    plot([1:200], test_PTPE, "-", "Color", hex2rgb(colormap(2)), "LineWidth", 3);
    xlabel("epoch")
    ylabel("ELBO")
    set(gca, 'FontSize', 20); 
    ylim([-180, -70]);
    if dim == 2
        subplot(2,3,1);
        legend("train (vanilla)", "test (vanilla)", "train (PTPE)", "test (PTPE)", 'Location', "northeast");
    end

    if dim == 2
        subplot(2,3,1);
        title("2D latent space");
    elseif dim == 5
        subplot(2,3,2);
        title("5D latent space");
    elseif dim == 10
        subplot(2,3,4);
        title("10D latent space");
    elseif dim == 20
        subplot(2,3,5);
        title("20D latent space");
    end

    subplot(2,3,6);
    color = hex2rgb(colormap(3));
    plot([2 5 10 20], [30.734825897216798, 19.466510229110717, 11.633707365989686, 8.803342308998108], "o-", 'Color', color, ...
        'MarkerSize', 10, 'MarkerFaceColor', color, "MarkerEdgeColor", color, 'LineWidth', 3);
    hold on;
    color = hex2rgb(colormap(4));
    plot([2 5 10 20], [26.66540386199951, 16.140279521942137, 10.239319205284119, 6.921008944511414], "o-", 'Color', color, ...
        'MarkerSize', 10, 'MarkerFaceColor', color, "MarkerEdgeColor", color, 'LineWidth', 3);
    legend("Vanilla", "PTPE");
    ylabel("Reconstruction Loss");
    xlabel("Latent Space");
    set(gca, 'FontSize', 20);

end