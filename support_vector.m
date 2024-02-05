X = [randn(50,2)+1; randn(50,2)-1];
Y = [ones(50,1); -ones(50,1)];

kernels = {'linear', 'polynomial', 'rbf'}; % rbf is Gaussian


%looping through different kernel dunctions 
for k = 1:length(kernels)
    % Train SVM model 
    SVMModel = fitcsvm(X, Y, 'KernelFunction', kernels{k});
    
    % Support vectors
    sv = SVMModel.SupportVectors;

    % Create a grid
    [X1, X2] = meshgrid(min(X(:,1)):.02:max(X(:,1)), min(X(:,2)):.02:max(X(:,2)));
    [~, score] = predict(SVMModel, [X1(:), X2(:)]);
    scoreGrid = reshape(score(:,1), size(X1,1), size(X2,2));
    
    % Plot the data decision boundary and support vectors 
    figure;
    hold on;
    contour(X1, X2, scoreGrid, [0, 0], 'k');
    gscatter(X(:,1), X(:,2), Y, 'rb', 'o',[], 'off');
    
    plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 10);
    title(sprintf('Decision Boundary with %s Kernel', kernels{k}));
    xlabel('X_1');
    ylabel('X_2');
    legend({'Decision Boundary', 'Class 1', 'Class -1', 'Support Vectors'}, 'Location', 'Best');
    hold off;
end