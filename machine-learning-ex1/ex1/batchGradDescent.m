
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: Cost and Gradient descent ===================

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 3000;
alpha = 0.01;


fprintf('\nRunning Gradient Descent ...\n')

for count = 500: 500: iterations
    % run gradient descent
    theta = gradientDescent(X, y, theta, alpha, count);
    fprintf('Linear Regression after %d iteration \n', count)
    % Plot the linear fit
    hold on; % keep previous plot visible
    plot(X(:,2), X*theta, '-')
    legend('Training data', 'Linear regression')
    hold off % don't overlay any more plots on this figure
    fprintf('Program paused. Press enter to continue.\n');
    pause;

end
% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);



