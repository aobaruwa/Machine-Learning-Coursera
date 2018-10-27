function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
raw_x = X(:, 2:size(X, 2) - 1)
n = size(raw_x, 2)
J_history = zeros(num_iters, 1);
mu    = mean(X);
sigma = std(X);
for i = 1: n
  d = raw_x(:, i) - mu(i);
  X_norm(:, i) = d / sigma(i);
  fprintf('dim of X norm is %f', size(X_norm))
end

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
  m = size(X, 1);

  predictions = X_norm* theta;
  
  error = predictions - y;
  sqrError = (predictions - y).^2;
  temp = theta - (alpha/m) * sum(sqrError);
  theta = temp;
    
     
     







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end


