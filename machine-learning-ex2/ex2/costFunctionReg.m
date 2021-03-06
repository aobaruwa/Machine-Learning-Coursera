function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%compute the sigmoid parameter
z = X * theta;
%calculate h(theta) using the logistic function
predictions = sigmoid(z);
%cost function without the regularization term
costJ = (1/m) * sum(-y.* log(predictions) - (1- y).*log(1 - predictions));
%regularization term
costReg = lambda/ (2*m) * sum(theta(2: end).^2);
J = costJ + costReg;
%calculate the first gradient
grad(1) = (1/ m) * sum ((predictions - y).* X(:, 1));
%calculate the remainder of the gradient vector
n = size(X, 2);
for i = 2: n
  grad(i) = (1/m) * sum((predictions - y).* X(:, i)) + (lambda / m) * theta(i);



end



% =============================================================

end
