function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


cChoice = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmaChoice = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

params = zeros(size(cChoice,1).^ 2, 2);

%generate all 64 combinations of the pair of parameters
count = 1;
for c = 1: size(cChoice, 1)
    for s = 1: size(sigmaChoice, 1)
        params(count, 1) = cChoice(c);
        params(count, 2) = sigmaChoice(s);
        count = count+ 1;
    end
end
params
bestError = 1000
%computing the prediction errors to identify the optimal pair of parameters
error = zeros(size(cChoice, 1) ^2, 1);
for row = 1: 64;
    x1 = [1 2 1]; x2 = [0 4 -1]; 
    model = svmTrain(X,y, params(row, 1), @(x1, x2) gaussianKernel(x1, x2, params(row, 2)));

    predictions = svmPredict(model, Xval);
    error(row) = mean(double(predictions ~= yval));
    if error(row) < bestError && error(row) > 0
        bestError = error(row)
        C = params(row, 1)
        sigma = params(row, 2)
    end
end


%fprintf('minimum error is: '), min(error)
% =========================================================================

end