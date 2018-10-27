function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%include the bias units in the X matrix
X = [ones(m,1), X];
%the first vector of activation units
A = sigmoid(Theta1 * X');
%add an additional row of ones(the bias units) to the matrix Z1  
refA = [ones(1, size(A, 2));A];
%the second vector of the activation units
B = sigmoid(Theta2 * refA);
%extract the indices of the element of the highest probability
[B, I] = max(B); 

%display a column vector predicting the label for each digit
p = I'; 







% =========================================================================


end
