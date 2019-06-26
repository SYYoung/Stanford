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
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
% all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
a_1 = [ones(m, 1) X]';
fprintf("dim of a_1 = %d, %d", size(a_1,1), size(a_1,2));

z_2 = Theta1 * a_1;
fprintf("dim of z_2 = %d, %d", size(z_2,1), size(z_2,2));
a_2 = sigmoid(z_2);
fprintf("dim of a_2 = %d, %d", size(a_2,1), size(a_2,2));
a_2 = [ones(size(a_2,2)); a_2];
fprintf("dim of new a_2 = %d, %d", size(a_2,1), size(a_2,2));

z_3 = Theta2 * a_2;
fprintf("dim of z_3 = %d, %d", size(z_3,1), size(z_3,2));
a_3 = sigmoid(z_3);
fprintf("dim of a_3 = %d, %d", size(a_3,1), size(a_3,2));

p1 = max(a_3, [], 1);
p = p1';
fprintf("dim of p = %d, %d", size(p,1), size(p,2));
fprint("first 5 of p = %d, %d, %d, %d, %d", p(1:5,1));
% =========================================================================


end
