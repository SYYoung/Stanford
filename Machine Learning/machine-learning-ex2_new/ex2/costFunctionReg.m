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

[cost_noreg, grad_noreg] = costFunction(theta, X, y);
%J = cost_noreg + (sum(theta .^ 2) - theta(1)^2) * lambda/(2*m);
J = cost_noreg + sum(theta(2:end) .^ 2) * lambda/(2*m);

grad_0 = grad_noreg(1);
grad = grad_noreg + theta * lambda/m;
grad(1) = grad_0;


% =============================================================

end
