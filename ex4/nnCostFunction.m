function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%fprintf("\n theta1 :%d",size(Theta1));
%fprintf("\n theta2:%d\n",size(Theta2));

% Setup some useful variables
m = size(X, 1);
X=[ones(m,1) X];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%


% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
a1=sigmoid(Theta1*X');
a1=a1';
a1=[ones(size(a1,1),1) a1];
a2=sigmoid(a1*Theta2');
%[~,p]=max(a2,[],2);
size(Theta1);
size(Theta2);
%fprintf("\n size of a1 is: %d",size(a1))
%fprintf("\n size of a2 is: %d\n",size(a2))

number_of_labels=size(a2,2);
J=0;
for i=1:number_of_labels,
  y_new=(y==i);
  J= J + (1/m) * (((-1 * y_new') * log(a2(:,i))) - ((1-y_new') * log(1-a2(:,i))));
end

%Regularize cost function
rtheta1 = sum(sum(Theta1(:,2:end).^2));
rtheta2 = sum(sum(Theta2(:,2:end).^2));
bias = lambda/(2*m);

J = J + (bias * (rtheta1+rtheta2));




%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
% Theta1=[25 401]
%Theta2=[10 26]
%a1=[5000 26]
%a2=[5000 10]
for i=1:m
    y_new2(y(i),i)=1;  
endfor
for t=1:m
  a_1=X(t,:)';  %[400,1]
  %a_1=[1;a_1];  %[401,1]
  z_2=Theta1*a_1;     %[25 1]
  a_2=sigmoid(z_2);     %[25 1]
  a_2=[1;a_2];          %[26 1]
  z_3=Theta2*a_2;    %[10 1]
  a_3=sigmoid(z_3);   %[10 1]
  delta_3=a_3 - y_new2(:,t);    %[10 1]
  z_2=[1;z_2];
  delta_2=(Theta2'*delta_3).*sigmoidGradient(z_2);%[26*10]*[10*1]=[26*1]
  delta_2=delta_2(2:end);
  Theta2_grad=Theta2_grad + delta_3 * a_2'; %[10*26]
  Theta1_grad=Theta1_grad + delta_2 * a_1'; %[25*401]
endfor
Theta2_grad=(1/m)*Theta2_grad;
Theta1_grad=(1/m)*Theta1_grad;





%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+((lambda/m)*(Theta1(:,2:end)));
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+((lambda/m)*(Theta2(:,2:end)));















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
