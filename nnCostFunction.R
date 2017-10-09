# This cost function returns a list with the cost (J) and the gradient.
# It can NOT be used with optim(), the optimization function used in ex4.R,
# fail on seeing a list since optim() expects a scalar and will.

# optim is the general-purpose optimization in the stats package
# For more information: ?optim()
# To get the cost (J) returned as a scalar use nnCost.R

nnCostFunction <- function(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lambda) {
      
      # Create a matrix of weights
      # Theta1: input to hidden layer
      # Theta2: hidden layer to output layer
      Theta1 <- matrix(nn_params[1:(hidden_layer_size * ( input_layer_size + 1))], nrow=hidden_layer_size, ncol=input_layer_size+1)
      Theta2 <- matrix(nn_params[1+(hidden_layer_size * ( input_layer_size + 1)):(length(nn_params)-1)], nrow=num_labels, ncol=hidden_layer_size+1)
      
      # Number of rows
      m <- nrow(x)
      # Add a bias unit
      x <- cbind(1, x)
      
      # Start error (J) at zero
      J <- 0;
      
      Theta1_grad <- matrix(0, nrow(x), ncol(x))
      Theta2_grad <- matrix(0, nrow(x), ncol(x))
      I = diag(num_labels)
      Y_ten = matrix(0, nrow=m, ncol=num_labels)
      #for (i in 1:m) {
      #      Y_ten[i, ] = I[y[i], ]
      #}
      # Convert y to a dataframe for One-hot-encoding 
      df_y <- as.data.frame(y)
      colnames(df_y) <- c("y")
      # We need factors to do One-hot-encoding 
      df_y$y <- as.factor(df_y$y)
      # Encode y
      Y_ten <- model.matrix(~ . + 0, data=df_y, contrasts.arg = lapply(df_y, contrasts, contrasts=FALSE))
      
      
      a1 = x
      z2 = a1 %*% t(Theta1)
      a2 = sigmoid(z2)
      a2 = cbind(1, a2)
      z3 =  a2 %*% t(Theta2);
      a3 = sigmoid(z3);
      
      # Regularization
      reg = (lambda / (2*m)) * (sum(sum(Theta1[, -1] ^ 2)) + sum(sum((Theta2[, -1]) ^ 2)))
      
      J <- (1/m) * sum(sum(-Y_ten * log(a3) - (1 - Y_ten) * log(1 - a3))) + reg
      
      delta_3 <- a3 - Y_ten;
      delta_2 <- (delta_3 %*% as.matrix(Theta2[, -1])) * sigmoidGradient(z2)
      
      delta_cap2 <- t(delta_3) %*% a2;
      delta_cap1 <- t(delta_2) %*% a1;
      
      Theta1_grad <- ((1/m) * delta_cap1) + ((lambda/m) * (Theta1));
      Theta2_grad <- ((1/m) * delta_cap2) + ((lambda/m) * (Theta2));
      
      Theta1_grad[, 1] <- Theta1_grad[, 1] - ((lambda/m) * (Theta1[, 1]));
      Theta2_grad[, 1] <- Theta2_grad[, 1] - ((lambda/m) * (Theta2[, 1]));
      
      grad <- unlist(c(Theta1_grad, Theta2_grad), use.names=FALSE)
      
      return(list(J = J, grad = grad))
      
}
