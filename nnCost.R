# This cost function returns the cost (J) as a scalar and can be used with optim()
# optim is the general-purpose optimization in the stats package 
# For more information: ?optim() 
# To get the gradient with the cost see nnCostFunction.R

nnCost <- function(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lambda) {
      
      
      Theta1 <- matrix(nn_params[1:(hidden_layer_size * ( input_layer_size + 1))], nrow=hidden_layer_size, ncol=input_layer_size+1)
      Theta2 <- matrix(nn_params[1+(hidden_layer_size * ( input_layer_size + 1)):(length(nn_params)-1)], nrow=num_labels, ncol=hidden_layer_size+1)
      
      m <- nrow(x)      
      x <- cbind(1, x)
      
      J <- 0;
      
      # Create a matrix of zeros to hold the gradient 
      Theta1_grad <- matrix(0, nrow(x), ncol(x))
      Theta2_grad <- matrix(0, nrow(x), ncol(x))
      # Create an identity metrix 
      I = diag(num_labels)
      Y_ten = matrix(0, nrow=m, ncol=num_labels)
      #for (i in 1:m) {
      #    print(length(I[y[i], ]))
      #    Y_ten[i, ] = I[y[i], ]
      #}
      # Convert y to a dataframe for One-hot-encoding 
      df_y <- as.data.frame(y)
      colnames(df_y) <- c("y")
      # We need factors to do One-hot-encoding 
      df_y$y <- as.factor(df_y$y)
      # Encode y
      Y_ten <- model.matrix(~ . + 0, data=df_y, contrasts.arg = lapply(df_y, contrasts, contrasts=FALSE))
      
      # Input layer 
      a1 = x
      
      # Hidden Layer 
      z2 = a1 %*% t(Theta1)  
      a2 = sigmoid(z2)
      a2 = cbind(1, a2)
      
      #Output Layer 
      z3 =  a2 %*% t(Theta2);
      a3 = sigmoid(z3);
      
      # Compute regularization 
      reg = (lambda / (2*m)) * (sum(sum(Theta1[, -1] ^ 2)) + sum(sum((Theta2[, -1]) ^ 2)))
      # Compute the cost using Cross-entropy error function
      J <- (1/m) * sum(sum(-Y_ten * log(a3) - (1 - Y_ten) * log(1 - a3))) + reg
      
      return(J)
      
}
