# Neural Network using back propagation 


neural_net <- function(x = x, y = y, hidden_layer_size = 25, trace=TRUE) {
      
      # functional will provide our optimization function: optim()  
      library(functional)
      source("randInitialWeights.R")
      source("nnCost.R")
      source("logistic_functions.R")
      source("nnGrad.R")
      source("predict.R")
      
      # Check data types and coerce if possible 
      x <- convert_data(x)
      
      # Store the levels to convert back from numeric  
      y_levels <-levels(y)
      
      
      # Set initial hyperparameters 
      input_layer_size <- ncol(x)
      num_labels <- length(unique(y))
      
      # Create random initial parameters 
      initial_Theta1 = randomInitialWeights(input_layer_size, hidden_layer_size)
      initial_Theta2 = randomInitialWeights(hidden_layer_size, num_labels)
      
      # Unroll parameters
      t1 <- unlist(initial_Theta1)
      t2 <- unlist(initial_Theta2)
      
      # Setup initial network parameters to use in optima() below
      initial_nn_params <- as.vector(c(t1, t2))
      
      # Pre-specify the procedures named parameters and use the return as the new procedure
      # We will use this in the optim() function
      lambda <- 2
      
      costF <- Curry(nnCost, input_layer_size=input_layer_size, hidden_layer_size=hidden_layer_size, num_labels=num_labels, x=x, y=y, lambda=lambda)
      grad <- Curry(nnGrad, input_layer_size=input_layer_size, hidden_layer_size=hidden_layer_size, num_labels=num_labels, x=x, y=y, lambda=lambda)
      
      # There are lots of different options for optim
      # Some different configs are commented out...please see ?optim() for more information 
      ctrl <- list(maxit=100, type=1, trace = TRUE)
      #ctrl <- list(maxit=100)
      #theta_optim <- optim(par=initial_nn_params, fn=costF, method="CG",  gr=grad, control = ctrl)
      theta_optim <- optim(par=initial_nn_params, fn=costF, method="BFGS",  gr=grad, control = ctrl)
      #theta_optim <- optim(par=initialTheta, fn=costF, x=x, lambda=lambda, y=this_y, gr=grad, method="BFGS", control = list(maxit=50))
      
      # Obtain Theta1 and Theta2 back from nn_params
      nn_params <- theta_optim$par
      Theta1 <- matrix(nn_params[1:(hidden_layer_size * ( input_layer_size + 1))], nrow=hidden_layer_size, ncol=input_layer_size+1)
      Theta2 <- matrix(nn_params[1+(hidden_layer_size * ( input_layer_size + 1)):(length(nn_params)-1)], nrow=num_labels, ncol=hidden_layer_size+1)
      
      #print(nn_params)
      
      message = theta_optim$message
      return (list(J = theta_optim$value, Theta1 = Theta1, Theta2 = Theta2, counts = theta_optim$counts, convergence = theta_optim$convergence, message = theta_optim$message, categories = y_levels))
      
}


predict <- function(model, x) {
      
      Theta1 <- model$Theta1
      Theta2 <- model$Theta2 
      
      p <- nnPredict(Theta1, Theta2, x)
      
      # Convert back to factor
      predictions <- factor(p, labels = model$categories)
      predictions
      
}

convert_data <- function(x) {
      
      if (is.list(x)) {
            x <- as.matrix(x)
      }
      x
}