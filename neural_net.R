# Neural Network using back propagation 

needed.libraries <- function() {
      # functional will provide our optimization function: optim()  
      library(functional)
      source("randInitialWeights.R")
      source("nnCost.R")
      source("logistic_functions.R")
      source("nnGrad.R")
      source("predict.R")
}

neural_net <- function(x = x, y = y, hidden_layer_size = 25) {
   
      # Load need functions 
      needed.libraries() 
      
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
      
      
}
