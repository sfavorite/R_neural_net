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
      
      
      
}
