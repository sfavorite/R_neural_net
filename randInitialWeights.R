randomInitialWeights <- function(layers_in, layers_out) {
      InitialWeights <- matrix(0, nrow=layers_in, ncol=layers_out + 1)     
      
      epsilon_init = 0.12;
      InitialWeights <- matrix(runif((layers_in + 1) * layers_out, 0, 1) * 2 * epsilon_init - epsilon_init, nrow=layers_in + 1, ncol=layers_out)
      return (InitialWeights)
      
}
