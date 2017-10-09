# Neural Network prediction function 
nnPredict <- function(Theta1, Theta2, x) {
      
      m <- nrow(x)
      num_labels <- nrow(Theta2)
      
      p <- matrix(0, m, 1)
      
      t <- cbind(1, x)
      h1 <- sigmoid(t %*% t(Theta1))
      
      t <- cbind(1, h1)
      h2 <- sigmoid(t %*% t(Theta2))
      p <- apply(h2, 1, which.max)
      p
}