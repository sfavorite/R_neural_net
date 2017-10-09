# Return the total number of elements in a matrix
# A 4x5 matrix will return 20 
product <- function(data) {
      sizes <- dim(data)
      return (prod(sizes))
}

sigmoid <- function(x) {
      1 / (1 + exp(-x))
}

sigmoidGradient <- function(z) {
      #returns the gradient of the sigmoid function
      #evaluated at z
      #   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
      #   evaluated at z. This should work regardless if z is a matrix or a
      #   vector. In particular, if z is a vector or matrix, you should return
      #   the gradient for each element.
      
      if (!is.matrix(z)) {
            cols <- length(z)
            rows <- length(z)
      } else {
            cols <- ncol(z)
            rows <- nrow(z)
      }
      
      g = matrix(0, length(z))
      
      g = sigmoid(z) * (1 - sigmoid(z));
      g
}

cost <- function(theta, x, lambda, y) {
      m <- nrow(x)
      h <- sigmoid(x %*% theta)
      theta_reg <- theta
      theta_reg[1] <- 0
      reg_parameter <- (lambda/(2*m)) * t(theta_reg) %*% theta_reg
      #reg_parameter <- (lambda/(2*m)) * sum(theta ^ 2)
      # Commented sections are the same as the uncommented below...just in more steps
      # temp1 <- (-1 * (y * log(h)))
      # temp2 <- (1-y) * log(1-h)
      
      # sum1 <- sum(temp1 - temp2)
      # sumdiv <- 1/m
      # sum2 <- (1/m) * sum(temp1 - temp2)
      #sum3 <- (1/m) %*% sum(temp1 - temp2) + reg_paramater
      J <- (1/m) * sum((-y * log(h)) - ((1-y) * log(1-h))) + reg_parameter
      cat(sprintf("\r %i Cost: %f", i, J))
      #J <- (1/m) * sum(temp1 - temp2) + reg_paramater
      return (J)
}

grad <- function(x, y, lambda, theta) {
      m <- nrow(x)
      h <- sigmoid (x %*% theta)
      grad <- ((1/m) * (t(x) %*% (h - y))) + (lambda/m) * theta
      
      grad 
}

accuracy <- function(theta, y) {
      # Compute sigmoid of x times theta
      temp = sigmoid(x %*% theta)
      # How many are above 0.5
      p = temp >= 0.5
      return (mean(p == y) * 100)
}

# Maps the two input features to quadratic features used in regularization 
# Returns a new feature array with more features, comprising of 
# X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
mapFeatures <- function(x1, x2, degree) {
      degree <- 6;
      new_df <- matrix(1, length(x1))
      for (i in 1:degree) {
            for (j in 0:i) {
                  new_df <- cbind(new_df, (x1 ^ (i-j)) * (x2 ^ j))
            }
      }
      new_df
}
