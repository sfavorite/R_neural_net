source('neural_net.R')

data(iris)

x <- as.matrix(iris[, 1:4])
y <- as.matrix(iris[, 5])


model <- neural_net(x, y)

pred <- predict(model, x)


print(sprintf("Training Set %f", mean(pred==y) * 100))
confusionMatrix(pred, y)
