source('neural_net.R')

data(iris)

# define an 80%/20% train/test split of the dataset
split=0.80
trainIndex <- createDataPartition(iris$Species, p=split, list=FALSE)
# Make a training and test set 
data_train <- iris[ trainIndex,]
data_test <- iris[-trainIndex,]

# Train our network 
model <- neural_net(as.matrix(data_train[, 1:4]), as.matrix(data_train[, 5]))

# Make predictions
pred <- predict(model, as.matrix(data_test[, 1:4]))

# How did we do?
print(sprintf("Training Set %f", mean(pred==data_test[, 5]) * 100))
confusionMatrix(pred, data_test[, 5])
