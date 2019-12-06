
test = read.csv("Personality_Traits_Testset_v1.csv")
target_names = names(test)[2:8]

set.seed(48484)
submission = test[,1:8]
for (name in target_names) {
  submission[,name] = sample(1:nrow(test))
}
write.csv(submission, file = "./upload/random.csv", row.names = FALSE) 



train = read.csv("Personality_Traits_Trainingset_v1.csv")

#function for adding NAs indicators to dataframe and replacing NA's with a value---"cols" is vector of columns to operate on
#   (necessary for randomForest package)
appendNAs <- function(dataset, cols) {
  append_these = data.frame( is.na(dataset[, cols] ))
  names(append_these) = paste(names(append_these), "NA", sep = "_")
  dataset = cbind(dataset, append_these)
  dataset[is.na(dataset)] = -1
  return(dataset)
}

#replacements:
train <- appendNAs(train,9:ncol(train))
test <- appendNAs(test,9:ncol(test))

submissionRF = test[,1:8]
library("randomForest")
set.seed(939547)
for (name in target_names) {
  print(name)
  rf = randomForest(train[,9:ncol(train)],train[,name], do.trace=TRUE,importance=FALSE, sampsize = nrow(train)*.7, ntree = 100)
  predictions = predict(rf, test[,9:ncol(test)])
  submissionRF[,name] = predictions
}

write.csv(submissionRF, file = "./upload/random_forest.csv", row.names = FALSE) 





