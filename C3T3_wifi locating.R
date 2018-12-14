##Import libraries
library(readr)
library(caret)
library(dplyr)
library(ggplot2)
library(corrplot)
library(gdata)

##Parallel Computing##
#--- for Win ---#
library(doParallel) 

# Check number of cores and workers available 
detectCores()
getDoParWorkers()
cl <- makeCluster(detectCores()-1, type='PSOCK')
registerDoParallel(cl)

##Import data sets
trainData_raw <- read.csv("trainingData_mod.csv")

##Preprocessing 1 - Examine data set complete survey
summary(trainData_raw)
str(trainData_raw)

trainData_raw$FLOOR <- as.factor(trainData_raw$FLOOR)
trainData_raw$BUILDINGID <- as.factor(trainData_raw$BUILDINGID)
trainData_raw$SPACEID <- as.factor(trainData_raw$SPACEID)
trainData_raw$RELATIVEPOSITION <- as.factor(trainData_raw$RELATIVEPOSITION)
trainData_raw$USERID <- as.factor(trainData_raw$USERID)
trainData_raw$PHONEID <- as.factor(trainData_raw$PHONEID)

str(trainData_raw)

## Combine Floor, building, space and relative position into one position attribute
trainData_raw <-cbind(trainData_raw,paste(trainData_raw$FLOOR,trainData_raw$BUILDINGID,
                                   trainData_raw$SPACEID, trainData_raw$RELATIVEPOSITION), 
               stringsAsFactors=FALSE)

## Give the new attribute in the 530th column a header name
colnames(trainData_raw)[530] <- "LOCID"

## Move the LOCID attribute within the dataset
trainData_raw <- trainData_raw[,c(1,2,3,4,5,6,530,7:529)]
trainData_raw$LOCID <- as.factor(trainData_raw$LOCID)
str(trainData_raw)

## Create subset 1 of trainData_raw - create a random sample of 25%
set.seed(123)
sub1_ind <- createDataPartition(trainData_raw$LOCID, p=0.25,
                               list = FALSE)
sub1_train <-trainData_raw[sub1_ind, ]
nrow(sub1_train)

## Create subset 2 of trainData_raw - use only building 0
sub2_train <- filter(trainData_raw, BUILDINGID == 0)
nrow(sub2_train)

## Create sub1B_train from sub1_train - classify building ID instead of LOCID
sub1B_train <- sub1_train
str(sub1B_train)

## Drop attributes that are not needed for modeling for sub-train data
sub1_train[,c(1:6,8:10)] <- NULL
str(sub1_train)
sub1_train <- drop.levels(sub1_train)
str(sub1_train)

sub2_train[,c(1:6,8:10)] <- NULL
sub2_train <- drop.levels(sub2_train)
str(sub2_train)

sub1B_train[,c(1:3,5:10)] <-NULL

## Define training and test set from sub1_train and sub2_train
inTrain1 <- createDataPartition(sub1_train$LOCID,
                               p=0.75,
                               list = FALSE)
training1 <- sub1_train[inTrain1, ]
testing1 <- sub1_train[-inTrain1, ]
nrow(training1)
nrow(testing1)

inTrain2 <- createDataPartition(sub2_train$LOCID,
                                p=0.75,
                                list = FALSE)
training2 <- sub2_train[inTrain2, ]
testing2 <- sub2_train[-inTrain2, ]
nrow(training2)
nrow(testing2)

inTrain1B <- createDataPartition(sub1B_train$BUILDINGID,
                                p=0.75,
                                list = FALSE)
training1B <- sub1B_train[inTrain1B, ]
testing1B <- sub1B_train[-inTrain1B, ]
nrow(training1B)
nrow(testing1B)

## training model with kNN, with 10 fold CV,tuneLength = 20, on sub1_train
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(knn_Fit1 <- train(LOCID ~.,
                             data = training1,
                             method = 'knn',
                             tuneLength = 20,
                             trControl = ctrl))
knn_Fit1

## training model with kNN, with 10 fold CV,tuneLength = 8, on sub2_train
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(knn_Fit2 <- train(LOCID ~.,
                              data = training2,
                              method = 'knn',
                              tuneLength = 8,
                              trControl = ctrl))
knn_Fit2

## training model with Random Forest, with 10 fold CV,tuneLength = 5, on sub1_train
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(rf_Fit1 <- train(LOCID ~.,
                              data = training1,
                              method = 'rf',
                              tuneLength = 5,
                              trControl = ctrl))
rf_Fit1

## training model with Random Forest, with 10 fold CV,tuneLength = 3, on sub1B_train
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(rf_Fit1B <- train(BUILDINGID ~.,
                             data = training1B,
                             method = 'rf',
                             tuneLength = 3,
                             trControl = ctrl))
rf_Fit1B

## training model with random forest, with 10 fold CV,tuneLength = 5, on sub2_train
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(rf_Fit2 <- train(LOCID ~.,
                              data = training2,
                              method = 'rf',
                              tuneLength = 5,
                              trControl = ctrl))
rf_Fit2

## training model with C5.0, with 10 fold CV,tuneLength = 3, on sub1_train
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(c50_Fit1 <- train(LOCID ~.,
                             data = training1,
                             method = 'C5.0',
                             tuneLength = 3,
                             trControl = ctrl))
c50_Fit1

## training model with C5.0, with 10 fold CV,tuneLength = 3, on sub2_train
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 1)

system.time(c50_Fit2 <- train(LOCID ~.,
                             data = training2,
                             method = 'C5.0',
                             tuneLength = 3,
                             trControl = ctrl))
c50_Fit2


##Prediction based on test set from sub1_train and sub2_train
knn_Pred1 <- predict(knn_Fit1, newdata = testing1)
postResample(knn_Pred1,testing1$LOCID)

knn_Pred2 <- predict(knn_Fit2, newdata = testing2)
postResample(knn_Pred2,testing2$LOCID)

rf_Pred1 <- predict(rf_Fit1, newdata = testing1)
postResample(rf_Pred1,testing1$LOCID)

rf_Pred2 <- predict(rf_Fit2, newdata = testing2)
postResample(rf_Pred2,testing2$LOCID)

rf_Pred1B <- predict(rf_Fit1B, newdata = testing1B)
postResample(rf_Pred1B,testing1B$BUILDINGID)

c50_Pred1 <- predict(c50_Fit1, newdata = testing1)
postResample(c50_Pred1,testing1$LOCID)

c50_Pred2 <- predict(c50_Fit2, newdata = testing2)
postResample(c50_Pred2,testing2$LOCID)

##Compare models with resample
ModelData <- resamples(list(kNN = knn_Fit1, RF = rf_Fit1, C50 = c50_Fit1))
summary(ModelData)

#Generate comfusion matrix
confusionMatrix(data = rf_Pred1, reference = testing1$LOCID, mode = "prec_recall")
confusionMatrix(data = rf_Pred2, reference = testing2$LOCID, mode = "prec_recall")
confusionMatrix(data = knn_Pred1, reference = testing1$LOCID, mode = "prec_recall")
confusionMatrix(data = knn_Pred2, reference = testing2$LOCID, mode = "prec_recall")
confusionMatrix(data = c50_Pred1, reference = testing1$LOCID, mode = "prec_recall")
confusionMatrix(data = c50_Pred2, reference = testing2$LOCID, mode = "prec_recall")