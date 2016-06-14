setwd("~/Bellarmine/Summer Assignments")
#setwd('C:\\Users\\bgubser\\OneDrive\\Class Files\\Data Mining - Summer 2015\\Assignments')
if("e1071" %in% rownames(installed.packages()) == FALSE)  {install.packages("e1071")}
library(e1071)
options(scipen=999)

#read in data
rawdata <- read.csv('BreastCancerData.txt')

#replace obviously erroneous data with NAs to remain consistent with missing data
cleandata <- rawdata
cleandata$Perimeter_Mean[cleandata$Perimeter_Mean == 999 ] <- NA
cleandata$Smoothness_Mean[cleandata$Smoothness_Mean == 999 ] <- NA
cleandata$Concavity_Mean[cleandata$Concavity_Mean == 0 ] <- NA
cleandata$Concave_pts_Mean[cleandata$Concave_pts_Mean == 0 ] <- NA

#Impute 4 cols above and Texture using mean as lm didn't appear good method and not a lot of missing data
imputeddata <- cleandata
imputeddata$Texture_Mean[is.na(imputeddata$Texture_Mean)] <- mean(imputeddata$Texture_Mean[!is.na(imputeddata$Texture_Mean)])
imputeddata$Perimeter_Mean[is.na(imputeddata$Perimeter_Mean)] <- mean(imputeddata$Perimeter_Mean[!is.na(imputeddata$Perimeter_Mean)])
imputeddata$Smoothness_Mean[is.na(imputeddata$Smoothness_Mean)] <- mean(imputeddata$Smoothness_Mean[!is.na(imputeddata$Smoothness_Mean)])
imputeddata$Concavity_Mean[is.na(imputeddata$Concavity_Mean)] <- mean(imputeddata$Concavity_Mean[!is.na(imputeddata$Concavity_Mean)])
imputeddata$Concave_pts_Mean[is.na(imputeddata$Concave_pts_Mean)] <- mean(imputeddata$Concave_pts_Mean[!is.na(imputeddata$Concave_pts_Mean)])


#Create column to identify train and test data
set.seed(395285)  #For replication purposes
rand <- rbinom(nrow(imputeddata), 1, .7)
imputeddata$partition <- "test"
imputeddata$partition[rand==1] <- "train"  


#naiveBayes Model
naiveModel <- naiveBayes(Diagnosis ~ 
                         Texture_Mean + Perimeter_Mean + Area_Mean + Smoothness_Mean + Compactness_Mean + Concavity_Mean +
                         Concave_pts_Mean + Symmetry_Mean + Fractal_dim_Mean, data=imputeddata[imputeddata$partition=="train",])
bayesPrediction <- predict(naiveModel, imputeddata[imputeddata$partition=="train",])
trainTableBayes <- table(bayesPrediction, imputeddata$Diagnosis[imputeddata$partition=="train"])
#Misclassification Rate    #Sensitivity      #Specificity
BayesMisClassTrain <- 1-sum(diag(prop.table(trainTableBayes)))
BayesSensitivityTrain <- trainTableBayes["B","B"]/(trainTableBayes["B","B"] + trainTableBayes["M","B"])
BayesSpecificityTrain <- trainTableBayes["M","M"]/(trainTableBayes["M","M"] + trainTableBayes["B","M"])


#Logisitc Regression Model
cutoff <- 0.44 #.44 cutoff - like the sensitivity and specificity rate after examining 101 cuttoff values
logData <- imputeddata[imputeddata$partition=="train",]
logisticModel <- glm(Diagnosis ~ 
      Texture_Mean + Perimeter_Mean + Area_Mean + Smoothness_Mean + Compactness_Mean + Concavity_Mean +
      Concave_pts_Mean + Symmetry_Mean + Fractal_dim_Mean, family = binomial, data=logData)
logData$prediction <- predict(logisticModel, type='response')
logData$predictionBM <- 'B'
logData$predictionBM[logData$prediction>cutoff] <- 'M' 
#Misclassification Rate    #Sensitivity      #Specificity
LogMisClassTrain <- 1-sum(diag(prop.table(table(logData$Diagnosis, logData$predictionBM))))
LogSensitivityTrain <- prop.table(table(logData$Diagnosis, logData$predictionBM), margin=2)[1]
LogSpecificityTrain <- prop.table(table(logData$Diagnosis, logData$predictionBM), margin=2)[4]


#Use Models To Predict on Test Data

#Bayes Test
testbayesPrediction <- predict(naiveModel, imputeddata[imputeddata$partition=="test",])
testTableBayes <- table(testbayesPrediction, imputeddata$Diagnosis[imputeddata$partition=="test"])
#Misclassification Rate    #Sensitivity      #Specificity
BayesMisClassTest <- 1-sum(diag(prop.table(testTableBayes)))
BayesSensitivityTest <- testTableBayes["B","B"]/(testTableBayes["B","B"] + testTableBayes["M","B"])
BayesSpecificityTest <- testTableBayes["M","M"]/(testTableBayes["M","M"] + testTableBayes["B","M"])

#Logisitic Test
testData <- imputeddata[imputeddata$partition=="test",]
testData$prediction <- predict(logisticModel, newdata=testData, type='response')
testData$predictionBM <- 'B'
testData$predictionBM[testData$prediction>cutoff] <- 'M'
#Misclassification Rate    #Sensitivity      #Specificity
LogMisClassTest <- 1-sum(diag(prop.table(table(testData$Diagnosis, testData$predictionBM))))
LogSensitivityTest <- prop.table(table(testData$Diagnosis, testData$predictionBM), margin=2)[1]
LogSpecificityTest <- prop.table(table(testData$Diagnosis, testData$predictionBM), margin=2)[4]

#Print Results
c(
paste('Naive Bayes Training Statistics:          Misclass ',round(BayesMisClassTrain,3)*100,
      '%   Sensitivity ',format(round(BayesSensitivityTrain,3)*100,nsmall=1),
      '%   Specificity ',format(round(BayesSpecificityTrain,3)*100,nsmall=1),'%',
      sep=''),
paste('Naive Bayes Test Statistics:              Misclass ',round(BayesMisClassTest,3)*100,
      '%   Sensitivity ',format(round(BayesSensitivityTest,3)*100,nsmall=1),
      '%   Specificity ',format(round(BayesSpecificityTest,3)*100,nsmall=1),'%',
      sep=''),
paste('Logistic Regression Training Statistics:  Misclass ',round(LogMisClassTrain,3)*100,
      '%   Sensitivity ',format(round(LogSensitivityTrain,3)*100,nsmall=1),
      '%   Specificity ',format(round(LogSpecificityTrain,3)*100,nsmall=1),'%',
      sep=''),
paste('Logistic Regression Test Statistics:      Misclass ',round(LogMisClassTest,3)*100,
      '%   Sensitivity ',format(round(LogSensitivityTest,3)*100,nsmall=1),
      '%   Specificity ',format(round(LogSpecificityTest,3)*100,nsmall=1),'%',
      sep=''))

