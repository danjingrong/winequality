#loading data
wine.df <- read.csv("winequality-red.csv", stringsAsFactors = TRUE)

#percentage of the good quality of wine
nrow(wine.df[wine.df$quality>6.5,])
nrow(wine.df)

View(wine.df)

#kmeans
library(caret)
norm.values <- preProcess(wine.df, method=c("center", "scale"))
wine.df.norm <- predict(norm.values, wine.df)

#the seed number does not really matters
#and I think 4 will be a good number for cluster based on the elbow chart I generated
set.seed(1111)
km <- kmeans(wine.df.norm, 4)

km$cluster
km$centers
km$size


#classification tree
set.seed(1111)
#set the quality to shown as 'B' or 'G'
wine.df$quality <- ifelse(wine.df$quality > 6.5, "G", "B")  
#indexing 
train.index <- sample(1:nrow(wine.df), nrow(wine.df)*0.6) 

#partitionize the data into train and valid group
train.df <- wine.df[train.index, ]
valid.df <- wine.df[-train.index, ]

library(rpart)
library(rpart.plot)

#plot the classification tree
default.ct <- rpart(quality~ ., data = train.df, method = "class")
#prp(default.ct)
prp(default.ct, type=1, extra = 1)

#validating the model
default.ct.point.pred <- predict(default.ct, valid.df, type = "class")
confusionMatrix(default.ct.point.pred, factor(valid.df$quality), positive = "G")

#Logicstic Regression 

#set the base class
wine.df$quality <- as.numeric(wine.df$quality == "G")

#indexing and partitionize the data
train.index <- sample(1:nrow(wine.df), nrow(wine.df)*0.6)  
train.df <- wine.df[train.index, ]
valid.df <- wine.df[-train.index, ]

#come up with the logit
logit.reg <- glm( quality~ ., data = train.df, family = "binomial")
summary(logit.reg)

#this is the model you suggested
#logit.reg <- glm( quality~ (volatile.acidity+sulphates+alcohol), data = train.df, family = "binomial")
#summary(logit.reg)

#validating the data, setting the cut of value = 0.2(the percentage of the good in our data set = 0.13)
logit.reg.pred <- predict(logit.reg, valid.df,  type = "response")
pred <- ifelse(logit.reg.pred > 0.2, 1, 0)  

#draw the confusionMatrix
library(caret)
confusionMatrix(factor(pred), factor(valid.df$quality), positive = "1")


library(pROC)

#plot the ROC graph
r <- roc(valid.df$quality, logit.reg.pred)
plot.roc(r)

#come up with the threshold
coords(r, x = "best", transpose = FALSE)





















