# winequality
Predict quality of wine based on chemical components


Description of the Dataset 
For this project, we chose the Red Wine Quality Dataset from Kaggle. The data is collected by the UCI machine learning repository with 1599 red wine examples. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). This dataset has 19,188 observations of 12 variables. Below are the variables: 

Based on the goal of the project, we will use quality as the target variable. 
The quality variable is the sensory score, which is the median value of three evaluations that are given by a minimum of three sensory assessors. 

Data Preparation Details
From the dataset table above, we can find that all the data types are numeric. The dataset does not contain any null values but the unit and the scale of the chemical composition are not normalized. We need to factor in some of the variables to run the supervised learning such as classification tree and logistic regression and we also need to pre-normalize the data before we process the unsupervised learning such as k-means algorithm.  

Data Visualization & Exploratory Analysis
Figure 2-1
Good/Bad Wine: In order to identify what makes up a good or bad wine, we created a new column called good wine and standardized what makes up a good wine by learning more about the target variable: Quality. As the dataset suggested that quality greater than 6.5 will be considered as good and a bad wine will be with a quality lower than 6.5. We labelled 0 as bad wine and 1 as good wine. The distribution result is shown as Figure 2-1. We can see that the majority of the red wines were rated bad with an approximate percentage of 86% while only 14% were rated good. 

Figure 2-2
To get a better breakdown of the wine distribution, we constructed a bar chart (Figure 2-2) that is based on the overall quality rating. Essentially, the response variable (Quality) ranges from 3 to 8 with the majority of the wine piling up with a quality of either a 5 or 6. However, since the threshold of a good wine is a 6.5, only a rating of 7 and 8 can be considered good, which makes up only 14% of the entire distribution, leaving the remaining 86% as a bad wine, supporting our initial assumption of how bad wine outweighs good wine.

Figure 2-3
Goal: In order to determine what influences a wine to be bad or good, we start off by plotting a series of bar charts of each independent variable with the dependent variable (Quality) to see if there are any relevant relations as seen from Figure 2-3. From the above illustration (Figure 2-3), we make our assumption that:
Variables such as Fixed Acidity, Residual Sugar, Free Sulfur Dioxide, Total Sulfur Dioxide, Density and pH might not give any specification to classify/predict the quality since they are not shown with obvious trends
Quality increases with:
decrease in volatile acidity
increase in citric acid
decrease in chlorides
increase in sulphate
increase in alcohol

Correlation Between Variables: To verify the above statements, we believed that the most effective method to recognize the potential relationships/correlations amongst the different predictor variables is to leverage a pair-wise correlation matrix and depict it as a heatmap, illustrated in Figure 2-4. The gradients in the heatmap vary based on the 
Figure 2-4

strength of the correlation and you can clearly see it is very easy to spot potential attributes having strong correlations amongst themselves. As our goal is to find which predictor variables have the strongest correlation with the target variable quality, we will only focus on the row that contains quality. Additionally, Correlation ranges from -1 to +1; if values are closer to 0 can be concluded to have no linear trend between the two desired variables. Correlation closer to +1 means that the two variables are positively related while a correlation closer to -1 means that the two variables are negatively related. Variables that lean towards +1 or -1 are just strongly correlated, the only difference is correlation closer to +1 has a positive trend while correlation with -1 has a negative trend. Based on our observations, we find that alcohol (a positive correlation of +0.48) , sulphate (a positive correlation of +0.25) and volatile acid (a negative correlation of -0.39) have the strongest correlation with response to the target variable, quality.

Data Modeling and the Methodology
 
The main goal of our project is to predict the quality of the red wine based on its chemical composition. The dataset suggested that quality scores higher than 6.5 will be considered as good quality of wine; otherwise, it is considered as bad quality. This logic helps us to turn this problem into a classification task. We tried to come up with a model that can learn what significant factors will be contributing to higher quality of wine based on the data analysis. We use logistic regression and classification trees to build our model, and we use k-means to try to see the characteristics of good wine and validate the data with our model later.
 
Characteristic Exploration 
Method: K-means Algorithm
Process:
We tried to use the k-means method to see if there is a similar pattern of chemical composition within the red wine dataset and what will be their scores. To perform the clustering activity, we first selected all the attributes in the red wine data and normalized them in order to avoid that the largest scales would dominate and skew results. Then, we tried to find the optimal number of cluster k. We tried to choose the number of cluster k using two approaches as below.

Approach 1:  The “good” and “bad” quality of wine should have relatively distinct clusters for their chemical composition, so we use k = 3 and expecting to see two similar groups of clusters and one relatively distinct cluster.
Approach 2: We compute the average within-cluster distance to find the optimal number of clusters.
 
Results:
1) 	The result from assumption 1 is not what we expected to be (Figure3-1). There is no obvious distinction with two groups and the other. Each group has its own pattern, and it is hard for us to segment them with our target result.
2) 	By generating the elbow chart (Figure 3-2), we could see before k = 5, the average within-cluster distance drops rapidly and after that, it decreases more slowly. With k=5, we come up with the following result (Figure 3-3). We observe the pattern for clusters that have relatively high quality scores (Cluster 2&4): both low in volatile acidity; both high in sulphates and alcohol.
3) 	We use the k-means method to find the possible factors relating to the quality score and also use it to validate with the model we build.  
 

Figure 3-1
 
Figure 3-2

 
      Figure 3-3
 

Model 1: Classification Tree
Method: We build a classification tree to see what the “rule” is for predicting wine quality. We turn the numeric quality scores into categorical data “B”and “G”. We randomly split this data set into a 60% training set and 40% validation set with a seed of 1111.  We use a confusion matrix to validate and compute the accuracy of our scores.
Initial Approach: We first use all predictors available in the dataset to build the classification tree.
Improving the Model: We only took the first three root factors shown in the classification tree (alcohol, sulphate, volatile acidity)to see if the accuracy of the tree improves. 
Results:
1)   The initial tree shown in Figure 3-4. It has 13 roots and an accuracy of 0.8797 with kappa=0.3171. It gives a specificity of 0.97122 and a sensitivity of 0.27381.
2)   The second tree shown in Figure 3-5. It has 15 roots and an accuracy of 0.8719 with kappa=0.3944. It gives a specificity of 0.96209 and a sensitivity of 0.29070.   

Figure 3-4                                                        Figure 3-5
 
Model 2: Logistic Regression
Method: We tried to come up with logistics regression to see if a regression can be built as the logit of the wine quality, and we also want to see what the p-values for each factor is. We set the “good quality” associated with “1” status. We randomly split this data set into a 60% training set and 40% validation set with a seed number of 1111. As the percentage of the good quality wine in this data set is approximately 13.7%, we set our cut-off value to be 0.2 and use a confusion matrix to validate and compute the accuracy of the probability. We also come up with the ROC graph and threshold to find out the best response in our model.
 
Initial Approach: We first use all predictors available in the dataset to build the logistics regression.
Improving the Model: We only took the factor with *** and ** correlation shown in the first logistics regression model to see if the accuracy increases.
Results:
1) 	The initial regression model result shown as Figure 3-6. This logistics regression gives us an accuracy of 0.8359 with kappa of 0.4139. However, the coefficients in this model are relatively small in number, but the p-values do show a strong correlation within the factors like sulphates, alcohol, and volatile acidity. The roc graph is attached in Figure 3-7. The threshold cut-off value of this model is 0.07831512, achieving a specificity of 0.7356115 and a sensitivity of 0.8571429.
 
2) 	The second logistic regression shown as Figure 3-8, with an accuracy of 0.8344, a kappa of 0.4383. The threshold cut-off value of this model is 0.159432, achieving a specificity of 0.8201439 and a sensitivity of 0.7619048. The coefficient in this model is relatively larger in scale compared to the first model. The roc graph is attached in Figure 3-9.
 

Figure 3-6                                                  Figure 3-8

Figure 3-7                                                     Figure 3-9





Results and Insights

Let us now try to understand the results of the Classification tree and the logistic regression models.
Looking at the visualization from Figure 2-3 and comparing it with the Classification tree Figure 3-4, we can see that the few factors are more important to a quality of wine than others. They are as below:
Alcohol
Sulphates
Volatile acidity

When we tried to plot a classification using all the predictors, we could see that the tree is classified based on these factors exactly in the order of decreasing weightage. So, to get a more clear view of the tree, we next tried to restrict the list of predictors to only the top four as seen in the resulting tree in Figure 3-5. This tree gives a better shot at understanding the results and also gives a clearer picture of the analysis. If we have values like alcohol levels = 2 and volatile acidity = 0.6 then we can say that it would be a ‘bad quality’ wine.
Let us now have a look at the logistic regression output and try to understand it. Looking at the coefficient values from Figure 3.6 we can see that exactly as the results from classification tree analysis, the predictors which most influence the quality of wine are alcohol , sulphates and volatile acidity. Looking at the results from Figure 3-8 where we used only the top three predictors (volatile acidity, sulphates and alcohol), we can deduce that as the value of volatile acidity increases the quality of wine would decrease since the corresponding coefficient is a negative value. Similarly, for  other two predictors of sulphates and alcohol we can deduce that as their values would increase the quality of wine would also improve since they have positive coefficient values. This analysis would be extremely useful when making a decision on the wine quality.

These above results can be used by below groups of people and can benefit them
Farmers - if they can figure out methods to increase the quality of grapes they produce based on the influence of factors like sulphates, acidity and the alcohol content. 
Manufacturer of wine - even if the quality of grapes does not match the desired level, the manufacturer can use natural ingredients to boost the quality of wine in the process of wine fermentation.
Consumers - each bottle of wine has its composition printed on the bottle which would give the percentage of each component in the wine. With these insights, the consumer can decide whether they can treat it as a good quality or not and buy accordingly.


Conclusion
In this project, we would like to know the formula for high-quality wine--what chemical components influence the quality of wine--from this specific dataset. For exploratory analysis, we used bar charts to look at the distribution of quality value among the samples, then looked at the correlation among chemicals with a correlation heatmap. We have found out the amount of alcohol, sulphates, volatile acidity and free sulphur dioxide are highly correlated with quality, thus forming our basic hypothesis about them being the key predictors in our model. Furthermore, we applied and adjusted three models to conduct regression analysis, experimenting with the key predictors and quality as the response variable. The key predictors do play important parts in these three models, but their order of importance slightly differs. 
Further examination is needed for figuring out the order difference for predictors. This requires that we interrogate the different methodology of the models again--how they measure distance/correlation, and what they are fit for--in order to develop further understanding of our results. We could also run other models, for example, random forest, to test if the result differs from classification tree. After we introduce other models and further study the different results, we could look at what  data we need to expand our scope. For example, if we have the grape types and selling price of the wines along with quality and chemical components, we can find out what kind of grape produces higher quality of wine, and higher quality will lead to higher price or profit. 
With the assistance of these supplementary data, our model could help wine companies predict the quality of their product. They can base on the model to choose the kind of grape that produces better quality or higher profit; it could also help the consumers predict the quality when they read the chemical composition on a wine bottle. Applied in the real world, our findings could assist both wine makers and consumers in quality and taste predicting.
