###### library packages ###### 
library(arules)
library(tm)
library(SnowballC)
library(h2o)
library(xgboost)

###### Association Rule Learning ######
## Sparse Matrix 
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep=',', rm.duplicates = T)

summary(dataset)
size(head(dataset))
size(dataset) %>% quantile(prob = seq(0.1, 0.5, 0.1))

itemFrequencyPlot(dataset, topN = 10)

## 1. Apriori
rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.2))

rules_conf = sort(rules, by = 'confidence', decreasing = T)
rules_lift = sort(rules, by='lift', decreasing = T)
inspect(rules_conf[1:10])

rules2 = apriori(dataset, parameter = list(support = .001, confidence = 0.2, minlen = 2), 
                 appearance = list(default = 'rhs', lhs = 'cereals'), control = list(verbose = F))

library(arulesViz)
sub_rules = subset(rules, confidence > 0.9)
sub_rules
plot(sub_rules, method = 'matrix')
sub_rules2 = subset(rules, lift > 6)
plot(sub_rules2, method = 'graph')


## 2. Eclat  
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))
rules = eclat(dataset, parameter = list(support = 0.06, maxlen = 5))

inspect(head(rules))
A = sort(rules, by='support')
inspect(A[1:10])

itemsets = sort(rules)
as(items(itemsets), 'list')


#####################################################
###### Reinforcement Learning 
###### 1. Upper Confidence Bound ######
###### 2. Thompson Sampling ######

#####################################################
###### Deep Learning 
###### 1. Artificial Neural Networks ######
library(h2o)
h2o.init(nthreads = -1, max_mem_size = "2G")
h2o.removeAll()

glass = h2o.importFile("glassClass.csv")
head(glass)
summary(glass, exact_quantiles = T)
h2o.table(glass$K)

part = h2o.splitFrame(data = glass, ratios = .75)
part
training_h2o= part[[1]]
testing_h2o = part[[2]]

## Normal dataset -> h2o : Importing the dataset,
## Encoding the categorical variables as factors,
## Splitting the dataset, 
dataset = read.csv("dataset.csv")
set.seed(2018)
randomN = sample(nrow(dataset))
dataset = dataset[randomN, ]
library(caret)
part = createDataPartition(dataset$label, p = 0.8, list = F)
training = dataset[part, ]
testing = dataset[-part, ]

training_h2o = as.h2o(training)  
testing_h2o = as.h2o(testing)


## Fitting ANN 
classifier = h2o.deeplearning(y = 'label', training_frame = training_h2o, 
                              activation = 'Rectifier', hidden = c(6, 6), 
                              epochs = 100, train_samples_per_iteration = -2)

## Predicting 
h2o.confusionMatrix(classifier, testing_h2o)

pred.prob = h2o.predict(classifier, testing_h2o[ ,-ncol(testing)])
pred.y = (pred.prob>= 0.5) %>% as.vector()
table(pred.y, testing$label)


h2o.shutdown()
Y

###### 2. Convolutional Neural Networks ######

