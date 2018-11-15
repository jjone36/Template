# https://www.latex-project.org/get/
# devtools ####
Sys.setenv(PATH = paste(Sys.getenv("PATH"), "*InstallDirectory*/Rtools/bin/",
                        "*InstallDirectory*/Rtools/mingw_64/bin", sep = ";"))
Sys.setenv(BINPREF = "*InstallDirectory*/Rtools/mingw_64/bin")
library(devtools)

assignInNamespace("version_info", c(devtools:::version_info, list("3.5" = list(version_min = "3.3.0", version_max = "99.99.99", path = "bin"))), "devtools")
find_rtools()
install.packages("devtools")
library(devtools)
devtools::install_github("bmschmidt/wordVectors")
library(wordVectors)

###### Setting ######
# par(mfrow=c(1,1))

# getOption("digits")
# options(digits = 3)

# Grid-search : Optimize the Hiperparameter of Models
# http://topepo.github.io/caret/index.html

normal = function(x){
  return((x - min(x))/(max(x)-min(x)))
}

rmse = function(x){
  return(sqrt(mean(x^2)))
}

# http://www.stat.columbia.edu/~tzheng/files/Rcolor.pdf
'lightseagreen'
'mediumseagreen'
'indianred1'
myRed = '#99000D'
myBlue = '#377EB8'
myPink = '#FEE0D2'
'lightblue'

myTheme = theme(axis.text.x = element_text(angle = 45, hjust = 1),
                panel.background = element_rect(fill = 'white', color = 'grey50'),
                panel.grid.major = element_line(color = 'grey90'))

###### library packages ######
library(tidyverse)
library(magrittr)
library(ggthemes)
library(rebus)
library(broom)

library(caret)
library(car)
library(scales)
library(plyr)
library(reshape2)
library(lubridate)

library(ggthemes)
library(gridExtra)
library(corrplot)
library(Metrics)
library(pROC)
library(MASS)
library(randomForest)
library(GGally)
library(memisc)

#####################################################
## Importing the dataset
dataset = read.csv('Data.csv', stringsAsFactors = T)

dim(dataset)
names(dataset)
str(dataset)
summary(dataset)
head(dataset, n = 15)

## Remove unrelated variables
## Encoding categorical variables as factors
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

library(tidyverse)
table(dataset$Geography)
dataset$Geography = factor(x = dataset$Geography,
                           levels = c('France', 'Germany', 'Spain'),
                           labels = c(1, 2, 3)) %>% as.numeric()
table(dataset$Gender)
dataset$Gender = factor(x = dataset$Gender,
                        levels = c('Female', 'Male'),
                        labels = c(1, 2)) %>% as.numeric()

###### Dummy variables ######
library(dummies)
dummy.data.frame(df)

library(vtreat)
treatplan = designTreatmentsZ(dataset, c('carat', 'cut'), verbose = F)
(A = treatplan$scoreFrame %>%
  filter(code %in% c('clean', 'lev')) %>%
  select(varName, origName, code))
newvars = A$varName

treat_df = prepare(treatplan, dataset, varRestriction = newvar)
head(treat_df)

###### Scaling & Splitting the dataset ######
dataset[, -ncol(dataset)] = scale(dataset[, -ncol(dataset)])

set.seed(2018)
randomN = sample(nrow(dataset))
dataset = dataset[randomN, ]

library(caret)
part = createDataPartition(dataset$price, p = 0.8, list = F)
training = dataset[part, ]
testing = dataset[-part, ]

###### Sampling dataset ######
set.seed(2018)
sampleData = sample(x = dataset, size = 16)

r_assign = sample(1:3, size = nrow(dataset), replace = T, prob = c(.7, .15, .15))
training = dataset[r_assign == 1, ]
validation = dataset[r_assign == 2, ]
testing = dataset[r_assign == 3, ]

###### Dimensionality Reduction ######
## 1) Pricipal Component Analysis : Unsupervised
library(e1071)
pca = prcomp(x = training[, -ncol(training)], center = T, scale. = T)

library(caret)
pca2 = preProcess(x = training[, -ncol(training)], method = 'pca', pcaComp = 2)
summary(pca)

training_pca = predict(pca, training)
training_pca = training_pca[, c(2, 3, 1)]
testing_pca = predict(pca, testing)
testing_pca = testing_pca[, c(2, 3, 1)]

plot(pca, type = 'l')

pr.var = pca$sdev^2
pve = pr.var / sum(pr.var)

plot(pve, ylim = c(0, 1), type = 'b')   #scree plot
plot(cumsum(pve), ylim = c(0, 1), type = 'b')

biplot(pca)
plot(pca$x[, c(1,2)], col = diagonosis + 1)

df_pca = pca$x %>% as.data.frame()
ggplot(df_pca, aes(x = PC1, y = PC2, col = dataset$diagnosis)) +
  geom_point() +
  theme_classic()

library(ggord)
ggord(pca, grp_in = dataset$diagnosis)


## 2) Linear Discriminant Analysis : Supervised
library(MASS)
lda = lda(Customer_Segment ~., data = training)

training_lda = predict(lda, training) %>% as.data.frame()
training_lda = training_lda[, c(5, 6, 1)]
testing_lda = predict(lda, testing) %>% as.data.frame()
testing_lda = testing_lda[, c(5, 6, 1)]

## 3) Kernel PCA (Non-liear data case)
library(kernlab)
k_pca = kpca(~., data = training[, 3:4], kernel = 'rbfdot', features = 2)

training_k_pca = predict(k_pca, training) %>% as.data.frame()
training_k_pca$Purchased = training$Purchased
testing_k_pca = predict(k_pca, testing) %>% as.data.frame()
testing_k_pca$Purchased = testing$Purchased


#####################################################
###### Regression : Predicting a continuous number
###### 1) Linear Regression
fit = lm(Profit ~ . , training)
summary(fit)
glance(fit)

coef(fit)
resid(fit)
fitted.values(fit)

# influencePlot(fit, id.method = 'identify')
# plot(fit, which = 4, cooks.level = cutoff)

## K-fold Cross Validaion
library(caret)
trC = trainControl(method = 'cv', number = 10)
fit = train(Profit~. , dataset, trControl = trC,
            method = 'lm', metric = 'RMSE')

myFold = createFolds(y = dataset$Profit, k = 5)
myControl = trainControl(method = 'cv', number = 10, verboseIter = F,
                         preProcess = c('zv', 'medianImpute', 'center', 'scale', 'pca'),
                         summaryFunction = twoClassSummary, classProbs = T, index = myFold)

myGrid = expand.grid(alpha = 0:1, lambda = 10^seq(-1, 1, length = 50))
fit2 = train(Profit~. , dataset, trControl = myControl,
             method = 'glmnet', tuneGrid = myGrid, metric = 'ROC')


## Predicting the model
pred = predict(fit, testing)
actual = testing$price
cor.test(pred, actual)
library(Metrics)
rmse(pred, actual)

###### Conditions of Linear Regression Model ######
library(gvlma)
gvlma(fit)
## 1. Linear relation
mean(fit$residuals)

## 2. Normality of Residuals
## 3. Constant Variances of Residuals
par(mfrow = c(2,2))
plot(fit)

ncvTest(fit)
spreadLevelPlot(fit)
# ->Data scale Transformations

## 4. Independence of Residuals
library(lmtest)
dwtest(fit)

## 5. Multicollinearity
library(car)
sqrt(vif(fit))

num_df = dataset[sapply(dataset, is.numeric)]
num_df = num_df[ , !names(num_df) %in% c('Profit')]
cor_df = cor(num_df)
library(corrplot)
corrplot(cor_df)
library(caret)
high_cor = findCorrelation(cor_df, cutoff = 0.75)
names(num_df[, c(high_cor)])

###### Data Scale Transformation ######


###### Variable Selection ######
## 1. stepwise selection
step(object = fit, direction = 'backward')

null = lm(Profit ~ 1, dataset)
full = lm(Profit ~ ., dataset)
step(object = null, scope = list(lower = null, upper = full), direction = 'both')

## 2. regsubsets method
library(leaps)
regfit = regsubsets(Profit ~., dataset)
summary(regfit)
plot(regfit, scale = 'adjr2', main = 'Adjusted R2')
sum_regfit = summary(regfit)
sum_regfit$rsq

## 3. the Contribution/Importance of Predictors
varImp(fit, scale = F)

library(relaimpo)
calc.relimp(fit)

library(hier.part)
x = dataset[, !names %in% c('Profit')]
h = hier.part(dataset$Profit, x, family = 'gaussian', gof = 'Rsqu' )

## Others...
# library(FSelector)
# linear.correlation(Profit~., dataset)
# cfs(Profit~., dataset)
# chi.squared(Profit~., dataset)
# information.gain(Profit~., dataset)
# library(Boruta)
# Boruta(x = dataset, y = dataset$Profit, maxRuns = 100, doTrace = 0)

## 4. Outliers
library(car)
outlierTest(fit)
library(robustbase)
lts_fit = ltsReg(Profit ~. , dataset)

###### 2) Polynomial Regression ######
x = dataset$R.D.Spend
fit = lm(Profit ~ x + I(x^2) + I(x^3), training)
library(stats)
AIC(fit2)

fit2 = lm(Profit ~ poly(x, 3), training)
AIC(fit2a)

library(mgcv)
fit_gam = gam(Profit ~ s(x, k = 50) + s(y), training, sp = 0.001, method = 'REML')

gam.check(fit_gam)
concurvity(b = fit_gam, full = T)
pred = predict(fit_gam, testing) %>% as.numeric()


plot(fit_gam, page = 1)
plot(fit_gam, residuals = T)
plot(fit_gam, select = 2, shade = TRUE, shade.col = "hotpink",
     shift = coef(mod)[1], seWithMean = TRUE)

###### 3) Logistic Regression ######
## 1. Binary Case
fit = glm(Purchased~., family = binomial(link = 'logit'),
              data = training)
summary(fit)
exp(coef(fit))  #odds ratio

pred.prob = predict(fit, testing, type = 'response') %>% as.data.frame()
pred.y = ifelse(pred.prob>= 0.5, 1, 0)
pred.y = round(pred.prob)

actualData = testing$Purchased
table(actualData, pred.y)
mean(actualData == pred.y)  #Accuracy

library(broom)
augment(fit, type.predict = 'response')

library(ResourceSelection)
hoslem.test(actualData, pred.y)

# predict(fit_glm, newdata = data.frame(a= , b= ), type = 'response')

## 2. Multinomial Case
library(nnet)
fit = multinom(Type~., data = training, maxit = 500, trace = T)
pred.prop = predict(fit, testing, type = 'probs')
pred.y = predict(fit, testing, type = 'class')

postResample(testing$Type, pred.y)


#####################################################
###### Classification : Predictting a category
###### 1) Support Vector Machine ######
## Support Vector Regression
library(e1071)
fit = svm(Purchased~. , training, type = 'eps-regression')

## SVM
classifier = svm(Purchased~. , training[3:5], type = 'C-classification',
                 kernel = 'linear')

pred = predict(classifier, newdata = testing[, -5])
pred
actualData = testing$Purchased
table(actualData, pred)

## Kernel SVM
# install.packages("kernlab")
classifier = svm(Purchased~. , training[3:5], type = 'C-classification',
                 kernel = 'radial')

library(caret)
classifier = train(Purchased~. , training[3:5], method = 'svmRadial')
classifier$bestTune

###### 2) Naive Bayes Classification ######
library(e1071)
classifier = naiveBayes(x = training[,3:4], y = factor(training$Purchased), laplace = 1)
classifier

pred = predict(classifier, testing[, -5], type = 'probs')
pred
actualData = testing$Purchased
table(actualData, pred)


###### 3) Decision Tree ######
library(rpart)
fit = rpart(Purchased~., training, method = 'anova')

classifier = rpart(Purchased ~., training, method = 'class', parms = list(split = 'gini'))
classifier2 = rpart(Purchased~., training, control = rpart.control(maxdepth = 5, minsplit = 20))

pred = predict(classifier, newdata = testing[, -5], type = 'class')

library(rpart.plot)   # visualization
rpart.plot(classifier, type = 3, fallen.leaves = T)


## pruning
plotcp(classifier)
rsq.rpart(classifier)

classifier$cptable
opt_index = which.min(classifier$cptable[, 'xerror'])
cp_opt = classifier$cptable[opt_index, 'CP']

classifier_pruned = prune(classifier, cp = cp_opt)


## grid search
hyper_grid = expand.grid(min_split = 1:4, max_depth = 1:6)

grade_models = list()
rmse_values = c()
for(i in 1:nrow(hyper_grid)){
  grade_models[[i]] = rpart(formula = final_grade~.,
                            data = grade_train,
                            method = 'anova',
                            minsplit = hyper_grid$min_split[i],
                            maxdepth = hyper_grid$max_depth[i])

  pred = predict(object = grade_models[[i]], newdata = grade_valid)

  rmse_values[i] = rmse(actual = grade_valid$final_grade,
                        predicted = pred)
}
which.min(rmse_values)
best_model = grade_models[[which.min(rmse_values)]]

## predict
best_model$control
pred = predict(best_model, grade_test)
rmse(actual = grade_test$final_grade,
     predicted = pred)


###### 4) Random Forest ######
library(randomForest)
classifier = randomForest(x = training, y = training$Purchased)

## the error rate of 'Out-Of-Bag' (built-in validation set)
classifier$importance
(err = classifier$err.rate) %>% head()
err[nrow(err), ]        # -> final oob err.rate : outcome of rf
classifier

plot(classifier)         # -> how many trees do I need by oob err-rate
legend(x = 'right', legend = colnames(err), fill = 1:nrow(err))

## grid search
# randomForest(mtry = , maxnodes = , nodesize = , sampsize = , formula = default~., data = training)
hyper_grid = expand.grid(m_try = seq(4, ncol(training), 2),
                         node_size = seq(3, 8, 2),
                         samp_size = nrow(training)*c(.7, .8))

models = list()
for(i in 1:nrow(hyper_grid)){
  models[[i]] = randomForest(default~., training,
                             mtry = hyper_gird$m_try[i],
                             nodesize = hyper_grid$node_size[i],
                             sampsize = hyper_grid$samp_size[i])

  err = models[[i]]$err.rate
  hyper_grid$oob_err[i] = err[nrow(err), 'OOB']
}
head(hyper_grid)
hyper_grid[which.min(hyper_grid$oob_err), ]
(opt_model = models[[which.min(hyper_grid$oob_err)]])

## predict
pred = predict(classifier, testing[, -5])
actualData = testing$Purchased
table(actualData, pred)
mean(actualData == pred)

varImp(classifier)
importance(classifier)


###### 5) K-Nearest Neighbor Classification : Supervised ######
library(class)
pred.y = knn(train = training[, 3:4], test = testing[, 3:4],
             cl = training$Purchased, k = 5)


pred.prob = knn(training[, 3:4], test = testing[, 3:4], cl = training$purchased,
                k = 5, prob = T)
str(pred.prob)
attr(pred.prob, "prob")

#####################################################
###### Clustering
###### 1) K-Means Clustering : Unsupervised ######
## 1-1. Finding the Optimal Number of Clusters
set.seed(20)
wcss = vector()
for (i in 1:15){
  wcss[i] = kmeans(x = dataset[, -5], centers = i)$tot.withinss
}                           #total within-cluster sum of squares
plot(1:15, wcss, type = 'b', pch = 19, frame.plot = F,
     xlab = 'Number of Clusters', ylab = 'WCSS')

library(factoextra)
fviz_nbclust(x = dataset[, -5], FUNcluster = hcut, method = 'wss')

library(cluster)
pam(dataset[, -5], k = 3)

library(NbClust)
nb = NbClust(data = dataset[, -5], method = 'kmeans')
hist(nb$Best.nc[1,], breaks = 15)

## 2. Clustering
set.seed(25)
clu = kmeans(x = dataset[, -5], centers = 3, iter.max = 300, nstart = 10)
clu$cluster

## 3. Visualizing
ggplot(dataset, aes(x = Petal.Length, y = Petal.Width)) +
  geom_jitter(col = clu$cluster, width = .5, alpha = .3)

library(cluster)
clusplot(x = dataset[, -5], clu$cluster,
         lines = 0, shade = T, color = T, labels = 2,
         main = 'Cluster Plot')

library(factoextra)
fviz_cluster(clu, data = dataset[, -5],
             geom = 'point', frame.type = 'norm', main = 'Cluster Plot')


###### 2) Hierarchical Clustering : Unsupervised ######
## 1. Finding the Optimal Number of Clusters : Dendrogram
x = scale(dataset[, 4:5])
d = dist(x, method = 'euclidean')
hc = hclust(d, method = 'complete')
hc2 = hclust(d, method = 'ward.D')

pltree()
plot(x = hc, cex = 0.5, hang = -1, main = 'Dendrogram')
rect.hclust(tree = hc, k = 4)

library(dendextend)
dend_colored = as.dendrogram(hc) %>%
  color_branches(h = 20) %>%
  color_labels(h = 20)
plot(dend_colored)


## 2. Clustering
hc_sub = cutree(tree = hc, k = 4)
table(hc_sub)
dataset$cluster = hc_sub

## 3. Visualizing
fviz_cluster(list(data=x, cluster=hc_sub),
             geom = 'point', frame.type = 'norm', main = 'Cluster Plot')

###### Hierarchical Clustering- Aggolomerative ######
x = scale(dataset[, 4:5])
hc.agg = agnes(x = x, method = 'complete')
hc.agg2 = agnes(x, method = 'ward')

hc.agg$ac
hc.agg2$ac

pltree(x = hc.agg2, cex = 0.5, hang = -1, main = 'Dendrogram')
rect.hclust(tree = hc.agg2, k = 5)

###### Hierarchical Clustering- Divisive ######
x = scale(dataset[, 4:5])
hc.div = diana(x)
hc.div$dc

pltree(x = hc.div, cex = 0.5, hang = -1, main = 'Dendrogram')
rect.hclust(tree = hc.div, k = 5)


#####################################################
###### Model Evaluation ######
library(caTools)
colAUC(predicted_prob, actual, plotROC = TRUE)

library(pROC)
ROC = roc(actual, pred)
plot(ROC, col = 'blue')
AUC(ROC)


preds_list = list(pred_glm, pred_rf, pred_svm2, pred_xg)

m = length(preds_list)
actuals_list = rep(list(testing$default), m)

library(ROCR)
pred = prediction(labels = actuals_list, predictions = preds_list)
rocs = performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright",
       legend = c("Logisitc", "Random Forest", "SVM", "Xgboost"),
       fill = 1:m)


###### Gradient Boosting ######
## Encoding the categorical variables as indicator(dummy),
## Splitting the dataset (No scaling)
library(gbm)
model_gbm = gbm(formula = default ~., distribution = 'bernoulli',
                 data = training, n.trees = 10000, cv.folds = 10)

names(training)
summary(model_gbm)    # -> variance importance

pred = predict(object = model_gbm, newdata = testing, n.trees = 10000, type = 'response')

## Optimazing ntree based on OOB / CV
(opt_ntree_oob = gbm.perf(object = model_gbm, method = 'OOB'))
(opt_ntree_cv = gbm.perf(object = model_gbm, method = 'cv'))

pred_oob = predict(object = model_gbm, newdata = testing, n.trees = opt_ntree_oob, type = 'response')
pred_cv = predict(object = model_gbm, newdata = testing, n.trees = opt_ntree_cv, type = 'response')

roc_oob = roc(response = testing$default, predictor = pred_oob)
roc_cv = roc(response = testing$default, predictor = pred_cv)
auc(roc_oob)
auc(roc_cv)

###### XGBoost ######
library(xgboost)
full_df2 = full_df %>%
  mutate_if(is.factor, as.integer)
str(full_df2)

part = 1:903653
tr_xgb = full_df2[part, ]
te_xgb = full_df2[-part, ]

val_xgb = tr_xgb[ind == 2, ]
tr_xgb = tr_xgb[ind == 1, ]

dtr = xgb.DMatrix(data = data.matrix(tr_xgb[, -1]), label = tr_xgb$logRevenue)
dval = xgb.DMatrix(data = data.matrix(val_xgb[, -1]), label = val_xgb$logRevenue)
dte = xgb.DMatrix(data = data.matrix(te_xgb[, -1]), label = te_xgb$logRevenue)

# training a xgb model
myParam = list(objective = 'reg:linear',
               eval_metric = 'mae',
               eta = .025,
               max_depth = 8,
               min_child_weight = 10,
               subsample = .7,
               colsample_bytree = .5)

cv = xgb.cv(data = dtr,
            params = myParam,
            nrounds = 3000,
            nfold = 5,
            early_stopping_rounds = 100,
            maximize = F,
            print_every_n = 50)

a = cv$evaluation_log$test_mae_mean %>% which.min()
cv$evaluation_log[a]
cv$best_iteration

model_xgb = xgb.train(data = dtr,
                      params = myParam,
                      nrounds = cv$best_iteration,
                      watchlist = list(val = dval),
                      print_every_n = 50,
                      early_stopping_rounds = 100)

pred_xgb = predict(model_xgb, dval)

xgb.importance(feature_names = names(tr_xgb), model = model_xgb) %>% xgb.plot.importance(top_n = 15)
