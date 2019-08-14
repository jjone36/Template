'https://scikit-learn.org/stable/index.html'
'https://campus.datacamp.com/courses/machine-learning-with-the-experts-school-budgets'
#################################################
############# Data Preprocessing #############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Impute values into NA
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)     # 'most_frequent'
imp = imp.fit(X[:, 1:3])
X[:, 1:3] = imp.transform(X[:, 1:3])
print(X)

###### Encoding
category_mask = df.dtypes == object
df_category = df.columns[category_mask].tolist()
# Encode categorical data
X = pd.get_dummies(X)
X = X.drop('columnName', axis = 1)
X = pd.get_dummies(X, drop_first = True)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
y = labelencoder.fit_transform(y)

# encode integer columns as dummies(One-Hot-Encode)
onehotencoder = OneHotEncoder(categorical_features = [0], sparse = False)
X = onehotencoder.fit_transform(X).toarray()

# -> encoding & OneHotEncoder in one go
from sklearn.feature_extraction import DictVectorizer
# Convert df into a dictionary: df_dict
df_dict = df.to_dict('records')
# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse = False)
print(dv.vocabulary_)
# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

###### Data Preprocessing
from sklearn.preprocessing import scale
X_scaled = scale(X)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train)
X_test_scaled = scaler.transform(X_tsst)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
mm_scaler = MinMaxScaler()
X_train_scaled = mm_scaler.fit_transform(X_train)
X_test_scaled = mm_scaler.transform(X_test)

from sklearn.preprocessing import normalizer
normalizer = normalizer(X)

# feature engineering
df.time = pd.to_datetime(df.time)
df['mnoth'] = df.time.apply(lambda row: row.month)

# MultiLabelBinarizer
# https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df.genres).astype('int')
mlb.classes_
df_genres = pd.DataFrame(X, columns = mlb.classes_)

# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

###### Dimensionality Reduction
# Pricipal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
pca.components_
pca.n_components_
pca.mean_
pca.explained_variance_ratio_     # explained-variance ratio

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

# Kernel PCA (Non-liear data case)
from sklearn.decomposition import kernelPCA
kpca = kernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)    # LDA: supervised model
X_test = lda.transform(X_test)

# t-SNE => 2D
from sklearn.manifold import TSNE
model = TSNE(learning_rate = 200)
tsne_features = model.fit_transform(samples)
x = tsne_features[:, 0]
y = tsne_features[:, 1]
plt.scatter(x, y, c = label)
plt.show()

#################################################
###### Regression : Predicting a continuous number
############# 1) Linear Regression #############
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict the model
y_pred = reg.predict(X_test)
reg.score(X_test, y_test)

plt.scatter(y_train, y_pred, color = 'red')
plt.show()
reg.predict(new_value = 6.5)

###### K-fold Cross Validaion
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(estimator = reg, X, y, cv = 5, scoring = 'roc_auc')
print(cv_score)
cv_score.mean()    # mean accuracy value of validation sets
cv_score.std()

###### Stratified Cross Validation
from sklearn.model_selection import StratifiedKFold
skfolds = StratifiedKfold(n_splits = 5, random_state = 42)

for tr_idx, te_idx in skfolds.split(X_train, y_train):
    X_train_fold = X_train[tr_idx]
    X_test_fold = X_train[te_idx]
    y_train_fold = y_train[tr_idx]
    y_test_fold = y_train[te_idx]

    model.fit(X_train_fold, y_train_fold)
    pred_fold = model.predict(X_test_fold)
    print("Accuracy score: ", mean(pred_fold == y_test_fold))

############# 2) Regularized Regression #############
###### Elastic-Net
from sklearn.linear_model import ElasticNet
elas = ElasticNet()
myParam = {'l1_ratio': np.linspace(0, 1, 30)}

gm_cv = GridSearchCV(elas, myParam, cv = 5)
gm_cv.fit(X_train, y_train)
y_pred = gm_cv.predict(X_test)

gm_cv.best_score_
gm_cv.best_params_
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))

r2 = gm_cv.score(X_test, y_test)
rmse = gm_cv.mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))

###### Ridge
from sklearn.linear_model import Ridge
ridge = Ridge(alpha = .3, normalize = True)
ridge.fit(X_train, y_train)
ridge.predict(X_test)

# alpha tuning
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []
# Create a ridge regressor: ridge
ridge = Ridge(normalize = True)
# Compute scores over range of alphas
for alpha in alpha_space:
    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv = 10)
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))
# Display the plot
display_plot(ridge_scores, ridge_scores_std)


###### Lasso
from sklearn.linear_model import Lasso
lasso = Lasso(alpha, normalize = True)
lasso.fit(X_train, y_train)
lasso.predict(X_test)
lasso_coef = lasso.coef_

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()


############# 2) Polynomial Regression #############
from sklearn.preprocessing import PolynomialFeatures
reg_poly = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)
X_poly = reg_poly.fit_transform(X_train)

reg_poly_2 = LinearRegression()
reg_poly_2.fit(X_poly, y_train)
y_pred = reg_poly_2.predict(X_test)

plt.scatter(X_test, y_test, 'red')
plt.plot(X_test, y_pred, 'blue')
plt.show()


#################################################
###### Classification : Predictting a category
############# 1) Logisitc Regression #############
from sklearn.linear_model import LogisticRegression
clas = LogisticRegression()
clas.fit(X_train, y_train)
y_pred = clas.predict(X_test)
y_pred_prob = clas.predict_proba(X_test)[:, 1]

# decisiton boundary
plot_decision_boundary(lambda x: clas.predict(x), X_test, y_test)
plt.title('Decision Boundary')

# StratifiedShuffleSplit??
# multiple class
from sklearn.multiclass import OneVsRestClassifier
clas = OneVsRestClassifier(LogisticRegression())

############# 2) Support Vector Machine #############
from sklearn.svm import LinearSVR
reg = LinearSVR(epsilon=1.)

from sklearn.svm import SVR
reg = SVR(kernel = 'rbf')
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

from sklearn.svm import LinearSVC
clas = LinearSVC(c = 1, losss = 'hinge')

from sklearn.svm import SVC
clas = SVC(kernel = 'rbf')

############# 3) Naive Bayes Classification #############
from sklearn.naive_bayes import GaussianNB
clas = GaussianNB()
clas.fit(X_train, y_train)

from sklearn.naive_bayes import MultinomialNB
clas = MultinomialNB()
clas.fit(X_train, y_train)

clas.classes_

############# 4) Decision Tree #############
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
clas = DecisionTreeClassifier(criterion = 'entropy',
                              max_depth = 4)
clas.fit(X_train, y_train)


############# 5) Random Forest #############
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 100,
                            criterion="mse",
                            min_samples_split= .7,
                            max_depth = 5)
reg.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
clas = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 5, n_jobs = -1)
clas.fit(X_train, y_train)

pred = []
for tree in clas.estimators_:
    pred.append(tree.predict_proba(X_val)[None, :])
pred = np.vstack(pred)
np.cumsum(pred, axis=0)/np.arange(1, pred.shape[0] + 1)[:, None, None]

############# 6) K-Nearest Neighbor #############
from sklearn.neighbors import KNeighborClassifier
clas = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)

clas.fit(X_train, y_train)
y_pred = clas.predict(X_test)

#################################################
###### Clustering
############# 1) K-Means Clustering #############
from sklearn.cluster import kMeans
# Elbow method for the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('The Number of clusters')
plt.ylabel('WCSS')
plt.show()

# clustering
clus = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10)
y_clus = clus.fit_predict(X)
print(y_clus)

centroids = clus.cluster_centers_

############# 2) Hierarchical Clustering #############
import scipy.cluster.hierarchy as sch
# Dendrogam for the optimal number of clusters
mergings = sch.Linkage(X_train_scaled, method = 'ward')     # dist()
dendogram = sch.dendrogram(mergings, labels, leaf_rotation = 90, leaf_font_size = 5)    # hclust()
plt.title('Dendrogram')
plt.xlabel('Clusters')
plt.ylabel('Euclidean distances')
plt.show()

labels = sch.fcluster(mergings, 15, criterion = 'distance')     # cutree()

# Aggolomerative clustering
from sklearn.cluster import AgglomerativeClustering
clus = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_clus = clus.fit_predict(X)


#################################################
###### Ensemble Learning
############# 1) XGBoost #############
'https://xgboost.readthedocs.io/en/latest/build.html#'
import xgboost as xgb
## classification
xg_claf = xgb.XGBClassifier(objective = 'binary:logistic',
                            max_depth = 5,
                            learning_rate = .05,
                            n_estimators = 300)
xg_claf.fit(X_train, y_train)
y_pred = xg_claf.predict(X_test)

## regression
xg_reg = xgb.XGBRegressor(objective = 'reg:linear',
                          booster = 'gbtree',
                          max_depth = 7,
                          early_stopping_rounds = 200,
                          learning_rate = .01,
                          gamma = 1.5,
                          #alpha, #lambda,
                          subsample = .7,
                          colsample_bytree = .7,
                          colsample_bylevel = .5
                          Silent = True)
xg_reg.fit(X_train, y_train)

## cross validation xgboost
DM_train = xgb.DMatrix(data= X_train, label= y_train)
DM_test = xgb.DMatrix(data= X_test, label= y_test)

# Create the parameter dictionary: params
params = {'objective' : 'reg:linear',
          'max_depth' : 7,
          'learning_rate' : 0.005,
          'early_stopping_rounds' : 500,
          'gamma' : 1.5,
          'subsample' : 0.8,
          'colsample_bytree' : 0.7,
          'colsample_bylevel' : 0.9,
          'silent' : True}

# Create list of additional parameters and empty list
reg_params = [1, 10, 100]
best_rmse = []
for reg in reg_params:
    # Update l2 strength
    myParam['lambda'] = reg
    cv_results = xgb.cv(dtrain= DM_X, params= params, nfold= 3,
                        num_boost_round= 100,
                        metrics="rmse",    # mae, rmse, auc, error
                        early_stopping_rounds = 100,
                        as_pandas= True)
     best_rmse.append(cv_results['test-rmse-mean'].tail().values[-1])
results = pd.DataFrame(list(zip(reg_params, best_rmse)), columns=["lambda", "best_rmse"])

## visualization
# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees = 100, rankdir = 'LR')
plt.show()

xgb.plot_importance(xg_reg)
plt.show()

############# 2) Catboost #############
from catboost import CatBoostClassifier
cat_idx
clas = CatBoostClassifier(iterations = 1000,
                          depth = 5,
                          learning_rate = 0.1,
                          colsample_bylevel = .7,
                          early_stopping_rounds = 200,
                          bagging_temperature = .2,
                          eval_metric = 'RMSE',
                          logging_level = 'Silent')

clas.fit(X_tr, y_tr, cat_features = cat_idx,
         eval_set = (X_val, y_val), plot = True)

from grader_v2 import Grader
grader = Grader()
mean = 0
grader.submit_tag('logloss_mean', mean)

############# 3) LightGBM #############
import lightgbm as lgb
tr_data = lgb.Dataset(X_tr, label = y_tr)

params = {'objective' : 'regression',
          'max_depth' : 7,
          #'num_leaves' : 30,
          'learning_rate': 0.01,
          'metric' : 'rmse',
          'min_data_in_leaf' : 100,
          'colsample_bytree': 0.7,
          'subsample_freq': 1,
          'lambda_l1' : 0.2,
          #'lambda_l2' : .3
          'subsample' : .7,
          #"bagging_seed" : 42,
          "verbose" : -1}

hist = {}
model_lg = lgb.train(params, tr_data,
                     num_iteration = 10000,
                     early_stopping_rounds = 200,
                     callbacks = [lgb.record_evaluation(hist)])
pred = model_lg.predict(X_te, num_iteration = model_lg.best_iteration)

#################################################
############# Evaluation #############
# Pearson correlation
from scipy.stats import pearsonr
correlation, pvalue = pearsonr(width, length)

# R2
r2 = reg.score(X_test, y_test)
print("R-squred score: %f" % (r2))

# RMSE
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test, y_pred))


# confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred, label = ['Spam', 'NoSpam'])
classification_report(y_test, y_pred)

# accuracy
from sklearn.metircs import accuracy_score, precision_score, recall_score, f1_score
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)     # How well the model detect the class
recall_score(y_test, y_pred)        # How trustful the prediction is
f1_score(y_test, y_pred)

# ROC
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thres = roc_curve(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)

def plot_roc(fpr, tpr):
    plt.plot(x = fpr, y = tpr, linewidth = 2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')


############# Grid Search #############
from sklearn.model_selection import GridSearchCV
# Logisitc
c_space = np.logspace(-5, 8, 15)
myParam = {'C': c_space, 'penalty': ['l1', 'l2']}
# SVM
myParam = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}
           {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.05, 0.01]}]
# decisiontree
myParam = {'max_depth': [3, None], 'max_features': randint(1, 9), 'criterion': ['gini', 'entropy']}
# knn
myParam = {'n_neighbors': np.arange(1, 50)}
# elastic net
myParam = {'l1_ratio': np.linspace(0, 1, 30)}
# xgboost
myParam = {'learning_rate': [0.01, 0.05, 0.1, 0.5],
           'n_estimators': [200],
           'subsample': [.3, .5, .7]}

# Grid Search
myParam = {}
cv = GridSearchCV(estimator = model,
                  param_grid = myParam,
                  scoring = 'neg_mean_squared_error',   # 'neg_mean_squared_log_error'
                  cv = 5,
                  verbose = 0)
cv = cv.fit(X_tr, y_tr)
print("=======================================")
print("Best score is {}".format(cv.best_score_))
print(cv.best_params_)

# Randomized Grid Search
from sklearn.model_selection import RandomizedSearchCV
cv = RandomizedSearchCV(estimator = clas,
                        param_distributions = myParam,
                        scoring = 'roc_auc',
                        cv = 5,
                        n_iter = 5,
                        verbose = 1)
print(cv.best_score_)
print(cv.best_params_)


#################################################
###### Model Save
import pickle
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

#################################################
############# Ensemble #############
# bagging - changing seed & averaging
n_bags = 10
seed = 1

bagged_pred = np.zeros(test.shape[0])
for i in range(n_bags):
    model.set_params(random_state = seed + i)
    model.fit(tr_X, y)
    preds = model.predict(te)
    bagged_pred += preds
bagged_pred /= n_bags

# stacking
# prediction on the valid set
valid_pred_1 = model_1.predict(valid)
valid_pred_2 = model_2.predict(valid)
stacked_valid = np.column_stack((valid_pred_1, valid_pred_2))

# prediction on the test set
te_pred_1 = model_1.predict(te)
te_pred_2 = model_2.predict(te)
stacked_te = np.column_stack((te_pred_1, te_pred_2))

# final prediction
meta_model.fit(stacked_valid, y_val)
final_pred = meta_model.predict(stacked_te)

############# Feature Selection  #############
# 1. Recursive Feature Elimination
from sklearn.feature_selection import RFECV
rfe = RFECV(estimator = LogisticRegression(), cv=5, scoring='accuracy')
rfe = rfe.fit(X_train, y_train)

# Select variables and calulate test accuracy
cols = X_train.columns[rfe.support_]
acc = accuracy_score(y_test, rfe.estimator_.predict(X_test[cols]))
print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))

# Plot number of features vs CV scores
plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# 2. Boruta Feature Selection
from boruta import BorutaPy
boruta = BorutaPy(model, n_estimators='auto', verbose=2)
boruta.fit(X_tr.values, y_tr.values.ravel())

# Select features and fit Logistic Regression
cols = X_tr.columns[boruta.support_]
est_boruta = LogisticRegression()
est_boruta.fit(X_tr[cols], y_tr)


# Building the optimal model using Backward Eliminations
import statsmodels.formula.api as sm

X = np.append(arr = np.ones(shape = (50, 1)).astype(int), vales = X)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]      # update
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
reg_OLS.summary()

# Automatic Backward Elimination with p-values only
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

# Automatic Backward Elimination with p-values & Adjusted R2
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
