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
imputer = Imputer(missing_values = 'NaN', strategy = 'median')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
y = labelencoder.fit_transform(y)

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Avoid the dummy variable trap


# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_tsst)

############# Dimensionality Reduction #############
# Pricipal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

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
accuracies = cross_val_score(estimator = reg, X = X_train, y = y_train, cv = 10)
accuracies.mean()    # mean accuracy value of validation sets
accuracies.std()


############# 2) Regularized Regression #############
## Ridge
from sklearn.linear_model import Ridge
ridge = Ridge(alpha, normalize = True)
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


## Lasso   (-> feature selection)
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
reg_poly = PolynomialFeatures(degree = 2)
X_poly = reg_poly.fit_transform(X_train)

reg_poly_2 = LinearRegression()
reg_poly_2.fit(X_poly, y_train)
y_pred = reg_poly_2.predict(X_test)

plt.scatter(X_test, y_test, 'red')
plt.plot(X_test, y_pred, 'blue')
plt.show()

############# 3) XGBoost #############
'https://xgboost.readthedocs.io/en/latest/build.html#'
from xgboost import XGBClassifier
clas = XGBClassifier(max_depth = 5, learning_rate = .05, n_estimators = 300)
clas.fit(X_train, y_train)
y_pred = clas.predict(X_test)


#################################################
###### Classification : Predictting a category
############# 1) Logisitc Regression #############
from sklearn.linear_model import LogisticRegression
clas = LogisticRegression()
clas.fit(X_train, y_train)
y_pred = clas.pred(X_test)

# Make the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

y_pred_prob = clas.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)


############# 2) Support Vector Machine #############
from sklearn.svm import SVR
reg = SVR(kernel = 'rbf')     # non-linear SVM
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)


from sklearn.svm import SVC
clas = SVC(kernel = 'linear')
clas2 = SVC(kernel = 'rbf')        # kernel SVM

clas.fit(X_train, y_train)
# Predict and print the label for the new data point X_new
y_pred = clas.predict(X_test)


############# 3) Naive Bayes Classification #############
from sklearn.naive_bayes import GaussianNB
clas = GaussianNB()
clas.fit(X_train, y_train)


############# 4) Decision Tree #############
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)


from sklearn.tree import DecisionTreeClassifier
clas = DecisionTreeClassifier(criterion = 'entropy')
clas.fit(X_train, y_train)


############# 5) Random Forest #############
from sklearn.tree import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 100, max_depth = 5)
reg.fit(X_train, y_train)


from sklearn.ensemble import RandomForestClassifier
clas = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 5)
clas.fit(X_train, y_train)


############# 6) K-Nearest Neighbor #############
(from sklearn.neighbors import KNeighborClassifier
from sklearn.model_selection import train_test_split

# Create a k-NN classifier with 6 neighbors
clas = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
# Fit the classifier to the data
clas.fit(X_train, y_train)
# Predict and print the label for the new data point X_new
y_pred = clas.predict(X_test)
# Make the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


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

plt.plot(range(1:11), wcss)
plt.title('The Elbow Method')
plt.xlabel('The Number of clusters')
plt.ylabel('WCSS')
plt.show()

# clustering
clus = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10)
y_clus = clus.fit_predict(X)


############# 2) Hierarchical Clustering #############
from scipy.cluster.hierarchy as sch
# Dendrogam for the optimal number of clusters
dendogram = sch.dendrogram(sch.Linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Clusters')
plt.ylabel('Euclidean distances')
plt.show()

# Aggolomerative clustering
from sklearn.cluster import AgglomerativeClustering
clus = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_clus = clus.fit_predict(X)


#################################################
###### Association Rule Learning
############# Apriori #############







#################################################
############# Evaluation Metrics #############
# R2
reg.score(X_test, y_test)
# RMSE
np.sqrt(mean_squared_error(y_test, y_pred))
# confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)
# ROC
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)




############# Grid Search #############
from sklearn.model_selection import GridSearchCV
myParam = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}
           {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.05, 0.01]}]
myGrid = GridSearchCV(estimator = clas,
                      param_grid = myParam,
                      scoring = 'accuracy',
                      '''n_jobs = -1,'''
                      cv = 10)
myGrid = myGrid.fit(X_train, y_train)
best_score = myGrid.best_score_
best_param = myGrid.best_params_
print(best_param)



############# Variable Selection functions #############
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
