#################################################
############# Pipeline #############
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
############# 1. impute + glm
steps = [('imputation', imp), ('logistic_regression', logreg)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
pipe.score(X_test, y_test)

############# 2. scale + KMenas
steps = [('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters = 4))]
pipe = Pipeline(steps)
y_clus = pipe.fit_predict(X)

############# 3. scale + knn
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
# Create the pipeline: pipeline
pipe = Pipeline(steps)
# Specify the hyperparameter space
param_grid = {'knn__n_neighbors' = np.arange(1, 50)}              # 'step name'__'parameter name'
# Create the GridSearchCV object: cv
cv = GridSearchCV(pipe, param_grid, cv = 5)
# Fit to the training set
cv.fit(X_train, y_train)

y_pred = cv.predict(X_test)

############# 4. scale + SVM
steps = [('scaler', StandardScaler()), ('SVM', SVC())]
pipe = Pipeline(steps)
param_grid = {'SVM__C':[1, 10, 100], 'SVM__gamma':[0, 1, 0.01]}
cv = GridSearchCV(pipe, param_grid, cv = 3)
cv.fit(X_train, y_train)

y_pred = cv.predict(X_test)

############# 5. Impute + scale + elasticnet
steps = [('imputation', Imputer(missing_values= 'NaN', strategy= 'mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]
pipeline = Pipeline(steps)
param_grid = {'elasticnet__l1_ratio':np.linspace(0,1,30)}
gm_cv = GridSearchCV(pipeline, param_grid)
gm_cv.fit(X_train, y_train)
y_pred = gm_cv.predict(X_test)


#################################################
############# FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

# Obtain the text data: get_text_data
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)
just_text_data = get_text_data.fit_transform(sample_df)

# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)
just_numeric_data = get_numeric_data.fit_transform(sample_df)

############# FeatureUnion
from sklearn.pipeline import FeatureUnion
# Create a FeatureUnion with nested pipeline: process_and_join_features
process_and_join_features = FeatureUnion(transformer_list = [
                ('numeric_features', Pipeline([('selector', get_numeric_data),
                                               ('imputer', Imputer())])),
                ('text_features', Pipeline([('selector', get_text_data),
                                            ('vectorizer', CountVectorizer())]))
                                                            ])

# Instantiate nested pipeline: pl
pl = Pipeline([('union', process_and_join_features),
               ('clf', OneVsRestClassifier(LogisticRegression()))])

# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']],
                                                    pd.get_dummies(sample_df['label']),
                                                    random_state=22)
pl.fit(X_train, y_train)


#################################################
############# XGBoost
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

myParam = {'objective': 'reg:linear',
           'booster': 'gbtree',
           'max_depth': 3}
steps = [('ohe_onestep': DictVectorizer(sparse = False)),
         ('xgb': xgb.XGBRegressor(params = myParam))]
X_train_dict = X_train.to_dict('record')
pipe = Pipeline(steps)
pipe.fit(X_train_dict, y_train)

# K-fold cross validation
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(estimator = pipe, X_train_dict, y, cv = 5, scoring = 'neg_mean_squared_error')

# Grid searching
from sklearn.model_selection import GridSearchCV
steps = [('ohe_onestep': DictVectorizer(sparse = False)),
         ('xgb': xgb.XGBRegressor())]
pipe = Pipeline(steps)
param_grid = {'objective': 'reg:linear',
              'booster': 'gbtree',
              'xgb__max_depth': np.arange(3, 8, 1),
              'xgb__learning_rate':np.arange(0.05, 1, 0.05)}
cv = GridSearchCV(pipe, param_grid, cv = 5)
cv.fit(X_train_dict, y_train)
y_pred = cv.predict(X_test_dict)


# What We Have Not Covered (And How You Can Proceed)
# Using XGBoost for ranking/recommendation problems (Netflix/Amazon problem)
# Using more sophisticated hyperparameter tuning strategies for tuning XGBoost models (Bayesian Optimization)
# Using XGBoost as part of an ensemble of other models for regression/classification

#################################################
'https://campus.datacamp.com/courses/extreme-gradient-boosting-with-xgboost/using-xgboost-in-pipelines?ex=9'
############# sklearn_pandas
from sklearn_pandas import categoricalImputer
from sklearn_pandas import DataFrameMapper

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object
# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()
# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper([([numeric_feature], Imputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                            input_df = True, df_out = True)
# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper([(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
                                                input_df = True, df_out = True)

from sklearn.pipeline import FeatureUnion
# Create the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([("num_mapper", numeric_imputation_mapper),
                                          ("cat_mapper", categorical_imputation_mapper)])

# Combine all into pipeline
pipeline = Pipeline([("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort = False)),
                     ("clf", xgb.XGBClassifier())])
# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, kidney_data, y, scoring="roc_auc", cv= 3)
