############# Pipeline #############
from sklearn.pipeline import Pipeline
steps = [('imputation', imp), ('logistic_regression', logreg)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
pipe.score(X_test, y_test)


############# 1. scale + knn
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
# Create the pipeline: pipeline
pipe = Pipeline(steps)
# Specify the hyperparameter space
myParam = {'knn__n_neighbors' = np.arange(1, 50)}              # 'step name'__'parameter name'
# Create the GridSearchCV object: cv
cv = GridSearchCV(pipe, param_grid = myParam, cv = 5)
# Fit to the training set
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)


############# 2. scale + SVM
steps = [('scaler', StandardScaler()), ('SVM', SVC())]
pipe = Pipeline(steps)
myParam = {'SVM__C':[1, 10, 100], 'SVM__gamma':[0, 1, 0.01]}
cv = GridSearchCV(pipe, param_grid = myParam, cv = 3)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)


############# 3. Impute + scale + elasticnet
steps = [('imputation', Imputer(missing_values= 'NaN', strategy= 'mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]
pipeline = Pipeline(steps)
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}
gm_cv = GridSearchCV(pipeline, parameters)
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
pl = Pipeline([
        ('union', process_and_join_features),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']],
                                                    pd.get_dummies(sample_df['label']),
                                                    random_state=22)
pl.fit(X_train, y_train)
