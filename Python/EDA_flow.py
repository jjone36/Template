import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 0. Understand the data (Please don't skip it)

# Step 1. Import the dataset
tr = pd.read_csv('train.csv')
te = pd.read_csv('test.csv')

y = tr.target
tr = tr.drop('target', axis = 1)

# Dimension check (duplication)
print("Train shape is : ", tr.shape)
print("Test shape is : ", te.shape)

# Data overview
tr.head()
print ("Data types : \n" , tr.info()
print ("\nFeatures : \n" , tr.columns.tolist())
print ("\nUnique values :  \n", tr.nunique())
print ("\nMissing values :  ", tr.isnull().sum())
tr.fillna('NaN', inplace = True)

# Concat train and test set
a = len[tr]
tr_te = pd.concat([tr, te], axis = 0)


# Step 2. Data overview
# Separte categorical & numerical features
cat_feats = tr_te.column[tr_te.dtypes == 'object']
num_feats = tr_te.column[tr_te.dtypes != 'object']

print("Categorical variables ", len(cat_feats), cat_feats))
print("Numeric variables ", len(num_feats), num_feats)

# Distributions of categorical features
fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize = (20, 20))   # change the nrows, ncols accordingly
for i in range(0, 16):
    ax = axes[i//4, i%4]
    sns.countplot(x = cat_feats[i], data = df, hue = df.Churn, dodge = False, ax = ax)
    plt.subplots_adjust(wspace = .5, hspace = .5)
    plt.title(str(cat_feats[i]))

# Distributions of numeric features
fig, axes = plt.subplots(nrows = 3, figsize = (10, 5))    # change the nrows, ncols accordingly
for i in range(0, 3):
    ax = axes[i]
    sns.kdeplot(df[num_feats[i]], shade = 'b', ax = ax)
    plt.subplots_adjust(wspace = .5, hspace = .5)

# Reorganzie mis-classified features if it's neccesssary


# Step 3. Preprocessing
# Categorical features
from sklearn.preprocesing import LabelEncoder
encoder = LabelEncoder()

for i in cat_feats:
    tr_te[i] = encoder.fit_transform(tr_te[i])

tr_te[cat_feats].T.drop_duplicates()

# Numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
tr_te[num_feats] = scaler.fit_transform(tr_te[num_feats])

# Constant variable check
n_unique = tr_te.nunique(dropna = False)
n_unique.sort_values()
const_feats = feats_counts.loc[n_unique == 1].index.tolist()
print("Constant features are : ", const_feats)
tr_te = tr_te.drop(const_feats, axis = 1)

# Time features
import datatime as dt



# Investigate features with lots of values
plt.hist(n_unique.astype('float')/tr_te.shape[0], bins = 50)

mask = (n_unique.astype('float')/tr_te.shape[0] < .8)
tr_te.loc[mask].head(20)


# Duplication check with cols

# Duplication check with rows

# Shuffle check


# Step 3. Visualization



# Step 4. Feature engineering



# Step 5. Split into train, valid, test
X_train = tr_te[:a, :]
X_test = tr_te[a:, :]
del tr_te

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size = .2, random_state = 36)


# Step 6. Modeling



# Step 7. Evaluation
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
conf_matrix = confusion_matrix(actual, pred)
print ("Classification report : \n", classification_report(actual, pred))
print ("Accuracy : \n", accuracy_score(actual, pred))
print ("AUC : ", roc_auc_score(actual, pred), "\n")

fpr,tpr,thresholds = roc_curve(actual, pred[:,1])
