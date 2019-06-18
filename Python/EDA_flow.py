import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pandas_profiling
pandas_profiling.ProfileReport(df)

# Step 0. Understand the data (Please don't skip it)



# Step 1. Import the dataset
tr = pd.read_csv('train.csv')
te = pd.read_csv('test.csv')

tr.head()

# Data overview
print("Data types : \n" , tr.info())
print("\nUnique values :  \n", tr.nunique())
print("\nMissing values :  ", tr.isnull().sum())
tr.fillna('NaN', inplace = True)

# Concat train and test set
cut = len(tr)
tr_te = pd.concat([tr, te], axis = 0)

# Separate categorical & numerical features
cat_feats = tr_te.column[tr_te.dtypes == 'object']
num_feats = tr_te.column[tr_te.dtypes != 'object']

print("Categorical variables ", len(cat_feats), cat_feats)
print("Numeric variables ", len(num_feats), num_feats)

# Reorganzie mis-classified features if it's neccesssary

# Drop the unnecessary features
drop_feats = []
tr_te = tr_te.drop(drop_feats, axis = 1)




# Step 2. Preprocessing
# Separate categorical & numerical features
df_cat = tr_te[cat_feats]
df_num = tr_te[num_feats[:-1]]

# 2-1. Numerical features
# Distributions of numeric features
fig, axes = plt.subplots(nrows = 3, figsize = (10, 5))    # change the nrows, ncols accordingly
for i in range(0, 3):
    ax = axes[i]
    sns.kdeplot(df[num_feats[i]], shade = 'b', ax = ax)
    plt.subplots_adjust(wspace = .5, hspace = .5)
tr_te[num_feats] = scaler.fit_transform(tr_te[num_feats])

# Constant variable check
n_unique = tr_te.nunique(dropna = False)
n_unique.sort_values()
const_feats = feats_counts.loc[n_unique == 1].index.tolist()
print("Constant features are : ", const_feats)
tr_te = tr_te.drop(const_feats, axis = 1)

n_unique = tr_te.nunique(dropna = False)
plt.hist(n_unique.astype(flaot) / tr_te.shape[0], bins = 100)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# 2-2. Categorical features
from sklearn.preprocesing import LabelEncoder
encoder = LabelEncoder()

for i in cat_feats:
    tr_te[i] = encoder.fit_transform(tr_te[i])
    tr_te[cat_feats].T.drop_duplicates()


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
df_oh = mlb.fit_transform(tr['col']).astype('int')
df_oh = pd.DataFrame(X, columns = mlb.classes_)


# Time features
import datatime as dt


# Duplication check with cols
# Duplication check with rows

# Shuffle check
# https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html





# Step 3. Visualization
# Investigate features with lots of values
plt.hist(n_unique.astype('float')/tr_te.shape[0], bins = 50)

mask = (n_unique.astype('float')/tr_te.shape[0] < .8)
tr_te.loc[mask].head(20)


# covariance matrix plot (heatmap)
sns.heatmap(df.corr())



# Step 4. Feature engineering
# Mean encoding
tr_y = pd.concat([tr, y], axis = 1)
means_map = tr_y.groupby(col).target.mean()
tr[col + '_mean_target'] = tr[col].map(means)




# Step 5. Split into train, valid, test
X_train = tr_te[:a, :]
X_test = tr_te[a:, :]
del tr_te

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size = .2, random_state = 36)


# Step 6. Modeling
# Baseline





# Step 7. Evaluation
# Regression


# Classification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
conf_matrix = confusion_matrix(actual, pred)
print ("Classification report : \n", classification_report(actual, pred))
print ("Accuracy : \n", accuracy_score(actual, pred))
print ("AUC : ", roc_auc_score(actual, pred), "\n")

fpr, tpr, thres = roc_curve(actual, pred[:,1])

def plot_roc(fpr, tpr):
    plt.plot(x = fpr, y = tpr, linewidth = 2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

plot_roc(fpr, tpr)
plt.show()
