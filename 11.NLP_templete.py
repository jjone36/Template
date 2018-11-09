#################################################
######### Data Preprocessing #########
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Reviews.tsv', delimiter = '\t', quoting = 3)   # ignoring double quotes

# Clean the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(1000):
    text = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)

# Create the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()     # making as a matrix
y = dataset.iloc[:, 'target'].values
