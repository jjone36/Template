# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:23:43 2018

@author: jjone
"""

# https://github.com/jjone36/machine_learning_examples/tree/master/nlp_class

import pandas as pd

df = pd.read_csv('spam.csv', encoding = 'ISO-8859-1')

df.columns
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

df.columns = ['labels', 'text']
df.describe
df['b_labels'] = df['labels'].map({'ham':0, 'spam':1})

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(decode_error = 'ignore')
X = cv.fit_transform(df['text'])

y = df['b_labels']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

from sklearn.naive_bayes import MultinomialNB
clas = MultinomialNB()
clas.fit(X_train, y_train)
y_pred = clas.predict(X_test)

print("train score:", clas.score(X_train, y_train))
print("test score:", clas.score(X_test, y_test))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

############# wordcloud funcion
def plot_WordCloud(x):
    words = ''
    for word in df[df['label']== x]['text']:
        word = word.lower()
        words += word + ' '

    wordcloud = WordCloud(width = 500, height = 500).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

plot_WordCloud('spam')
plot_WordCloud('ham')

#####################################################################
import pandas as pd

df = pd.read_csv('Reviews.csv')
df.info()
df = df.iloc[:100]   # just for pratice!

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

##########################
corpus = []
for i in range(len(df)):
    text = re.sub('[^a-zA-Z]', ' ', df['Text'][i])
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]    # stemming
    tokens = [t for t in tokens if t not in set(stopwords.words('english'))]
    tokens = ' '.join(tokens)
    corpus.append(tokens)

##########################
def my_tokenizer(x):
    text = re.sub('[^a-zA-Z]', ' ', x)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]    # stemming
    tokens = [t for t in tokens if t not in set(stopwords.words('english'))]
    return tokens

word_index_map = {}
current_index = 0
corpus = []

for i in range(len(df)):
    tokens = my_tokenizer(df['Text'][i])
    corpus.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


word_index_map['advertising']

def tokens_to_vector(tokens):
    # initialize empty array
    x = np.zeros(len(word_index_map))
    # vectorize
    for token in tokens:
        i = word_index_map[token]
        x[i] += 1
    x = x / x.sum()
    return x

#####################################################################
titles = [line.rstrip() for line in open('all_book_titles.txt')]

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

############# tokenizer funtion
def my_tokenizer(x):
    text = re.sub('[^a-zA-Z]', ' ', x)     # alpha only
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]    # stemming
    tokens = [t for t in tokens if t not in set(stopwords.words('english'))]
    return tokens


word_index_map = {}
index_word_map = []
current_index = 0
corpus = []
all_titles = []
error_count = 0

for title in titles:
    try:
        title = title.encode('ascii', 'ignore').decode('utf-8')
        all_titles.append(title)
        # tokenization
        tokens = my_tokenizer(title)
        corpus.append(tokens)
        # giving indices to tokens
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except Exception as e:
        print(e)
        print(title)
        error_count += 1

index_word_map[426]
word_index_map['access']

############# vectorizeing
import numpy as np
N = len(corpus)
D = len(word_index_map)
X = np.zeros((D, N))

i = 0

def tokens_to_vector(tokens):
    # initialize empty array
    x = np.zeros(len(word_index_map))
    # vectorize
    for token in tokens:
        i = word_index_map[token]
        x[i] = 1
    return x

for tokens in corpus:
    X[:, i] = tokens_to_vector(tokens)
    i += 1

# Decomposition
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD()
Z = svd.fit_transform(X)

import matplotlib.pyplot as plt
plt.scatter(Z[:, 0], Z[:, 1])
for i in range(D):
    plt.annotate(s = index_word_map[i], xy = (Z[i, 0], Z[i, 1]))
plt.show()
