###########################################################
# install
'C:\Users\jjone\Anaconda3\Scripts'
pip install selenium
Chrome Driver download
'https://sites.google.com/a/chromium.org/chromedriver/'

'https://blog.michaelyin.info/how-crawl-infinite-scrolling-pages-using-python/'

####################### Web Scraping #######################
import requests
from bs4 import BeautifulSoup
url = 'https://www.python.org/~guido/'
# package the request, send the request and catch the response
r = requests.get(url)
# extrack the response as html
html_doc = r.text
# create a BeautifulSoup object from the html
soup = BeautifulSoup(html_doc)

pretty_soup = soup.prettify()    # prettify the BeautifulSoup object
print(pretty_soup)

guido_title = soup.title()    # get the title of the page
print(guido_title)
print(soup.title)
print(soup.get_text)    # get the text of the page

a_tags = soup.find_all('a')    # find all hyperlinks (a tags)
for link in a_tags:
    print(link.get('href'))


############ BeautifulSoup
import urllib.request import urlopen, Request
from bs4 import BeautifulSoup

tickers = ['AAPL', 'MMM', 'T', 'S', 'V2']
for ticker in tickers:
    url = 'https://finance.yahoo.com/quote/' + ticker + '?p=' + ticker
    r = requests.get(url)
    html_doc = r.text
    # create a BeautifulSoup object
    soup = BeautifulSoup(html_doc)
    table = soup.find('div', id = 'quote-summary')
    print(table.text)
    # print in a table format
    rows = table.find_all('tr')
    for row in rows:
        print(a.text)
    for row in rows:
        a = row.find_all('td')[0].text.str
        label.append(a)
        b = row.find_all('td')[1].text.str
        value.append(b)
    # make them in a data frame
    dic = {'Label': label, 'Value': value}
    df = pd.DataFrame(dic)
    df.to_csv(ticker + '.csv')


############ Selenium
from selenium import webdriver
chrome_path =  "C:\\Users\jjone\Downloads\chromedriver"
driver = webdriver.Chrome(executable_path = chrome_path)
driver.implicitly_wait(10)

# open the web page
url = ''
driver.get(url)

# scraping elements
# find_elements_by_(id, name, xpath, link_text, partial_link_text, tag_name, class_name, css_selector)
element = driver.find_elements_by_link_text('Download')
element.text
element.click()
subpageURL = element.get_attribute('href')


element = driver.find_elements_by_class_name(css)
subpageURL = element.get_attribute('href')
subpageURL = element.find_all('href')

# clicking pages
xpath = ''
btn = driver.find_element_by_xpath(xpath)
btn.click()

# searching
search = driver.find_element_by_xpath(xpath)
search = driver.find_element_by_id('')
search.send_keys('belif').submit()


from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait       # explicitly wait
from selenium.webdriver.support import expected_conitions as EC
from selenium.common.exceptions import TimeoutException

url = ''
xpath = ''
def init_driver():
    driver = webdriver.Chrome(executable_path = chrome_path)
    driver.wait = WebDriverWait(10, driver)
    return driver

def get_data(driver):
    driver.get(url)
    element = driver.wait.until(EC.presence_of_element_located(By.XPATH, xpath))
    print element.text

driver = init_driver()
get_data(driver)

###########################################################
from selenium import webdriver
chrome_path
driver = webdriver.Chrome(executable_path = chrome_path)
driver.implicitly_wait(30)

## sephora review
url = 'https://www.sephora.com/product/c-firma-day-serum-P400259?icid2=products%20grid:p400259:product'
driver.get(url)
reviews = driver.find_elements_by_class_name('css-eq4i08')

driver.find_element_by_xpath('//*[@id="ratings-reviews"]/div[10]/button').click()
reviews2 = driver.find_elements_by_class_name('css-eq4i08')

reviews = reviews.append(reviews2)
for review in reviews:
    print(review.text)


type = ['skin-care-solutions']
url = 'https://www.sephora.com/shop/' + type + '?pageSize=300'
driver.get(url)
page = WebDriverWait(driver, 10).until(EC.presence_of_elements_located())
item_pages = page.find_all('href')

# 2nd page click
driver.find_element_by_xpath('/html/body/div[1]/div[2]/div/div/div/div/div[2]/div[2]/div[2]/div/div[2]/div[2]/div/button[3]').click()

###########################################################
'https://regex101.com/#python'
####################### String Manipulations #######################
import re

sentence = 'My phone number is 425-125-9535'
pattern = r'(\d{3})-(\d{3}-\d{4})'
phone_Regex = re.compile(pattern)
c1 = phone_Regex.search(sentence)
c1.group()
c1.group(1)
c1.group(2)

c2 = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')
print(c2)

# gsub()
df['total_price_replaced'] = tips.total_price.apply(lambda x: x.replace('$', ''))
# str_match()
df['total_price_re'] = tips.total_price.apply(lambda x: re.findall('\d+\.\d+', x)[0])
# str_trim()
df.columns = df.columns.str.strip()
# str_contains()
df['Destination'].str.contains('DAL')
df['name'].str.replace('Mr', 'Mrs')
# tolower(), str_to_lower() / toupper()
df.name = df.name.str.lower()
df.name = df.name.str.upper()

re.search()
re.match()
re.findall()
re.sub(pattern, replacement, text)
[re.sub(pattern, '', l) for l in lines]
re.split()

###########################################################
####################### NLP_Udemy #######################
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
    text = [ps.stem(t) for t in text if not t in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)

# Create the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv = CountVectorizer(token_pattern, ngram_range = (1, 2))
X = cv.fit_transform(corpus).toarray()     # making as a matrix
y = dataset.iloc[:, 'target'].values


############# TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

original_vocab = tfidf.vocabulary_
vocab = [v:k for k, v in tfidf.vocabulary_.items()]

csr_mat = tfidf.fit_transform(documents)
csr_mat.shape
print(csr_mat.toarray())
# Get the words: words
words = tfidf.get_feature_names()
print(words)


############# NMF (Non-negative Matrix Factorization)
from sklearn.decomposition import NMF
nmf = NFM(n_components = 10)
nmf.fit(words)
nmf_features = nmf.transform(words)
# -> topics (similar NMF features, similar documents: cosine similarity)
df = pd.DataFrame(nmf_features, index = titles)

components_df = pd.DataFrame(nmf.components_, column = words)
components_df.iloc[3, :].nlargest()

# cosine similarity (-> recommendation)
import pandas as pd
from sklearn.preprocessing import normalize
# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)
# Create a DataFrame: df
df = pd.DataFrame(norm_features, index = titles)
# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']
# Compute the dot products: similarities
similarities = df.dot(article)
# Display those with the largest cosine similarity
print(similarities.nlargest())


####################### NLP_Datacamp #######################
############# nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
sentences = sent_tokenize(text)
tokens = word_tokenize(text)
# Convert the tokens into lowercase
lower_tokens = [t.lower() for t in tokens]
# Create the bag-of-words
bow = Counter(lower_tokens)
# Print the 10 most common tokens
print(bow.most_common(10))


from nltk.tokenize import TweetTokenizer, regexp_tokenize
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]

regexp_tokenize(textdata, pattern)
tokenized_lines = [regexp_tokenize(s, '[#|@]\w+') for s in lines]
line_nwords = [len(a) for a in tokenized_lines]
plt.hist(line_nwords)
plt.show()


from nltk.stem import WordNetLemmatizer
# Retain alphabetic words
alpha_only = [t for t in lower_tokens if t.isalpha()]
# Remove all stop words
stops_removed = [t for t in alpha_only if t not in english_stops]
# Lemmatize all tokens into a new list
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in stops_removed]
bow = Counter(lemmatized)
print(bow.most_common(10))


############# gensim
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
tokenized = [word_tokenize(t.lower()) for t in text]
# mapping each of tokens
dictionary = Dictionary(tokenized)
dictionary.token2id
id_computer = dictionary.token2id.get('computer')
print(dictionary.get(id_computer))
# Create a corpus (token ID, frequency)
corpus = [dictionary.doc2bow(a) for a in tokenized]
print(corpus[4][:10])


from gensim.models.tfidfmodel import TfidfModel
tfidf = TfidfModel(corpus)
# Calculate the tfidf weights of doc
tfidf_weights = tfidf[doc]
# Sort the weights from highest to lowest
sorted_tfidf_weights = sorted(tfidf_weights, key = lambda w: w[1], reverse=True)
for term_id, weight in sorted_tfidf_weights:
    print(dictionary.get(term_id), weight)


############# pos_tag
import nltk
nltk.pos_tag('What is machine learning?'.split())

sen = 'Hugh Michael Jackman is an Australian actor, singer, and producer.'
tags = nltk.pos_tag(sen.split())
nltk.ne_chunk(tags)
nltk.ne_chunk(tags).draw()
