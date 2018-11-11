###########################################################
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
# install
'https://sites.google.com/a/chromium.org/chromedriver/'
get pip.py install
pip install selenium
Chrome Driver download

from selenium import webdriver
chrome_path
driver = webdriver.Chrome(executable_path = chrome_path)
driver.implicitly_wait(10)

# open the web page
url = ''
driver.get(url)

# scraping elements
# find_elements_by_(id, name, xpath, link_text, partial_link_text, tag_name, class_name, css_selector)
element = driver.find_elements_by_class_name(css)
element = driver.find_elements_by_link_text('Download')
element.click()
element.text
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
    driver = webdriver.Chrome(executable_path = '/Users/...')
    driver.wait = WebDriverWait(driver, 5)
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
####################### Regular expression #######################



###########################################################
####################### Text Mining #######################
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
cv = CountVectorizer(token_pattern, ngram_range = (1, 2))
X = cv.fit_transform(corpus).toarray()     # making as a matrix
y = dataset.iloc[:, 'target'].values


############# TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
csr_mat = tfidf.fit_transform(documents)
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
