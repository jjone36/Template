##############################################
############# Twitter API
# Import package
import tweepy
# Store OAuth authentication credentials in relevant variables
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""
# Pass OAuth details to tweepy's OAuth handler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
# Initialize Stream listener
l = MyStreamListener()

# Create your Stream object with authentication
stream = tweepy.Stream(auth, l)
# Filter Twitter Streams to capture data by the keywords:
stream.filter(track = ['clinton', 'trump', 'sanders', 'cruz'])

##############################################
############# jason
import jason

with open('movie.jason') as json_file:
    json_data = json.load(json_file)
type(json_data)     # -> dictionary!

for key, value in json_data.item():
    print(key + ': ', value)

# Assign URL to variable: url
url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Decode the JSON data into a dictionary: json_data
json_data = r.json()
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])
# Print the Wikipedia page extract
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)


############# HTTP
from urllib.request import urlretrieve    # downloading web files
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
'winequality.csv' = urlretrieve(url)
df = pd.read_csv('winequality.csv', ';')
df = pd.read_csv(url, ';')

from urllib.request import urlopen, Request    # Get request from HTTP page
url = 'http://www.datacamp.com/teach/documentation'
# This packages the request
request = Request(url)
# send the request and catches the response
response = urlopen(request)
# print the datatype of response
print(type(response))    # -> `http.client.HTTPResponse`
# extract the response
html = response.read()
print(html)
# close the request
response.close()

import requests   # the higher level package / not require response.close()
# package the request, send the request and catch the response
r = requests.get(url)
# extrack the response
text = r.text
print(text)

##############################################
############# BeautifulSoup
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

############# Scrapy
from scrapy import Selector
import requests
url
html = requests.get(url).content
sel = Selector(text = html)

sel.xpath(xpath)


############# Selenium
import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

chrome_path = "C:\\Users\jjone\Downloads\chromedriver"
driver = webdriver.Chrome(executable_path = chrome_path)

url = 'https://www.sephora.com'
driver.get(url)

# Click
xpath
driver.find_element_by_xpath(xpath).click()

# input & search
search = driver.find_element_by_id('search')
text = 'download'
search.send_keys(text)
search.send_keys(Keys.ENTER)

# hyperlink scraping
element = driver.find_elements_by_class_name('hyperlink class')
element.get_attribute('href')

subpageURL = []
for a in element:
    subURL = a.get_attribute('href')
    subpageURL.append(subURL)

# text scraping
detail = driver.find_element_by_class_name('css-192qj50').text
pattern = r"✔ \w+\n"
df.skin_type[i] = re.findall(pattern, detail)

# Adding exception
from selenium.common.exceptions import NoSuchElementException
for i in range(len(df)+1):
    try:
        rank = driver.find_element_by_class_name('css-ffj77u').text
        rank = re.match('\d.\d', rank).group()
        df['rank'][i] = str(rank)

    except NoSuchElementException:
        df['rank'][i] = 0

# Scroll down function
def scrollDown(driver, numberOfScrollDowns):
    body = driver.find_element_by_tag_name("body")
    while numberOfScrollDowns >=0:
        body.send_keys(Keys.PAGE_DOWN)
        numberOfScrollDowns -= 1
    return driver

browser = scrollDown(driver, 10)
time.sleep(20)

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException



df.to_csv('test.csv', encoding = 'utf-8-sig', index = False)
