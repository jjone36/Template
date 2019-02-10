####################### Importing Data #######################
############# text
file = open('moby_dick.txt', mode = 'r')
print(file.read())
# Read & print the first 2 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
# Check whether file is closed
print(file.closed)
# Close file
file.close()

import numpy as np
file = 'moby_dick.txt'
data = np.loadtxt(file, delimiter = '\t', skiprows = 1)
data = np.genfromtxt(file, delimiter=',', names=True, dtype=None)
data = np.recfromcsv(file)
data = pd.read_csv(file, sep= '\t', comment= '#', na_values= 'Nothing')

############# Excel
import pandas as pd
file = 'sales.xlsx'
xl = pd.ExcelFile(file)
print(xl.sheet_names)
df = xl.parse(1, parse_cols= [0, 3], skiprows= [0], names= ['Country', 'AAM due to War (2002)'])


############# SQL
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///Datacamp.sqlite')
table_names = engine.table_names()
print(table_names)

con = engine.connect()
rs = con.execute('SELECT * FROM Album')
con.close()

# fetchall()
df = pd.DataFrame(rs.fetchall())
print(df.head())

# fetchmany()
with engine.connect() as con:
    rs = con.execute('SELECT LastName, Title FROM Employee')
    df = pd.DataFrame(rs.fetchmany(3))
    df.columns = rs.keys()

# One line!
df = pd.read_sql_query('SELECT LastName, Title FROM Employee', engine)


############# glob
import glob
import pandas as pd

files = glob.glob('*.csv')
all_df = [pd.read_csv(a) for a in files]


############# pickle
import pickle
# Open pickle file and load data: d
with open('data.pkl', 'rb') as file:
    d = pickle.load(file)

############# SAS
HDF5
from sas7bdat import SAS7BDAT
with SAS7BDAT('sales.sas7bdat') as file:
    df_sas = file.to_data_frame()
print(df_sas.head())

############# Stata
df_dta = pd.read_stata('sales.dta')
print(df_dta.head())

############# HDF5
import h5py
data = h5py.File('data.hdf5', 'r')
print(type(data))
# Print the keys of the file
for key in data.keys():
    print(key)               # => 'meta', 'quality', 'strain'

group = data['strain']
# Check out keys of group
for key in group.keys():
    print(key)               # => 'A', 'B', 'C'
data['strain']['A'].value

############# MATLAB
import scipy.io
df_mat = scipy.io.loadmat('data.mat')
# Print the keys of the MATLAB dictionary
print(df_mat.keys())


####################### Web Sraping #######################
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

####################### Dask #######################
# https://www.datacamp.com/courses/parallel-computing-with-dask
############# Big data Importing
df = []
for chunk in pd.read_csv('big_data.csv', chunksize = 10000):
    is_tall = (chunk.height > 200)
    filtered = chunk.loc[is_tall]
    df.append(filtered)

def filter_height(x):
    is_tall = (x.height > 200)
    return x.loc[is_tall]
df = [filter_height(chunk) for chuck in pd.read_csv('big_data.csv', chunksize = 10000)]

# genenrator (Lazy computation)
dfs = (pd.read_csv('file') for file in files)
tall_df = [filter_height(df) for df in dfs]

############# Dask delayed (Lazy Computation)
from math import sqrt
def f(a):
    return sqrt(a + 4)

def g(b):
    return b - 3

def h(c):
    return c ** 2

from dask import delayed
b = delayed(f)(a)
c = delayed(g)(b)
outcome = delayed(h)(c)

print(outcome)
outcome.compute()
outcome.visualize()

# Permenantly delayed
f = delayed(f)
g = delayed(g)
h = delayed(h)
outcome = h(g(f(4)))  # -> lazy computation

@delayed
def f(a):
    return sqrt(a + 4)

############# Dask array -> numpy array
n_chucks = 4
chunk_size = len(X) / n_chucks
result = 0

for k in range(n_chucks):
    a = n_chucks * k
    X_chunk = X[a : a + n_chunks]
    result += X_chunk.sum()

import dask.array as da
# Convert numpy array X into dask array
X_dask = da.from_array(X, chunks = len(X) // n_chucks)
X_desk.chunks
result = X_dask.sum()
result.compute()

############# Dask DataFrame
import dask.dataframe as dd
dd.compute().info()

df = dd.read_csv('*.csv', assume_missing = True)


############# Dask bags -> dataframe
import dask.bag as db
zen = db.read_text('zen.txt')
taken = zen.take(1)   # First element

upper = zen.str.upper()
upper.take(1)

words = zen.str.split(' ')
n_workds = map(len, words)

# from_sequence
nested = [[0, 1, 2, 3],
          {},
          [6.5, 3.14],
          'Python',
          {'version': 3},
          ""]
bag_nested = db.from_sequence(nested)
bag_nested.count()

# bag with map, filter
squared = map(to_square, [1, 2, 3, 4, 5, 6])
print(squared)    # -> map object (lazy implementation)
squared = list(squared)
print(squared)

evens = filter(is_even, [1, 2, 3, 4, 5, 6])   # boolen functions
list(evens)

nums = db.from_sequence([1, 2, 3, 4, 5, 6])
numbs.map(to_square).compute()
numbs.filter(is_even).compute()

############# json file
json_data = db.read_text('items-by-line.json')
json_data.take(1)    # Not file! It's strings

dict_items = json_data.map(json.loads)
dict_items.take(1)

dict_items.take(1)[0]['height']
dict_itmes.pluck('height').compute()
