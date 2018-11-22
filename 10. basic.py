####################### list method #######################
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
print(areas.index(20.0))
print(areas.count(9.50))
len(areas)

areas.append(24.5)  # add an element
areas.remove()  # remove the first element
areas.reverse()  # reverse the order

####################### array #######################
import numpy as np

height = [180, 215, 210, 210, 188, 176, 209, 200]  # -> list
np_height = np.array(height)

print(np_height[4])

baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]  # 2D array

np_baseball = np.array(baseball)
print(np_baseball.shape)  # dim of the array

print(np_baseball[2, ])  # -> [210, 98.5]
np_weight = np_baseball[:, 1]  # -> [78.4, 102.7, 98.5, 75.2]

np.mean(np_weight)
np.median(np_weight)
np.var(np_weight)
np.std(np_baseball[:, 0])

a = np.sqrt()
int(a)

a = np.array([2.5, 25, 50, 75, 97.5])
iqr = np.percentile(height, q = a)

np.corrcoef()
####################### Dictionary #######################
countries = ['spain', 'france', 'germany', 'norway']  # list
captials = ['madrid', 'paris', 'berlin', 'olso']

ind_ger = countries.index('germnany')
print(captials[ind_ger])

europe = {'spain': 'madrid', 'france': 'paris', 'germany': 'berlin', 'norway': 'oslo'}  # Dictionary

print(europe.items())
print(europe.keys())
print(europe.values)

europe['italy'] = 'rome'    # adding
europe.update({'italy': 'rome'})
del europe['germany']     # delete

print(europe['france'])
print('italy' in europe)    # -> True

# Dictionary in the Dictionary
europe = {'spain' : { 'capital':'madrid', 'population': 46.77 },
          'france' : { 'capital':'paris', 'population': 66.03 },
          'germany': { 'capital':'berlin', 'population': 80.62 },
          'norway': { 'capital':'oslo', 'population': 5.084 }}

# where is the capital of france?
print(europe['france']['capital'])
# add the info of italy
data = { 'capital' : 'rome', 'population' : 59.83 }
europe['italy'] = data
europe['italy'] = { 'capital' : 'rome', 'population' : 59.83 }

####################### Pandas DataFrame #######################
import pandas as pd

names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

my_dict = {'country' : names, 'drives_right' : dr, 'cars_per_cap' : cpc}  # Dictionary
cars = pd.DataFrame(my_dict)  # DataFrame

# gives row names (row labels) to the DataFrame
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']
cars.index = row_labels

cars = pd.read_csv('cars.csv', index_col = 0)
print(cars['country', 'cars_per_cap'])

cars[0:2]  # print out row
cars.loc['JAP']
cars.iloc[2]

cars.loc[['JAP', 'RU']]
cars.iloc[[2, 4]]

cars.loc['JAP', 'drives_right']
cars.iloc[2, 1]

cars.loc[['JAP', 'RU'], 'drives_right']
cars.iloc[[2, 4], 1]

cars.loc[:, 'drives_right']
cars.loc[:, ['drives_right', 'cars_per_cap']]

# indexing
price = [5, 7, 21, 1, 3]
days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri']
views = pd.Series(price, index = days)
print(views.index)
views.index.upper
views.index.name = 'Weekday'

df.index.name
df.columns.name = column_list     # renaming columns
df.set_index(['state', 'month'])
df.sort_index()
df.sort_index(ascending = False)

df.reindex(my_order)
df.swaplevel(0, 1)     # swiching the order of two indices

sales.loc['NY']
sales.loc[('NY', 1), :]
sales.loc[(['NY', 'TX'], 1), :]
sales.loc[(slice(None), 2), :]

a = (df.age > 10) & (df.pclass == 'C')     # boolean series
df_a = df[a]

df_bymonth = df.unstack(level = 'state')
df = df.stack(df_bymonth.stack(level = 'state'))

####################### Data Manipulations #######################
import pandas as pd
df = pd.read_csv('wine.csv')

print(df.shape)    # dim(df)
print(df.columns)    # names(df)
print(df.info())    # str(df)
df.dtypes    # class()
df.dtypes.value_counts()
df.describe()    # summary(df)

df['State'].count()     # length()
df['State'].nunique()     # n_distinct()
df['State'].value_counts(dropna = False)     # count()
df.sort_values('Temperature', ascending =  False)     # sort()

df['State'].median()
df['State'].min()
df.quantile(.05, .95)
df.idxmax(axis = 'columns')

mean = df.mean(axis = 'columns')
mean.plot()
plt.show()

df.gender = df.gender.astype('category')
df.gender = df.gender.astype('str')
df['total_price'] = pd.to_numeric(df['total_price'], errors = 'coerce')     # object => numeric + NaN
df['date'] = pd.to_datetime(df['date'])

# as.factor()
df_dummies = pd.get_dummies(df)
pd.get_dummies(df[['label']], prefix_sep = '_')
categorizer = lambda x: x.astype('cateory')
d[['label', 'sex']] = df[['label', 'sex']].apply(categorizer, axis = 0)

df.pclass = pd.categorical(values = medals.pclass, categories = ['Bronze', 'Silver', 'Gold'], ordered = True)
df.info()                    # ordering categories
df.year = df.year.astype('category')
df.year2 = df.year.cat.reorder_categories(['2017', '2018'], order = True)
print(df.year.cat.categories)

# NaN, NA
nulls_per_column = df.isnull().sum()
df.loc[pd.isnull(df.year)]

assert df.notnull.all()      # checking NA for each columns
assert df.notnull.all().all()     # checking NA for entire DataFrame

assert (df >= 0).all().all()     # asserting all values of df are greater than 0
assert df.gender.dtypes == np.object

# fill NA
df.dropna()
df.reindex('year').ffill()    # forward-filling NA
df.total_price.fillna(-1, inplace = True)
df['total_price'] = df.total_price.fillna(df.total_price.mean())


####################### dataframe with pandas #######################
# cast()
iris_melt = pd.melt(iris, id_vars = 'Species', var_name = 'Measurement', value_name = 'Values')
pd.melt(iris, col_level = 0)     # key-value pairs

# tidy()
iris_pivot = iris_melt.pivot_table(index = 'Species', columns = 'Measurement', values = 'Values',
                                   aggfunc = np.mean)     # aggfunc = count -> frequency
print(iris_pivot.index)    # print the original index before pivoting
iris_pivot_reset = iris_pivot.reset_index()    # reset the index
iris_pivot = iris_melt.pivot_table(index = 'Species', columns = 'Measurement', values = 'Values',
                                   aggfunc = 'sum', margins = True)

# separate()    "Sepal_Length, Sepal_Width"
iris_melt['Part'] = iris_melt.Part_Measure.str[0]
iris_melt['Measure'] = iris_melt.Part_Measure.str[1]

iris_melt['splitted'] = iris_melt.Part_Measure.str.split('_')
iris_melt['Part'] = iris_melt.splitted.str.get(0)
iris_melt['Measure'] = iris_melt.splitted.str.get(1)

# rbind(), cbind()
combined = df1.append(df2, ignore_index = True)
row_concat = pd.concat([df1, df2, df3])

col_concat = pd.concat([df1, df2, df3], keys = ['A1', 'A2', 'A3'], axis = 1, join = 'inner')
# axis = 0: rows(vertically) / 1: columns(horizontally)

# inner_join
blue_red = {'Obama':'blue', 'Trumph':'red'}
election['color'] = election['winner'].map(blue_red)      '''print(list(map(binge_male, num_drinks)))'''

pd.merge(df1, df2, on = 'city')
pd.merge(df1, df2, on = ['city', 'country'], suffixes = ['_D1', '_D2'])
df_merged = pd.merge(df1, df2, left_on = 'id', right_on = 'account_id')

pd.merge(df1, df2, on = ['city', 'country'], how = 'left', fill_method = 'ffill')

# drop duplicate rows
df_2 = df.drop_duplicates()
df.dropna(how = 'any')    # if there is any NA in a row
df.dropna(how = 'all')    # if all the values are NA in a row
df.dropna(thresh = 1000, axis = 'columns')    # drop the axis if it's over thresh

# groupby
titanic.groupby(['pclass', 'embarked']).['survived'].count()

titanic.groupby('pclass')['age']
aggregated = titanic.groupby('pclass')[['age', 'fare']].agg(['max', 'median'])
print(aggregated.loc[:, ('age', 'median')])

def data_range(A):
    return A.max() - A.min()
titanic.groupby('pclass')['age', 'fare'].agg(data_range)

aggregator = {'age':'max', 'fare':data_range}
titanic.groupby('pclass')[['age', 'fare']].agg(aggregator)

age10 = pd.Series(titanic['age'] < 10).map({True : 'under 10', False : 'over 10'})
titanic.groupby([age10, 'pclass'])['survivied'].mean()

from scipy.stats import zscore
df.groupby('region')[['population', 'GDP']].transform(zscore)

# pct_change()
####################### if #######################
num_drinks = [5, 3, 4, 1, 9, 4]

for drink in num_drinks:
    if num_drinks <= 3:
        print('non-binge')
    elif num_drinks <= 5:
        print('nornal')
    else:
        print('binge')

def binge_male(num_drinks):
    if num_drinks <= 5:
        return 'non-binge'
    else:
        return 'binge'

# map(func, list)
print(list(map(binge_male, num_drinks)))

# list comprehension
bingled = [bingle_male(i) for i in num_drinks]

hollywood_star = [['Spiderman': 1996], ['Hulk': 1967], ['Doctor Strange': 1976], ['Iron Man': 1965]]
avergers_dict = {key:value for key, value in hollywood_star}

####################### for loop #######################
# for loop of Dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }

for key, value in europe.items() :
    print('the capital of ' + str(key) + 'is ' + str(value))

# for loop for 2D array
for x in np_height :
    print(str(x) + ' inches')

for x in np.nditer(np_baseball) :
    print(x)

# for loop for pandas DataFrame
for lab, row in cars.iterrows() :
    print(lab)
    print(row)
    print(lab + ': ' + str(row['cars_per_cap']))

for lab, row in cars.iterrows() :
    cars.loc[lab, 'capital_length'] = len(row['capital'])

for lab, row in cars.iterrows() :
    cars.loc[lab, 'CAPITAL'] = row['capital'].upper()

cars['capital_length'] = cars['capital'].apply(len)
cars['CAPITAL'] = cars['capital'].apply(str.upper)
cars['CAPITAL'] = cars.capital.apply(str.upper)

####################### statistical distributions #######################
import numpy as np

np.random.seed(123)     # for reproducable code
print(np.random.rand())
print(np.random.randint(1, 7))

np.random.random()
# rnorm(mean, std, n_samples)
np.random.normal()
# dbinorm(n, p, size)
np.random.binomial()
# pnorm(n, t): the number of clicks in a given time t
np.random.poisson()
np.random.exponential()

####################### functions #######################
def three_shouts(X1, X2, X3):
    # nested function
    def inner(X):
        return X + '!!!'
    # return inner result as a tuple
    return(inner(X1), inner(X2), inner(X3))

print(three_shouts('a', 'b', 'c'))

def echo(n):
    # nested function
    def inner_echo(word):
        echo_word = word * n
        return echo_word
    # return inner
    return(inner_echo)

twice = echo(2)  # create twice function
thrice = echo(3)
print(twice('hello'), thrice('welcome'))

def echo_shout(word):
    echo_word = word*2
    print(echo_word)
    # nested function
    def shout():
        nonlocal echo_word
        echo_word = echo_word + '!!!'
    # call function shout()
    shout()
    print(echo_word)

print(echo_shout('hello'))

def gibberish(*args):
    # initialize an empty string
    hodgepodge = ''
    # concatenate the strings in args
    for word in args:
        hodgepodge += word
    return(hodgepodge)

print(gibberish('luke', 'leia', 'han', 'obi'))

def print_all(**kwargs):
    # print out key-value pairs in **kwargs
    for key, value in kwargs:
        print(key + ':' + value)

print_all(name = 'Hugh Jackman', occupation = 'Actor', Country = 'Australia')

simple_echo = (lambda word, echo: word * echo)
print(simple_echo('hey', 5))

####################### iterators #######################
hollywood_star = ['Spiderman', 'Hulk', 'Doctor Strange', 'Iron Man']

avergers = iter(hollywood_star)
next(avergers)

mylist = ['Jane', 'John', 'Alice', 'Mark']
list(enumerate(mylist))  # -> listing with indexing

text = 'I am an aspiring data scientist'
for i, j in enumerate(text):
    dict = {i:j}

# enumerate : returns elements indexing numbers
# zip : returns two lists in parallel tuple
####################### genenrator #######################
lengths = (len(person) for person in hollywood_star)
for num in lengths:
    print(num)

print(list(lengths))

def lengths(input_list):
    # genenrator function
    for person in input_list:
        yield(len(person))

for results in lengths(hollywood_star):
    print(results)

#########################################################################################
####################### Date & Time #######################
pd.read_csv(filename, parse_dates = True, index_col = 'Date')

time_format = '%Y-%m-%d %H:%M'
my_dates = pd.to_datetime(date_list, format = time_format)
time_temp = pd.Series(temperature_list, index = my_dates)

my_dates.dt.day
my_dates.dt.month
my_dates.dt.year

# indexing
df.set_index('Date', inplace = True)
df = df.set_index(my_dates)

df2 = df.reindex(ts1.index)
df2 = df.reindex(ts1.index, method = 'ffill')

# downsampling / upsampling
df1 = df['Temperature'].resample('6h').count()
df2 = df['Temperature'].resample('A').first().interpolate('linear')     # linear imputation

# rolling
august = df['Temperature']['2010-Aug']
daily_highs = august.resample('D').max()
daily_highs_smoothed = daily_highs.rolling(window = 7).mean()
print(daily_highs_smoothed)

# timezone
central = df['Date'].dt.tz_localize('US/Central')
central.dt.tz_convert('US/Eastern')

# merging on timedate
pd.merge(df1, df2, on = 'date').sort_values('date')
pd.merge_ordered(df1, df2, on = 'date')
pd.merge_asof()

####################### Visualization #######################
import matplotlib.pyplot as plt

iris.plot(kind = 'scatter', x = 'Sepal_Length', y = 'Sepal_Width')
iris['Speices'].plot(subplots = True, title = 'Iris Species')

plt.xlabel('Iris Sepal Length')
plt.ylabel('Iris Sepal Width')

plt.xlim(20, 40)
plt.ylim(20, 40)

plt.show()
plt.clf()

# facet
cols = ['weight', 'mpg']
df[cols].plot(kind = 'box', subplots = True)

plt.subplots(nrows = 2, ncols = 1)
df.weight.plot(ax = axes[0], kind = 'hist', normed = True, bins = 30, range = (0, .3))
df.weight.plot(ax = axes[1], kind = 'hist', normed = True, cumulative = True, bins = 30, range = (0, .3))
plt.show()
