###### read.table(). read_delim() ######
potatoes = read.table(url)
potatoes = read.table(url, sep = ',', header = T)
str(potatoes)

###### read.csv(). read_csv() ######
url = "https://assets.datacamp.com/production/course_1477/datasets/potatoes.csv"
potatoes = read.csv(url)
head(potatoes)

potatoes2 = read_csv(url, col_types = 'cccccnnn')
head(potatoes2)

###### read.delim(). read_tsv() ######
url = "https://assets.datacamp.com/production/course_1477/datasets/potatoes.txt"
potatoes = read.delim(url)
head(potatoes)
colname = c("area", "temp", "size", "storage", "method", "texture", "flavor", "moistness")
potatoes = read.delim(url, col.names = colname)
head(potatoes)

potatoes2 = read_tsv(url)
str(potatoes2)


url2 = "https://assets.datacamp.com/production/course_1477/datasets/hotdogs.txt"
hotdogs = read.delim(url2)
str(hotdogs)
hotdogs = read.delim(url2, header = FALSE, col.names = c("type", "calories", "sodium"),
                      colClasses = c('factor', 'NULL', 'numeric'))
str(hotdogs)
head(hotdogs)

###### readxl Package ######
library(readxl)
excel_sheets("urbanpop.xlsx")

path = "urbanpop.xlsx"
sheet1 = read_excel(path, sheet = 1)
sheet2 = read_excel(path, sheet = 2)
sheet3 = read_excel(path, sheet = 3)
pop_list = list(sheet1, sheet2, sheet3)
str(pop_list)

pop_list2 = lapply(excel_sheets("urbanpop.xlsx"), read_excel, path = "urbanpop.xlsx")
str(pop_list2)

###### read.xls() ######
library(gdata)
urbanpop = read.xls("urbanpop.xls", sheet = 2)  # ??

library(readxl)
url.xls = "http://s3.amazonaws.com/assets.datacamp.com/production/course_1478/datasets/latitude.xls"

excel_gdata = read.xls(url.xls)  # ??

download.file(url = url.xls, destfile = 'latitude.xls')
excel_sheets('latitude.xls')  # ??
sheet1 = read_excel("latitude.xls", sheet = 1)


###### XLConnect Package ######
library(XLConnect)
my_book = loadWorkbook("urbanpop.xlsx")  # loading
sheets = getSheets(my_book) # browsing sheets

my_book_S2 = readWorksheet(my_book, sheet = 2)  # loading sheets from the Excel file
c1 = readWorksheet(my_book, sheet = 2, startCol = 3, endCol = 5)

sheets = getSheets(my_book)[1:3]
sheets = getSheets(my_book)
fullData = lapply(sheets, readWorksheet, object = my_book)
str(fullData)

createSheet(object = my_book, name = 'data_Sum')  # writing a new sheet from R

sheets = getSheets(my_book)[1:3]
dims = sapply(sheets, function(x){
  sheet_x = readWorksheet(my_book, sheet = x)
  dim(sheet_x)
}, USE.NAMES = FALSE)
newData = data.frame(sheet = sheets, nrow = dims[1, ], ncol = dims[2, ])

writeWorksheet(object = my_book, sheet = 'data_Sum', data = newData)
saveWorkbook(my_book, file = "urbanpop2.xlsx")

renameSheet(my_book, sheet = 4, newName = 'Summary')
removeSheet(my_book, sheet = 'Summary')

###### RDS ######
download.file(url = url, destfile = 'dataset.csv')
dataset = read.csv('dataset.csv')

saveRDS(object = dataset, file = 'dataset_modified.csv')
dataset2 = readRDS(file = 'dataset_2.RDS')

###### Web Scraping ######
## JSON
library(httr)
url = "http://www.example.com/"
resp = GET(url)
page_data = content(x = resp)

if(http_error(resp)){
  warning('This Page Is Inappropriate')
} else {
  content(resp, as = 'text')
}

url2 = "http://www.omdbapi.com/?apikey=ff21610b&t=Annie+Hall&y=&plot=short&r=json"
resp2 = GET(url2)
http_type(resp2)

page_movie_raw = content(x = resp2, as = 'raw')
page_movie_text = content(x = resp2, as = 'text')
str(page_movie_text)

library(jsonlite)
page_movie = fromJSON(url2)
str(page_movie)
page_movie$Language

movie_ratings = content(resp2)$Ratings
movie_ratings %>%
  bind_rows()

## Html
library(rvest)
url = 'https://en.wikipedia.org/wiki/Hugh_Jackman'
resp = GET(url)
http_type(resp)

page = read_html(url)
page$node

html_nodes(page, css = 'table')
html_nodes(page, css = '#firstHeading')
infobox_element = html_nodes(page, css = '.infobox')

html_name(x = infobox_element)
html_node(x = infobox_element, css = '.fn') %>% html_text()
# script = read_html(url) %>% html_node(css = '#main-content') %>% html_text()

test <- read_html('
                  <h1 class = "main">Hello world!</h1>
                  ')
html_node(x = test, css = '.main') %>% html_text()    # get text contents
html_node(x = test, css = '.main') %>% html_name()    # get tag name

table_element_1 = html_node(x = page, css = 'table')
table_1 = html_table(table_element_1)
table_1 = table_1[-1, ]
colnames(table_1) = c('key', 'value')
table_1 = subset(table_1, !key == "")

## XML
base_url = "https://en.wikipedia.org/w/api.php"
query_params = list(action = 'parse',
                    page = 'Hugh Jackman',
                    format = 'xml')
resp_xml = GET(url = base_url, query = query_params)

page_text = content(resp_xml, as = 'text')
page_text %>% read_xml()

page = content(resp_xml)
infobox_element = xml_text(page) %>% read_html() %>% html_node(css = '.infobox')
fn_element = xml_text(page) %>% read_html() %>% html_node(css = '.fn')
fn_text = html_text(fn_element)

wiki_infobox = function(name){
  api_url = "https://en.wikipedia.org/w/api.php"
  query_params = list(action = 'parse',
                      page = name,
                      format = 'xml')
  resp = GET(url = api_url, query = query_params)
  page_xml = content(resp)

  page_html = page_xml %>%
    xml_text() %>%
    read_html()

  wiki_name = html_node(x = page_html, css = '.fn') %>% html_text()

  table_1 = html_node(x = page_html, css = '.infobox') %>% html_table()
  colnames(table_1) = c('key', 'value')
  table_1 = subset(table_1, !key == "")
  name_df = data.frame(key = 'Full name', value = wiki_name)
  wiki_table = rbind(name_df, table_1)

  return(wiki_table)
}
wiki_infobox(name = 'Emma Stone')
wiki_infobox(name = 'Hugh Jackman')
wiki_infobox(name = 'Ed sheeran')

#### API ####
## Youtube
#https://help.aolonnetwork.com/hc/en-us/articles/218079623-How-to-Create-Your-YouTube-API-Credentials#1
api_key = '542589435889-5o92q1aojo6lan2qmgd8in1v7hvdmuv6.apps.googleusercontent.com'
api_secret = 'ngqxw_Z2ZvVI_ZOI8mQvqlK6'

yt_oauth(app_id = api_key, app_secret = api_secret)
file.remove('.httr-oauth')

## Twitter
library(twitteR)

api_key = 'nSRwsAPYUm5qFuKMnZYCw4Cf4'
api_secret = 'i3hepe95lUqizYy9xBgGInhjfALzgJRqUGBBUhJ9Lzs5xCuWMs'
myToken = '1040508187407933440-nsCV8iTHIYorJPG0muY7lSoToV9THk'
myToken_secret = 'jDViyINeo4FtZBd0P1rv0p2dReGQ5Hk9FcbvkU10BGzF7'

setup_twitter_oauth(consumer_key = api_key, consumer_secret = api_secret,
                    access_token = myToken, access_secret = myToken_secret)

tweets = searchTwitter(searchString = '#BTS', n = 100, lang = 'en', resultType = 'recent')

tweets_df = lapply(tweets, as.data.frame) %>% do.call(what = 'rbind')
tweets_text = sapply(tweets, function(x) x$getText())


user_df = lookupUsers(tweets_df$screenName) %>% twListToDF()
user_located = !is.na(user_df$location)

######################################################################
##################### String Manipulation ############################
library(rebus)
START
END
ANY_CHAR
OPEN_PAREN
DGT
dgt(2)
DOT
SPC
WRD
"gr" %R% or('e', 'a') %R% "y"
(vowels = char_class('aeiouAEIOU'))
negated_char_class('aeiouAEIOU')
one_or_more(WRD)
zero_or_more()
exactly(vowels)
DGT %R% optional(DGT)
optional(SPC) %R% or('M', "F")
as.hexmode(utf8ToInt('a'))
?Unicode
ascii_upper()

###### chapter 1-1 ######
line1 <- "The table was a large one, but the three were all crowded together at one corner of it:"
line2 <- '"No room! No room!" they cried out when they saw Alice coming.'
line3 <- "\"There's plenty of room!\" said Alice indignantly, and she sat down in a large arm-chair at one end of the table."
lines = c(line1, line2, line3)
cat(lines, sep = "")


###### chapter 1-2 ######
format(c(0.0011, 0.011, 1), digits = 1)  # digits -> ���� ���� �� ����
format(c(1.0011, 2.011, 1), digits = 1)

percent_change  <- c(4, -1.91, 3.00, -5.002)
income <-  c(723.19, 1030.18, 10291.93, 1189192.18)
p_values <- c(0.12, 0.98, 0.0000191, 0.00000000002)

# Format percent_change to one place after the decimal point
format(percent_change, digits = 2)
# Format income to whole numbers
format(income, digits = 2)
format(income, scientific = T, digits = 2)
format(p_values, digits = 2)

(trimmed_income = format(income, trim = T, digits = 2))
(pretty_income = format(income, trim = T, digits = 2,
                         big.mark = ',', big.interval = 3))

x <- c(0.0011, 0.011, 1)
y <- c(1.0011, 2.011, 1)
format(x, digits = 2)
formatC(x, format = 'f', digits = 2)  # digits -> �Ҽ��� ���� �ڸ���
formatC(y, format = 'f', digits = 1)
(pretty_percent <- formatC(percent_change, format = 'f', flag = '+', digits = 1))
formatC(p_values, format = 'g', digits = 2)


paste('$', pretty_income, sep = "")
paste(pretty_income, '%', sep = "")

years <- c(2010, 2011, 2012, 2013)
(year_percent <- paste(years, ": ", pretty_percent, "%", sep = ""))
paste(years, ":", pretty_percent, collapse = ', ')


###### chapter 2-1 ######
library(stringr)
library(babynames)
library(tidyverse)

head(babynames)
babynames_2014 = babynames %>% filter(year == 2014)
boy_names = filter(babynames_2014, sex == 'M')$name
girl_names = filter(babynames_2014, sex == 'F')$name

head(girl_names)
str_length(girl_names) %>% head()
length(girl_names)
str(girl_names)
factor(girl_names) %>% str_length() %>% head()

str_sub(c("Hugh", "Jackman"), 2, 4)
str_sub(c("Hugh", "Jackman"), -4, -2)

contains_zz = str_detect(boy_names, 'zz')
sum(contains_zz)
boy_names[contains_zz] %>% str_replace('zz', 'Z')
str_subset(boy_names, 'zz')
str_extract(boy_names, 'zz')

starts_U = str_subset(girl_names, 'U')
tolower(starts_U)
toupper(starts_U)

number_as = str_count(girl_names, 'a')
number_As = str_count(girl_names, 'A')
total_as = number_as + number_As
girl_names[number_as > 3]

# library(dplyr)
# starts_with()
# ends_with()
# contains(), str_contains()

###### chapter 2-2 ######
date_ranges = c("23.01.2017 - 29.01.2017", "30.01.2017 - 06.02.2017")

dates = str_split(date_ranges, pattern = " - ")
dates
dates1 = str_split(date_ranges, pattern = fixed(" - "), simplify = T, n = 3)
dates1

dates1[, 1] %>% str_split(pattern = fixed('.'))
dates1[, 1] %>% str_split(pattern = fixed('.'), simplify = T)


both_names = c("Box, George", "Cox, David")
both_names_split = str_split(both_names, pattern = fixed(', '), simplify = T)
first_names = both_names_split[, 2]
last_names = both_names_split[, 1]

lines
words = str_split(lines, pattern = fixed(" "), simplify = F)
str_length(words)
lapply(words, str_length)


ids = c("ID#: 192", "ID#: 118", "ID#: 001")
id_nums = str_replace(ids, pattern = "ID#: ", replacement = "")
str(ids)
str(id_nums)
id_ints = as.numeric(id_nums)

phone_numbers = c("510-555-0123", "541-555-0167")
str_replace_all(phone_numbers, pattern = '-', replacement = " ")
str_replace_all(phone_numbers, pattern = "-", replacement = '.')

names = c("Diana Prince", "Clark Kent")
# Split into first and last names
names_split = str_split(names, pattern = fixed(' '), simplify = T)
# Extract the first letter in the first name
first_letter = names_split[, 1] %>% str_sub(1, 1)
# Combine the first letter ". " and last name
str_c(first_letter, ". ", names_split[, 2])


###### chapter 3-1 ######
library(rebus)
START
END
ANY_CHAR
OPEN_PAREN
DGT
dgt(2)
DOT
SPC
WRD
"gr" %R% or('e', 'a') %R% "y"
(vowels = char_class('aeiouAEIOU'))
negated_char_class('aeiouAEIOU')
one_or_more(vowels)
zero_or_more()
exactly(vowels)
DGT %R% optional(DGT)
optional(SPC) %R% or('M', "F")
as.hexmode(utf8ToInt('a'))
?Unicode
ascii_upper()

x <- c("cat", "coat", "scotland", "tic toc")
str_view(x, pattern = START %R% 'c')
str_view(x, pattern = "at" %R% END)

pattern = "q" %R% ANY_CHAR
pattern
part_with_q = str_extract(boy_names, pattern = 'q.')
table(part_with_q)


x = c('grey sky', 'gray elephant')
pattern = "gr" %R% or('e', 'a') %R% "y"
pattern2 = 'gr' %R% char_class('ae') %R% 'y'
str_view(x, pattern)
str_view(x, pattern2)

vowels = char_class('aeiouAEIOU')
str_view_all(x, vowels)
(num_vowels = str_count(girl_names, vowels)) %>% head()
(num_vowels = str_length(girl_names, vowels)) %>% head()

# See names with only vowels
str_view(girl_names, exactly(one_or_more(vowels)), match = T)

not_vowels = negated_char_class('aeiouAEIOU')
str_view(girl_names, exactly(one_or_more(not_vowels)), match = T)


###### chapter 3-2 ######
email = c('thor@avengers.com', "(wolverline@xmen.com)", 'flyj2w36@naver.com')
pattern = capture(one_or_more(WRD)) %R%
  '@' %R% capture(one_or_more(WRD)) %R%
  DOT %R% one_or_more(WRD)

str_view(email, pattern)
str_match(email, pattern)

pair_of_repeated = capture(WRD %R% WRD) %R% REF1
str_view(boy_names, pattern = pair_of_repeated, match = T)
# Names with a pair that reverses
pair_that_reverses = capture(WRD) %R% capture(WRD) %R% REF2 %R% REF1
str_view(boy_names, pattern = pair_that_reverses, match = T)


###### chapter 4 ######
x = c('cat', 'CAT', "Cat", "ccat", "cAt", 'caterpillar')
str_view(x, "cat")
str_to_lower(x) %>% str_view('cat')
tolower(x) %>% str_view('cat')
str_view(x, whole_word('cat'), match = T)

str_extract(x, 'cat') # 'caterpillar' -> cat, capital ignored
str_detect(x, 'cat')

str_to_lower(x) %>% str_detect('cat') # 'caterpillar' -> cat
str_detect(x, whole_word('cat')) # capital ignored
catcident = str_to_lower(x) %>% str_detect(whole_word('cat'))  # OK
x[catcident]

str_view(x, regex('cat', ignore_case = T), match = T)


emails <- c("john.doe@ivyleague.edu", "education@world.gov", "dalai.lama@peace.org",
            "invalid.edu", "quant@bigdatacollege.edu", "cookie.monster@sesame.tv")

'@' %R% zero_or_more(WRD) %R% DOT %R% 'edu'

grepl(pattern = '@.*\\.edu', x = emails)
(hits = grep(pattern = '@.*\\.edu', x = emails))
emails[hits]

sub(pattern = '@.*\\.edu$', replacement = '@datacamp.edu', x = emails)
REF1

######################################################################
###################### Time Series ###################################
###### chapter 1-1 ######
Sys.Date()
Sys.time()

x = "2013-04-03"
str(x)
as.Date(x)
str(as.Date(x))

str1 = "May 23, '96"
str2 = "2012-03-15"
str3 = "30/January/2006"

(date1 = as.Date(str1, format = "%b %d, '%y"))  # ??
(date2 = as.Date(str2, format = "%Y-%m-%d"))
(date3 = as.Date(str3, format = "%d/%B/%Y"))  # ??

format(date2, '%A')

library(anytime)
sep_10_2009 = c("September 10 2009", "2009-09-10", "10 Sep 2009", "09-10-2009")
sep_10_2009
anytime(sep_10_2009)

as.POSIXct('2010-10-01 12:12:00')
as.POSIXct("2010-10-01 12:12:00", tz = "America/Los_Angeles")

str1 = "May 23, '96 hours:23 minutes:01 seconds:45"
str2 = "2012-3-12 14:23:08"

time1 = as.POSIXct(str1, format = "%B %d, '%y hours:%H minutes:%M seconds:%S")  # ??


###### chapter 1-2 ######
library(readr)
logs = read_csv("https://assets.datacamp.com/production/course_5348/datasets/cran-logs_2015-04-17.csv")
logs
cut = as.POSIXct("2015-04-16 07:13:33", tz = "UTC")
logs %>% filter(datetime > cut) %>%
  ggplot(aes(x = datetime)) + geom_density() + facet_wrap(~ r_version, ncol = 1) +
  geom_vline(aes(xintercept = as.POSIXct("2015-04-18 12:30:00")))

releases = read_csv("https://assets.datacamp.com/production/course_5348/datasets/rversions.csv")
head(releases)

(p = ggplot(releases, aes(x = date, y = type, color = factor(major))) + geom_line(aes(group = 1)))
p + scale_x_date(date_breaks = "10 years", date_labels = '%Y')
p + xlim(as.Date("2010-01-01"), as.Date("2014-01-01"))

###### chapter 2-1 ######
library(lubridate)

ymd(20180815)
dmy("31-Jan-2017")
ymd_hm("2017-01-01 11:30", tz = "UTC")

ndmi = read.csv("PlotNDMI.csv")
str(ndmi)
ndmi = ndmi %>% mutate(Date2 = dmy(Date),
                       Year = year(Date2),
                       Month = month(Date2))
head(ndmi)
aggregate(NDMI ~ Month + Year, ndmi, mean)


two_orders <- c("October 7, 2001", "October 13, 2002", "April 13, 2003", "17 April 2005", "23 April 2017")
parse_date_time(two_orders, orders = c('mdy', 'dmy'))

short_dates <- c("11 December 1282", "May 1372", "1253")
parse_date_time(short_dates, orders = c('dOmY','OmY', 'Y'))
?parse_date_time

release_time = head(releases$datetime)
head(release_time)

year(release_time) %>% table()
month(release_time) %>% table()
wday(release_time, label = T) %>% table()
hour(release_time) %>% table()
am(release_time) %>% table()
ifelse(hour(release_time) < 12, paste(hour(release_time), "am", sep = ""), paste(hour(release_time) - 12, "pm", sep = ""))

###### chapter 2-2 ######
r_3_4_1 <- ymd_hms("2016-05-03 07:13:28 UTC")
floor_date(r_3_4_1, unit = 'day')
round_date(r_3_4_1, unit = '5 minutes')
ceiling_date(r_3_4_1, unit = 'week')

date_landing = mdy("February 20, 2018")
moment_stop = mdy_hms("February 19, 2018, 11:15:20", tz = "UTC")
difftime(today(), date_landing, units = 'days')
difftime(now(), moment_stop, units = 'secs')

mon_2pm <- dmy_hm("27 Aug 2018 14:00")
mon_2pm + days(1)
tue_9am <- dmy_hm("28 Aug 2018 9:00")
tue_9am + dhours(81)
today() - years(5)

###### Udemy ######
deng = read.csv("denguecases.csv")
head(deng)

dengue = aggregate(Dengue_Cases ~ Month + Year, data = deng, FUN = sum)
head(dengue)
tail(dengue)
dengue_ts = ts(dengue, start = c(2008,01), end = c(2016, 12), frequency = 12)  # Time-series
plot(dengue_ts)
start(dengue_ts)
end(dengue_ts)
frequency(dengue_ts)
window(dengue_ts)   # Extracting the subset of ts
dengue_ts2 = window(dengue_ts, start = c(2012, 01), end = c(2013, 12), frequency = 12)
plot(dengue_ts2)


gdp = read_csv("growth_gdp.csv")
head(gdp)
tail(gdp)
df = gdp %>% select(c("Time", "Country", "Value")) %>%
  filter(Country == "Germany")
df_GM = df$Value %>% ts(start = c(1985), end = c(2015), frequency = 1)
plot(df_GM)
library(changepoint)
cp = cpt.mean(df_GM)
plot(cp)


stock = read.csv('5stocks.csv')
head(stock)
tail(stock)
df = stock[, 1:2]
df_ts = ts(df, start = c(2001, 1), end = c(2017, 5), frequency = 12)
str(df_ts)
plot(df_ts)

plot.ts(df_ts[ ,2], main = "AAPL Stock Fluctuation", xlab = "Year", ylab = "ML")
cpt.mean(df_ts[ ,2]) %>% plot()

boxplot(df_ts[ ,2] ~ cycle(df_ts[ ,2]), xlab = 'Month', ylab = 'ML', main = "Monthly Variation in AAPL stocks")
