library(tidyverse)
library(magrittr)
library(rebus)
library(stringr)
library(tidytext)
library(qdap)
library(tm)
library(wordcloud)

######################################################################
####################### Web Scraping #################################
library(httr)
library(rvest)
library(jsonlite)

# script = read_html(url) %>% html_node(css = '#main-content') %>% html_text() 
url = 'https://pokemondb.net/pokedex/all'
resp = read_html(url)
pokemon = html_node(x = resp, css = '#pokedex') %>% html_table()

url = 'http://www.imdb.com/search/title?year=2017&title_type=feature&'
http_type(GET(url))
movie_title = read_html(url) %>% html_nodes(css = '.lister-item-header a') %>% html_text()

######################################################################
# https://www.tidytextmining.com/
################## Natural Language Processing  ######################
library(tidytext)
library(topicmodels)
library(qdap)
library(tm)
library(wordcloud)
library(plotrix)
library(xml2)
library(rvest)

url = "https://assets.datacamp.com/production/repositories/19/datasets/27a2a8587eff17add54f4ba288e770e235ea3325/coffee.csv"
tweets = read.csv(url, stringsAsFactors = F)
str(tweets)
textData = tweets$text

textData[which(is.na(textData))] <- "NULLVALUEENTERED"
textData = str_trim(textData)
textData = tolower(textData)

corpus = VectorSource(textData) %>% Corpus()  

corpus = tm_map(corpus, removePunctuation)   
corpus = tm_map(corpus, stripWhitespace)   
corpus = tm_map(corpus, removeNumbers)  

corpus = tm_map(corpus, removeWords, stopwords('english'))    
corpus = tm_map(corpus, stemDocument)   

corpus = tm_map(corpus, removeWords, unique_low_idf[1:500])
corpus = tm_map(corpus, removeWords, c('also', 'get', 'like', 'company', 'made', 
                                       'can', 'im', 'dress', 'just', 'i'))

corpus[[15]] 
content(corpus[[15]])
corpus[[15]][1]
meta(corpus)
corpus[[15]][2]

freq = freq_terms(text.var = clean_corp, top = 10, at.least = 3, stopwords = "Top200Words")
plot(freq)

term_tdm = TermDocumentMatrix(clean_corp) 
term_dtm = DocumentTermMatrix(clean_corp)
dtm_tidy = tidy(term_dtm)
head(dtm_tidy)

findFreqTerms(term_tdm, lowfreq = 100)
term_tdm = removeSparseTerms(term_tdm, sparse = 0.9)
term_m = as.matrix(term_tdm)

term_frequency = rowSums(term_m) %>% sort(decreasing = T)
barplot(term_frequency[1:20], col = 'dark green', las = 2)

review_cleaned = tidy(dtm) %>%
  group_by(document) %>%
  mutate(text = toString(rep(term, count))) %>%
  select(document, text) %>%
  unique()

dataset = as.data.frame(term_m)

#### Text Mining ####
library(qdap)
qdap_clean = function(x){
  x = replace_abbreviation(x)
  x = replace_contraction(x)
  x = replace_number(x)
  x = replace_ordinal(x)
  x = replace_symbol(x)
  return(x)
}

contraction_clean = function(x){
  x = gsub("won't", "will not", x)
  x = gsub("can't", "can not", x)
  x = gsub("n't", "not", x)
  x = gsub("'ll", "will", x)
  x = gsub("'re", "are", x)
  x = gsub("'ve", "have", x)
  x = gsub("'m", "am", x)
  x = gsub("'d", "would", x)
  x = gsub("'s", "", x)
  return(x)
}

tm_clean = function(corpus){
  corpus = tm_map(corpus, removePunctuation)
  corpus = tm_map(corpus, stripWhitespace)
  corpus = tm_map(corpus, removeWords, stopwords('en'))
  corpus = tm_map(corpus, stemDocument)
  return(corpus)
}


library(qdap)   # -> content_transformer()
text = "<b>She</b> woke up at       6 A.M. It\'s so early!  She was only 10% awake and began drinking coffee in front of her computer."
bracketX(text)
replace_number(text)
replace_abbreviation(text)
replace_contraction(text)
replace_symbol(text)

complicate = c('complicated', 'complication', 'complicatedly')
(stem_doc = stemDocument(complicate))
stemCompletion(stem_doc, "complicate")

sentence = "In a complicated haste, Tom rushed to fix a new complication, too complicatedly"
char_vec = strsplit(sentence, split = " ") %>% unlist()
stemDocument(char_vec) %>% stemCompletion('complicate')

#### Dataframe data #### 
str(tweets)
names(tweets)[1] = 'doc_id'
df_corpus = DataframeSource(tweets) %>% VCorpus() %>% clean_corpus()
df_corpus

content(df_corpus[[1]])
meta(df_corpus[1])

#### DTM Control #### 
# N-gram tokenization
library(RWeka)
myTokenizer = function(x){
  NGramTokenizer(x, Weka_control(min = 2, max = 2))
}

bigram_dtm = clean_corpus(chardonnay_corpus) %>% 
  DocumentTermMatrix(control = list(tokenize = myTokenizer))

bigram_dtm 
char_dtm

bigram_dtm_m = as.matrix(bigram_dtm)

str(bigram_dtm_m)
freq = colSums(bigram_dtm_m)
names(freq) %>% str_subset(pattern = "^marvin")
wordcloud(words = names(freq), freq = freq, max.words = 25, colors = 'darkred')

# Term Frequency-Inverse Document Frequency
char_m = char_tdm %>% as.matrix()
char_m[c("milkshake", "little", "chocolate"), 150:157]

char_dtm2 = clean_corpus(chardonnay_corpus) %>%
  TermDocumentMatrix(control = list(weighting = weightTfIdf)) 

austen = austen_books()
head(austen)

austen %>%
  unnest_tokens(output = word, input = text) %>%
  count(book, word, sort = T) %>%
  group_by(book) %>%
  mutate(total_n = sum(n)) %>%
  bind_tf_idf(term = word, document = book, n = n) %>%
  arrange(desc(tf_idf))

######################################################################
#### Visualization #### 
library(wordcloud)
term_vec = names(term_frequency)
term_frequency[1:10]
term_vec[1:10]
wordcloud(words = term_vec, freq = term_frequency, max.words = 50, min.freq = 10,
          colors = brewer.pal(8, 'Dark2'), random.order = F)

url2 = "https://assets.datacamp.com/production/repositories/19/datasets/13ae5c66c3990397032b6428e50cc41ac6bc1ca7/chardonnay.csv"
chardonnay_tweets = read.csv(url2, stringsAsFactors = F)
all_chardonnay = paste(chardonnay_tweets$text, collapse = " ")

url = "https://assets.datacamp.com/production/repositories/19/datasets/27a2a8587eff17add54f4ba288e770e235ea3325/coffee.csv"
coffee_tweets = read.csv(url, stringsAsFactors = F)


all_coffee = paste(coffee_tweets$text, collapse = " ")
all_tweets = c(all_coffee, all_chardonnay)
all_corpus = VectorSource(all_tweets) %>% VCorpus()

all_clean = tm_clean(all_corpus)
all_clean2 = tm_map(all_clean, removeWords, c('chardonnay', 'coffee'))
all_m = TermDocumentMatrix(all_clean) %>% as.matrix()
head(all_m)
colnames(all_m) = c('coffee', 'chardonnay')
head(all_m)

#commonality cloud
commonality.cloud(term.matrix = all_m, max.words = 100, color = 'steelblue1')  

term_frequency = rowSums(all_m) %>% sort(decreasing = T)
term_frequency[1:50]

#comparison cloud
comparison.cloud(term.matrix = all_m, max.words = 100, colors = c('orange', 'blue'))  

#polarized cloud
top25 = all_m %>%
  as_data_frame(rownames = "word") %>%
  filter_all(all_vars(. > 0)) %>%
  mutate(diff = chardonnay - coffee) %>%
  top_n(n = 25, wt = diff) %>%
  arrange(desc(diff))

library(plotrix)
pyramid.plot(lx = top25$chardonnay, rx = top25$coffee, labels = top25$word, 
             top.labels = c('Chardonnay', 'Words', 'Coffee'), main = "Words in Common", gap = 20)  

#Network plot
word_associate(coffee_tweets$text, match.string = 'barista', network.plot = T, 
               stopwords = c(Top200Words, 'coffee', 'amp'), cloud.colors = c('gray85', 'darkred')) 
title(main = 'Barista Coffee Tweet Associations')

word_associate(chardonnay_tweets$text, match.string = c('marvin'), network.plot = T,
               stopwords = c(Top200Words, 'chardonnay', 'amp'), cloud.colors = c('gray85', 'darkred'))  # ??


#Dendrogram 
chardonnay_tweets = read.csv(url2, stringsAsFactors = F)
chardonnay_corpus = VectorSource(chardonnay_tweets$text) %>% VCorpus()
char_tdm = clean_corpus(chardonnay_corpus) %>% TermDocumentMatrix()
dim(char_tdm)
char_m = removeSparseTerms(char_tdm, sparse = .975) %>% as.matrix()

term_m = char_m
d = dist(scale(text_m))
hc = hclust(d, method = 'complete')
plot(hc)


library(dendextend)
hcd = as.dendrogram(hc)
plot(hcd)
color_branches(hcd, h = 10) %>% plot()

labels(hcd)
hcd_colored = branches_attr_by_labels(dend = hcd, labels = c('marvin', 'gaye'), color = 'red')
plot(hcd_colored, main = 'Better Dendrogram')
rect.dendrogram(tree = hcd_colored, k = 2, border = 'grey50')

#Association 
associations = findAssocs(x = char_tdm, terms = 'rose', corlimit = .2)
associations
class(associations)

associations_df = list_vect2df(associations, col2 = 'word', col3 = 'score')
associations_df %>% ggplot(aes(score, word)) + geom_point(size = 3)


#JavaScript radar chart ####
books = readRDS('all_books.rds')
table(books$book)

moby_huck = books %>% 
  filter(grepl('huck|moby', book))

nrc = get_sentiments('nrc')
scores = moby_huck %>%
  inner_join(nrc, by = c('term' = 'word')) %>%
  filter(!grepl('positive|negative', sentiment)) %>%
  count(book, sentiment) %>%
  spread(book, n)

install.packages("radarchart")
library(radarchart)
chartJSRadar(scores)

#Treemap ####
book_length = books %>%
  count(book)

book_tree = books %>%
  inner_join(get_sentiments('afinn'), by = c('term' = 'word')) %>%
  group_by(author, book) %>%
  summarize(mean_score = mean(score)) %>%
  inner_join(book_length)
  
install.packages("treemap")
library(treemap)
treemap(book_tree, index = c('author', 'book'), 
        vSize = 'n', vColor = 'mean_score', type = 'value', pallete = c('red', 'white', 'green'))

######################################################################
#### Tiding sentimental data #### 
library(tidytext)
geocoded_tweets 
(nrc = get_sentiments('nrc'))
(bing = get_sentiments('bing'))
(afinn = get_sentiments('afinn'))

tweets_nrc = geocoded_tweets %>% inner_join(nrc)
head(tweets_nrc)
table(tweets_nrc$sentiment)

joy_words = tweets_nrc %>%
  filter(sentiment == 'joy') %>%
  group_by(word) %>%
  summarise(avg_freq = mean(freq)) %>%
  arrange(desc(avg_freq))

library(ggplot2)
joy_words %>%
  top_n(20) %>%
  ggplot(aes(x = word, y = avg_freq)) +
  geom_col() + coord_flip()

tweets_bing = geocoded_tweets %>% inner_join(bing)
head(tweets_bing)
tweets_bing %>%
  group_by(state, sentiment) %>%
  summarise(avg_freq = mean(freq)) %>%
  spread(sentiment, avg_freq) %>%
  ungroup() %>%
  mutate(ratio = positive / negative, 
         state = reorder(state, ratio)) %>%    
  ggplot(aes(x = state, y = ratio)) + geom_point() + coord_flip()


shakespeare %>% head()
shakespeare %>% count(title, type)
tidy_shakespeare = shakespeare %>%
  group_by(title) %>% 
  mutate(linenumber = row_number()) %>%        # row_number()
  unnest_tokens(output = word, input = text) %>% 
  ungroup()

tidy_shakespeare %>%
  count(title, word, sort = T) %>%
  inner_join(get_sentiments(lexicon = 'afinn')) %>%
  filter(title == 'The Tragedy of Macbeth',
         score < 0)

tidy_shakespeare %>%
  inner_join(bing) %>%
  count(title, type, index = linenumber %/% 70, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%  
  mutate(sentiment = positive - negative) %>%
  ggplot(aes(x = index, y = sentiment, fill = type)) + geom_col() + facet_wrap(~title, scales = 'free_x')


climate_text
tidy_tv = climate_text %>% unnest_tokens(input = text, output = word)

tidy_tv %>%
  anti_join(stop_words) %>% 
  count(word)

tv_sentiment %>%
  filter(sentiment == 'negative') %>%
  count(word, station) %>%
  group_by(station) %>%
  top_n(10) %>%
  ungroup() %>%
  mutate(word = reorder(paste(word, station, sep = '__'), n)) %>%
  ggplot(aes(x = word, y = n, fill = station)) + geom_col(show.legend = F) +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) +
  facet_wrap(~ station, nrow = 2, scales = 'free') + coord_flip()


#sentimental change over time
library(lubridate)
sentiment_by_time = tidy_tv %>% 
  mutate(date = floor_date(show_date, unit = '6 months')) %>%
  group_by(date) %>%
  mutate(total_words = n()) %>%
  ungroup() %>%
  inner_join(get_sentiments('nrc'))

sentiment_by_time %>%
  filter(sentiment %in% c('positive', 'negative')) %>%
  count(date, sentiment, total_words) %>%
  mutate(percent = n/total_words) %>%
  ggplot(aes(x = date, y = percent, color = sentiment)) +
  geom_line(size = 1.5) + 
  geom_smooth(method = 'lm', se = F, lty = 2) + expand_limits(y = 0)

library(tidytext)
tidy_lyrics = song_lyrics %>% unnest_tokens(output = word, input = lyrics)

lyric_sentiment = tidy_lyrics %>%
  group_by(song) %>%
  mutate(total_words = n()) %>%
  ungroup() %>%
  inner_join(nrc)

#the proportion of negative words for each songs
lyric_sentiment %>%
  count(song, sentiment, total_words) %>%
  ungroup() %>%
  mutate(percent = n / total_words) %>%
  filter(sentiment == 'negative') %>%
  arrange(desc(percent))

#the distribution of the percentage of positive (or negative) words by ranks. Is there any trend? 
lyric_sentiment %>%
  filter(sentiment == 'positive') %>%
  count(song, rank, total_words) %>%
  mutate(percent = n / total_words,
         rank = floor(rank/10)*10) %>%
  ggplot(aes(x = factor(rank), y = percent)) + geom_boxplot()

#the sentimental change of negative words over time 
lyric_sentiment %>%
  filter(sentiment == 'negative') %>%
  count(song, year, total_words) %>%
  mutate(percent = n / total_words, 
         year = floor(year/10)*10) %>%
  ggplot(aes(x = factor(year), y = percent)) + geom_boxplot()

#the proportion of positive words versus negative words for each book
library(dplyr)
books = readRDS('all_books.rds')
books_sent_count = books %>%
  inner_join(get_sentiments('nrc'), by = c('term' = 'word')) %>%
  filter(grepl('positive|negative', sentiment)) %>%
  count(book, sentiment)

book_pos = books_sent_count %>%
  group_by(book) %>%
  mutate(percent_positive = n / sum(n) * 100)

ggplot(book_pos, aes(x = book, y = percent_positive, fill = sentiment)) +
  geom_col() + coord_flip()


#### Sentiment lexicon Anaysis ####
str(bos_reviews)
dim(bos_reviews)

practice_pol = bos_reviews %>% head() %$% polarity(comments)
summary(practice_pol$all$polarity)

bos_pol = readRDS('bos_pol.rds')
summary(bos_pol$all$polarity)

ggplot(bos_pol$all, aes(x = polarity, y = ..density..)) +
  geom_histogram(binwidth = .25, fill = '#bada55', colour = 'grey60') +
  geom_density(size = .75) +
  theme_gdocs()

#make reviews into words making tidy data and sentiment analysis 
tidy_reviews = bos_reviews %>% unnest_tokens(output = word, input = comments)
head(tidy_reviews)
tidy_reviews = tidy_reviews %>%
  group_by(id) %>%
  mutate(origin_word_order = seq_along(word)) %>%
  anti_join(stop_words)

bing = get_sentiments('bing')

pos_neg = tidy_reviews %>%
  inner_join(bing) %>% 
  count(sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(polarity = positive - negative) 
summary(pos_neg)
head(tidy_reviews)
head(pos_neg)

#the tendency between effort and sentiment 
pos_neg_pol = tidy_reviews %>% 
  count(id) %>%
  inner_join(pos_neg) %>%
  mutate(pol = ifelse(polarity >= 0, 'Positive', 'Negative'))

pos_neg_pol %>%
  ggplot(aes(x = polarity, y = n, color = pol)) +
  geom_point(alpha = .25) +
  geom_smooth(method = 'lm', se = F) + theme_gdocs() +
  ggtitle("Relationship between word effort & polarity")


#separate reviews into positive and negative ones
pos_terms = bos_reviews %>%
  mutate(polarity = bos_pol$all$polarity) %>%
  filter(polarity > 0) %>%
  pull(comments) %>%
  paste(collapse = ' ') %>%
  str_trim()
neg_terms = bos_reviews %>%
  mutate(polarity = bos_pol$all$polarity) %>%
  filter(polarity < 0) %>% 
  pull(comments) %>% 
  paste(collapse = ' ') %>%
  str_trim()

#'grade inflation' of polarity scores
bos_reviews$scaled_polarity = scale(bos_pol$all$polarity)
bos_reviews_pos = subset(bos_reviews, scaled_polarity > 0)
bos_reviews_neg = subset(bos_reviews, scaled_polarity < 0)

pos_terms = paste(bos_reviews_pos, collapse = ' ')
neg_terms = paste(bos_reviews_neg, collapse = ' ')

all_corpus = c(pos_terms, neg_terms) %>% 
  VectorSource() %>% VCorpus()

all_tdm = TermDocumentMatrix(x = all_corpus, control = list(weighting = weightTfIdf, 
                                                            removePunctuation = T,
                                                            stopwords = stopwords(kind = 'en')))
all_tdm_m = as.matrix(all_tdm)
colnames(all_tdm_m) = c('Positive', 'Negative')
library(wordcloud)
comparison.cloud(term.matrix = all_tdm_m, max.words = 15, colors = c('darkgreen', 'darkred'))


######################################################################
#### Word2Vec ####
# https://github.com/bmschmidt/wordVectors/blob/master/vignettes/introduction.Rmd
# https://gist.github.com/primaryobjects/8038d345aae48ae48988906b0525d175#file-1-word2vec-r-L63
library(wordVectors)
library(magrittr)

download.file(url = 'http://archive.lib.msu.edu/dinfo/feedingamerica/cookbook_text.zip', destfile = 'cookbooks.zip')
unzip('cookbooks.zip', exdir = 'cookbooks')

prep_word2vec(origin = 'cookbooks', destination = 'cookbooks.txt', lowercase = T, bundle_ngrams = 2)

vsm = train_word2vec(train_file = 'cookbooks.txt', output_file = 'cookbook_vectors.bin', vectors = 200, 
                       threads = 4, iter = 3, negative_samples = 0)

vsm = as.data.frame(w2vec)

# Getting words close to the keyword
closest_to(matrix = vsm, vector = 'fish')
some_fish = closest_to(matrix = vsm, vector = w2vec[[c("fish","salmon","trout","shad","flounder","carp","roe","eels")]], n = 50)
str(some_fish)

fishy = vsm[[some_fish$word, average = F]]

vsm[1:1000, ] %>% cosineSimilarity('salmon')
plot(fishy, method = 'pca')

# Visualization
plot(vsm, perplexity = 50)

tastes = vsm[c('sweet', 'salty'), average = F]
sweet_salty = vsm[1:3000, ] %>% cosineSimilarity(tastes)

sweet_salty_close = sweet_salty[rank(-sweet_salty[, 1]) < 20 | rank(-sweet_salty[, 2] < 20)]
plot(sweet_salty_close, type = 'n')
text(sweet_salty_close, labels = rownames(sweet_salty_close))

high_similar_ss = sweet_salty[rank(-apply(sweet_salty, 1, max)) < 50, ]
high_similar_ss %>%
  prcomp() %>%
  biplot()

# Clustering with Word2Vec
keywords = c('madeira', 'beef', 'carrot')
sub = lapply(keywords, functions(x){
  nearest_words = closest_to(vsm[[x]], n = 20)
  return(nearest_words$word)
  }) %>% unlist()

subWords = vsm[[sub, average = F]]
d = subWords %>%
  cosineDist(subWords) %>%
  as.dist()
model_hc = hclust(d)
plot(model_hc)

model = kmeans(x = vsm, centers = 10, iter.max = 50)


#### text2vec ####
# https://srdas.github.io/MLBook/Text2Vec.html
# http://text2vec.org
# https://www.tidytextmining.com/topicmodeling.html
# https://cran.r-project.org/web/packages/text2vec/vignettes/text-vectorization.html#text_analysis_pipeline
library(text2vec)
data("movie_review")
str(movie_review)
movie = movie_review[1:500, ]

prep_fun = function(x){
  x = tolower(x)
  x = str_replace_all(x, '[^[:alnum:]]', ' ')
  x = str_replace_all(x, '\\s+', ' ')
}
movie$review_cleaned = prep_fun(movie$review)

# define 2 sets of documents 
doc_set_1 = movie[1:300, ]
it1 = itoken(iterable = doc_set_1$review_cleaned)

doc_Set_2 = movie[301:500, ]
it2 = itoken(doc_Set_2$review_cleaned)

# define common space and project vocabulary-based vectorization for better interpretability
it = itoken(movie$review_cleaned)
v = create_vocabulary(it) %>%
  prune_vocabulary(doc_proportion_max = .1, term_count_min = 5)
vectorizer = vocab_vectorizer(v)

# casting into dtm 
dtm1 = create_dtm(it = it1, vectorizer = vectorizer)
dtm2 = create_dtm(it = it2, vectorizer)
dim(dtm1)
dim(dtm2)

# calculate similarity between x & y
d1_d2_jac_sim = sim2(x = dtm1, y = dtm2, method = 'jaccard', norm = 'none')

dim(d1_d2_jac_sim)
d1_d2_jac_sim[1:2, 1:5]

# parrallel similarity
d1_d2_jac_psim = psim2(x = dtm1[1:200, ], y = dtm2, method = 'jaccard', norm = 'none')
str(d1_d2_jac_psim)

# cosine similarity
d1_d2_cos_sim = sim2(x = dtm1, y = dtm2, method = 'cosine', norm = 'l2')
dim(d1_d2_cos_sim)
d1_d2_cos_sim[1:2, 1:5]

# casting into dtm with Tf-Idf
dtm = create_dtm(it, vectorizer)
tfidf = TfIdf$new()
dtm_tfidf = fit_transform(dtm, tfidf)

d1_d2_tfidf_cos_sim = sim2(x = dtm_tfidf, method = 'cosine', norm = 'l2')
d1_d2_tfidf_cos_sim[1:2, 1:5]

# applying LSA model
lsa = LSA$new(n_topics = 100)
dtm_tfidf_lsa = fit_transform(dtm, lsa)
dim(dtm_tfidf_lsa)

d1_d2_tfidf_cos_sim = sim2(x = dtm_tfidf_lsa, method = 'cosine', norm = 'l2')
d1_d2_tfidf_cos_sim[1:2, 1:5]

