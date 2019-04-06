# Print the tables in the catalog
spark.catalog.listTables()

stmt = 'FROM flights SELECT * LIMIT 10'
movie = spark.sql(stmt)
movie.show()

movie_df = movie.toPandas()     # spark -> pd
movie_df.head()
# Add spark_temp to the catalog
movie_spark_temp = spark.createDataFrame(movie_df)     # pd -> spark: temporary local table
spark.catalog.listTables()

#####################################################################
import findspark
findspark.init('/home/jjone/spark-2.3.1-bin-hadoop2.7')

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('basic').getOrCreate()
spark = SparkSession.builder.getOrCreate()

# importing data
airport = spark.read.csv('appl_stock.csv', inferSchema = True, header = True)
fg = spark.table('flights')
fg
fg.printSchema()
fg.show()

# add new column
fg = fg.withColumn('flight_hrs', fg.air_time/60)
fg = fg.withColumnRenamed('dist', 'distance')

# asDict()
fg.filter(fg.dist > 100).show()
result = fg.filter(fg.dist > 100).collect()
result[0].asDict()

# SELECT
fg_sub = fg.select('tailnum', 'origin' 'dest', 'dist')
fg_sub = fg.select(fg.tailnum, fg.origin, fg.dest, fg.dist)

fg_sub.filter(fg.origin == 'SEA').filter('dist > 1000')

avg_speed = (fg.dist / (fg.air_time/60)).alias('avg_speed')
fg_sub2 = fg.select('tailnum', 'origin', 'dest', avg_speed)
fg_sub2 = fg.selectExpr('tailnum', 'origin', 'dest', 'dist/(air_time/60) as avg_speed')

# GROUP BY
fg_sub.filter(fg.origin == 'SEA').groupBy().max('air_time').show()
fg.withColumn('flight_hrs', fg.air_time/60).groupBy().avg('flight_hrs').show()

fg_sub.groupBy('tailnum').count()

import pyspark.sql.functions as F
fg.groupBy('tailnum').agg(F.stddev('avg_speed'))

fg.select(F.stddev('dist')).alias('std dist').select(F.format_number('std dist', 3)).show()

# ORDER BY
fg.orderBy(fg.dist).show()
fg.orderBy(fg['dist'].desc()).show()

# JOIN
fg_plane = fg.join(plane, on = 'tailnum', how = 'leftouter')

#  missing values
data = data.filter("arr_delay is not NULL and dep_delay is not NULL")

data.na.drop(thresh = 5).show()
data.na.drop(how = 'any')   # 'all'
data.na.drop(subset = ['dist'])

data.na.fill(0).show()
data.na.fill('No Info', subset = ['origin'])

import pyspark.sql.functions as F
dist_avg = fg.select(F.mean(fg['dist'])).collect()
dist_avg = dist_avg[0][0]
data = data.na.fill(dist_avg, subset = ['dist'])

# DateTime
from pyspark.sql.functions import weekofyear, dayofyear, dayofmonth, year, month, hour, date_format, format_number
data.select(month(df['date'], dayofmonth(df['date'])).show()
data = data.withColumn('year', year(df['date']))

# correlation
from pyspark.sql.functions import corr
data.select(corr('dist', 'time')).show()

#####################################################################
############# feature engineering
# adding new columns
data = data.withColumn('air_time', data.air_time.cast('integer'))
data = data.withColumn("is_late", data.arr_delay > 0)
data = data.withColumn('plane_age', data.year*data.dist)

# string factors encoding
from pyspark.ml.feature import StringIndexer, OneHotEncoder
carr_indexer = StringIndexer(inputCol = 'carrier', outputCol = 'carrier_index')
carr_encoder = OneHotEncoder(inputCol = 'carrier_index', outputCol = 'carrier_fact')

dest_indexer = StringIndexer(inputCol= 'dest', outputCol = 'dest_index')
dest_encoder = OneHotEncoder(inputCol = 'dest_index', outputCol = 'dest_fact')

# combine all columns into one feature
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCol= ['air_time', 'is_late', 'plane_age', 'carrier_fact', 'dest_fact'],
                            outputCol= 'features')

# pipeline
from pyspark.ml import Pipeline
pipe = Pipeline(stages = [carr_indexer, carr_encoder, dest_indexer, dest_encoder, assembler])

data_piped = pipe.fit(data).transform(data)
data_piped.show()
data_piped = data_piped.select('features', 'Survived')

# splitting into train, test set
tr, te = data_piped.randomSplit([.7, .3])

# fitting models
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'Survived')

model = lr.fit(tr)
pred = model.transform(te)

import pyspark.ml.evaluation as evals
evaluator = evals.BinaryClassificationEvaluator(rawPredictionCol = 'prediction', labelCol = 'Survived')
AUC = evaluator.evaluate(pred)
AUC

############# model tunning
from pyspark.ml.tuning import ParamGridBuilder
grid = ParamGridBuilder()
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])
grid = grid.build()

# create the CrossValidator
from pyspark.ml.tuning import CrossValidator
cv = CrossValidator(estimator = lr,
                    estimatorParamMaps = grid,
                    evaluator = evaluator)

models = cv.fit(tr)
# extract the best model
best_model = models.bestModel
pred = best_model.transform(te)
print(evaluator.evaluate(pred))

#####################################################################
# https://spark.apache.org/docs/latest/ml-guide.html
############# 1) Linear Regression
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol = 'Survived', predictionCol = 'prediction')
model = lr.fit(tr)

model.coefficients
model.intercept

result = model.evaluate(te)
result.residauls.show()

model.summary.r2
model.summary.rootMeanSquaredError

############# 2) Logisitc Regression
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'Survived')
model = lr.fit(tr)
pred = model.transform(te)

# evaluation
model.summary.predictions.printSchema
model.summary.predictions.describe().show()

import pyspark.ml.evaluation as evals
evaluator = evals.BinaryClassificationEvaluator(rawPredictionCol = 'prediction', labelCol = 'Survived', metricName = 'areaUnderROC')
auc = evaluator.evaluate(model.summary.predictions)
auc

result = model.evaluate(te)
result.show()
result.predictions.show()

############# 3) Random Forest
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
dtc = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'Survived')
gbt = GBTClassifier(featuresCol = 'features', labelCol = 'Survived')

rfc = RandomForestClassifier(featuresCol = 'features', labelCol = 'Survived', numTrees = 300)
model = rfc.fit(tr)
pred = model.transform(te)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol = 'Survived', metricName = 'accuracy')
evaluator.evaluate(pred)

model.featureImportances

############# 4) K-means Clustering
# combine all columns into one feature
from pyspark.ml.feature import VectorAssembler
assembler2 = VectorAssembler(inputCol = data.columns, outputCol = 'features')
data = assembler2.transform(data)

# standardize values
from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol = 'features', outputCol = 'scaled_features')
data_scaled = scaler.fit(data).transform(data)

from pyspark.ml.clustering import kMeans
kmeans = kMeans(featuresCol = 'scaled_features', k = 4)
model = kMeans.fit(data_scaled)

wcss = model.computeCost(data_scaled)
centers = model.clusterCenters()

result = model.transform(data_scaled)
result.show()
result.groupBy('prediction').count().show()

#####################################################################
############# Natural Language Preprocessing
from pyspark.ml.feature import Tokenizer, RegexTokenizer
tokenizer = Tokenizer(inputCol = 'text', outputCol = 'token')
tokenizer2 = RegexTokenizer(inputCol = 'text', outputCol = 'token', pattern = '#\w+')

data_token = tokenizer.transform(data)
data_token.show()

# n_words
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
count_token = udf(lambda token: len(token), IntegerType())
data_token.withColumn('n_token', count_token(col('token'))).show()

# stopwords
from pyspark.ml.feature import StopWordsRemover
remover = StopWordsRemover(inputCol = 'token', outputCol = 'filtered', stopwords = ['aaa'])
data_2 = remover.transform(data_token)

# N-gram
from pyspark.ml.feature import NGram
ngram = NGram(n = 2, inputCol = 'token', outputCol = 'bigram')
ngram.transform(data_token).select('bigram').show(truncate = False)

# Tfidf
from pyspark.ml.feature import HashingTF, IDF
hasing_tf = HashingTF(inputCol = 'tokens', outputCol = 'rawfeatures')
data_token = hasing_tf.transform(data_token)

idf = IDF(inputCol = 'rawfeatures', outputCol = 'features')
idf_model = idf.fit(data_token)

data_tfidf = idf_model.transform(data_token)
data_tfidf.show()

# CountVectorizer
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol = 'token', outputCol = 'features', vocabSize, minDF = 2)

result = cv.fit(data_token).transform(data_token)
result.show()

############# model fitting
tokenizer = Tokenizer(inputCol = 'text', outputCol = 'token')
remover = StopWordsRemover(inputCol = 'token', outputCol = 'token_stop')
cv = CountVectorizer(inputCol = 'token_stop', outputCol = 'token_cv')
idf = IDF(inputCol = 'token_cv', outputCol = 'token_tfidf')
assembler = VectorAssembler(inputCol = ['token_tfidf', 'length'], outputCol = 'features')

indexer_y = StringIndexer(inputCol = 'Survived', outputCol = 'label')

from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes()

pipe = Pipeline(stages = [indexer_y, tokenizer, remover, cv, idf, assembler])

nlp_cleaner = pipe.fit(data)
data_cleaned = nlp_cleaner.transform(data)
data_cleaned.show()
data_2 = data_cleaned.select('label', 'features')

tr, te = data_2.randomSplit([.7, .3])
model = nb.fit(tr)
pred = model.transform(te)

evaluator = MulticlassClassificationEvaluator()
acc = evaluator.evaluate(pred)


#####################################################################
# https://www.datacamp.com/courses/recommendation-engines-in-pyspark
############# Collaborative Filtering
mo.printSchema()
mo = mo.select(mo.UserId.cast('integer'), mo.MovieId.cast('integer'), mo.rating.cast('double'))
mo.show()

# coverting data into row-based dataframe
from pyspark.sql.functions import array, col, explode, lit, struct
def to_long(df, by = ["userId"]): # "by" is the column by which you want the final output dataframe to be grouped by
    cols = [c for c in df.columns if c not in by]
    kvs = explode(array([struct(lit(c).alias("movieId"), col(c).alias("rating")) for c in cols])).alias("kvs")
    long_df = df.select(by + [kvs]).select(by + ["kvs.movieId", "kvs.rating"]).filter("rating IS NOT NULL")
    # Excluding null ratings values since ALS in Pyspark doesn't want blank/null values
    return long_df

mo = to_long(mo)

# sparsity
n_rating = mo.select('rating').count()
n_users = mo.select('UserId').distinct().count()
n_movies = mo.select('MovieId').distinct().count()
sparsity = (1 - (n_rating * 1.0 / (n_users * n_movies)))*100
print("The dataframe is ", "%.2f" %sparsity + " % empty")

# preprocessing
from pyspark.sql.functions import monotonically_increasing_id

# get unique users and repartition to 1 partition
users = mo.select("UserId").distinct().coalesce(1)
# create a new column of unique integers called "userId" in the users dataframe.
users = users.withColumn("user", monotonically_increasing_id()).persist()

movies = mo.select('MovieId').distinct().coalesce(1)
movies = movies.withColumn('movie', monotonically_increasing_id()).persist()
mo = mo.join(users, on = 'UserId', how = 'left').join(movies, on = 'MovieId', how = 'left')

# split into train and test set
tr, te = mo.randomSplit([.7, .3])

# fit the ALS model
import pyspark.ml.recommendation import ALS
als = ALS(userCol = 'user', itemCol = 'movie', ratingCol = 'rating',
          rank = 25,  # the number of latent feactures
          maxIter = 100,  # iteration numbers
          regParam = .01,  # lambda
          # alpha = 40,
          implicitPrefs = False,
          nonnegative = True,
          coldStartStrategy = 'drop')

model = als.fit(tr)
pred = model.transform(te)
pred.show()

# evalutaion
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol = 'rating', predictionCol = 'prediction', metricName = 'rmse')
rmse = evaluator.evaluate(pred)
print('RMSE score is: ' + rmse)

# recommendation
from pyspark.sql.functions import col
single_user = te.filter(col('userId') == 15).select(['movieId', 'userId'])
single_user.show()
recommendations = model.transform(single_user)
recommendations.orderBy('prediction', ascending = False).show()

recommendations.filter(col('userId') == 15).show()

############# hyperparameter tuning
from pyspark.ml.tuning import ParamGridBuilder
grid = ParamGridBuilder()
grid = grid.addGrid(als.rank, np.arange(5, 40, 80, 120))
           .addGrid(als.maxIter, [100, 200, 300])
           .addGrid(als.regParam, np.arange(0, .1, .02))
           .build()

import pyspark.ml.recommendation import ALS
als = ALS(userCol = 'user', itemCol = 'movie', ratingCol = 'rating',
          implicitPrefs = False,
          nonnegative = True,
          coldStartStrategy = 'drop')

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol = 'rating', predictionCol = 'prediction', metricName = 'rmse')

from pyspark.ml.tuning import CrossValidator
cv = CrossValidator(estimator = als,
                    estimatorParamMaps = grid,
                    evaluator = evaluator,
                    numFolds = 5)

best_model = cv.fit(tr).bestModel
best_model.rank

pred = best_model.transform(te)
rmse = evaluator.evaluate(pred)

#####################################################################
# https://campus.datacamp.com/courses/recommendation-engines-in-pyspark/recommending-movies?ex=12
# https://campus.datacamp.com/courses/recommendation-engines-in-pyspark/what-if-you-dont-have-customer-ratings?ex=11
############# Implicit ratings
users = Z.select("userId").distinct()
products = Z.select("productId").distinct()
cj = users.crossJoin(products)
cj.show()
Z_2 = cj.join(Z, on = ['userId', 'productId'], how = 'left').fillna(0)
Z_2.show()

############# Rank Ordering Error Metric
# https://github.com/jamenlong/ALS_expected_percent_rank_cv/blob/master/ROEM_cv.py
def ROEM(predictions, userCol = "userId", itemCol = "songId", ratingCol = "num_plays"):
    #Creates table that can be queried
    predictions.createOrReplaceTempView("predictions")
    #Sum of total number of plays of all songs
    denominator = predictions.groupBy().sum(ratingCol).collect()[0][0]
    #Calculating rankings of songs predictions by user
    spark.sql("SELECT " + userCol + " , " + ratingCol + " , PERCENT_RANK() OVER (PARTITION BY " + userCol + " ORDER BY prediction DESC) AS rank FROM predictions").createOrReplaceTempView("rankings")
    #Multiplies the rank of each song by the number of plays and adds the products together
    numerator = spark.sql('SELECT SUM(' + ratingCol + ' * rank) FROM rankings').collect()[0][0]
    performance = numerator/denominator
    return performance

# Split the data into training and test sets
(training, test) = msd.randomSplit([0.8, 0.2])

#Building 5 folds within the training set.
train1, train2, train3, train4, train5 = training.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed = 1)
fold1 = train2.union(train3).union(train4).union(train5)
fold2 = train3.union(train4).union(train5).union(train1)
fold3 = train4.union(train5).union(train1).union(train2)
fold4 = train5.union(train1).union(train2).union(train3)
fold5 = train1.union(train2).union(train3).union(train4)

foldlist = [(fold1, train1), (fold2, train2), (fold3, train3), (fold4, train4), (fold5, train5)]

# Empty list to fill with ROEMs from each model
ROEMS = []

# Loops through all models and all folds
for model in model_list:
    for ft_pair in foldlist:

        # Fits model to fold within training data
        fitted_model = model.fit(ft_pair[0])

        # Generates predictions using fitted_model on respective CV test data
        predictions = fitted_model.transform(ft_pair[1])

        # Generates and prints a ROEM metric CV test data
        r = ROEM(predictions)
        print ("ROEM: ", r)

    # Fits model to all of training data and generates preds for test data
    v_fitted_model = model.fit(training)
    v_predictions = v_fitted_model.transform(test)
    v_ROEM = ROEM(v_predictions)

    # Adds validation ROEM to ROEM list
    ROEMS.append(v_ROEM)
    print ("Validation ROEM: ", v_ROEM)

import numpy
# Find the index of the smallest ROEM
i = numpy.argmin(ROEMS)
print ("Index of smallest ROEM:", i)
print ("Smallest ROEM: ", ROEMS[i])
