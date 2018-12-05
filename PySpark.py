from  pyspark.sql import SparkSession

my_spark = SparkSession.builder.getOrCreate()
print(my_spark)

# Print the tables in the catalog
spark.catalog.listTables()

stmt = 'FROM flights SELECT * LIMIT 10'
movie = spark.sql(stmt)
movie.show()

movie_df = movie.toPandas()    # spark -> pd
movie_df.head()
# Add spark_temp to the catalog
movie_spark_temp = spark.createDataFrame(movie_df)    # pd -> spark: temporary local table
spark.catalog.listTables()

##############################################
# importing data
spark = SparkSession.builder.getOrCreate()
airport = spark.read.csv(file_path)
fg = spark.table('flights')
fg
fg.show()

# add new column
fg = fg.withColumn('flight_hrs', fg.air_time/60)
fg = fg.withColumnRenamed('dist', 'distance')

# SELECT
fg_sub = fg.select('tailnum', 'origin' 'dest', 'dist')
fg_sub = fg.select(fg.tailnum, fg.origin, fg.dest, fg.dist)
fg_sub.filter(fg.origin == 'SEA').filter('dist > 1000')

avg_speed = (fg.dist / (fg.air_time/60)).alias('avg_speed')
fg_sub2 = fg.select('tailnum', 'origin', 'dest', avg_speed)
fg_sub2 = fg.selectExpr('tailnum', 'origin', 'dest', 'dist/(air_time/60) as avg_speed')

# GROUP BY
fg_sub.filter(fg.origin == 'SEA').groupBy().max('air_time').show()
fg.withColumn('flight_hrs', fg.air_time/60).groupBy().avg('flight_hrs').show

fg_sub.groupBy('tailnum').count()

import pyspark.sql.functions as F
fg.groupBy('tailnum').agg(F.stddev('avg_speed'))

# JOIN
fg_plane = fg.join(plane, on = 'tailnum', how = 'leftouter')

##############################################
############# feature engineering
model_data = model_data.withColumn('air_time', model_data.air_time.cast('integer'))
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)
model_data = model_data.withColumn('plane_age', model_data.year*model_data.dist)

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL")

# string factors
carr_indexer = StringIndexer(inputCol = 'carrier', outputCol = 'carrier_index')
carr_encoder = OneHotEncoder(inputCol = 'carrier_index', outputCol = 'carrier_fact')

dest_indexer = StringIndexer(inputCol= 'dest', outputCol = 'dest_index')
dest_encoder = OneHotEncoder(inputCol = 'dest_index', outputCol = 'dest_fact')

# combine all features
vec_assembler = VectorAssembler(inputCols= ['air_time', 'is_late', 'plane_age', 'carrier_fact', 'dest_fact'], outputCol= 'features')

# pipeline
from pyspark.ml import Pipeline
pipe = Pipeline(stages = [carr_indexer, carr_encoder, dest_indexer, dest_encoder, vec_assembler])

data_piped = pipe.fit(model_data).transform(model_data)

# splitting into train, test set
train, test = piped_data.randomSplit([.7, .3])

############# fitting models
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression()

import pyspark.ml.evaluation as evals
evaluator = evals.BinaryClassificationEvaluator(metricName = 'areaUnderROC')

############# model tunning
import pyspark.ml.tuning as tune
# create the parameter grid
grid = tune.ParamGridBuilder()

# add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# build the gridz
grid = grid.build()

# create the CrossValidator
cv = tune.CrossValidator(estimator = lr,
                         estimatorParamMaps = grid,
                         evaluator = evaluator)

# fit cross validation models
models = cv.fit(train)
# extract the best model
best_lr = models.bestModel

pred = best_lr.transform(test)
print(evaluator.evaluate(pred))

##############################################
