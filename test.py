import os
import findspark
import pyspark
from pyspark import SparkContext
findspark.init('/opt/spark-2.2.0-bin-hadoop2.7')

sc = SparkContext.getOrCreate()

small_ratings_file = "hdfs://20.0.9.10:54310/datasets/ratings.csv"
#os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')

small_ratings_raw_data = sc.textFile(small_ratings_file)
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]
small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()
print small_ratings_data.take(3)