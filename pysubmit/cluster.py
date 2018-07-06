import os
import findspark
import pyspark
from time import time
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import math

sc = SparkContext.getOrCreate()
#datasets_path = os.path.join('..', 'datasets')
findspark.init('/opt/spark-2.2.0-bin-hadoop2.7')

#small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')
#Load Ratings from small to RDD
hdfs="hdfs://"
ip="20.0.9.21"

# Setup variables for training
# GOAL find Best_rank and minimize error it will be used to train Large Dataset
seed = 5L
tolerance = 0.1

best_rank = 4
iterations = 5
regularization_parameter = 0.1

#Finally our goal is achieved now we can train our large dataset

# Load the complete dataset ratings file
#complete_ratings_file = os.path.join(datasets_path, 'ml-latest', 'ratings.csv')
#complete_ratings_file ="hdfs://20.0.9.10:54310/datasets/ml-latest/ratings.csv"
complete_ratings_file =hdfs+ip+":54310/datasets/ml-latest/ratings.csv"
complete_ratings_raw_data = sc.textFile(complete_ratings_file)
#Delete header because it's not necesary
complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

# Parse from HDFS .csv to a RDD
complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
    
print "There are %s recommendations in the complete dataset" % (complete_ratings_data.count())

tinicio = time()
# Now we divide our Large Dataset Data in 70% for Training and 30% for Testing
training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0L)
#=>>>> Here we train
complete_model = ALS.train(training_RDD, best_rank, seed=seed, 
                           iterations=iterations, lambda_=regularization_parameter)
#Here we take only UserID and Movie ID from our test_RDD
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
#We predict all for our test_RDD
predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
#Join our Predictions and Real rates to find RMSE
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
print 'For testing data the RMSE is %s' % (error)

# Load the complete dataset movies file to get Movie_Name from Movie_ID
#complete_movies_file = os.path.join(datasets_path, 'ml-latest', 'movies.csv')
#complete_movies_file ="hdfs://20.0.9.10:54310/datasets/ml-latest/movies.csv"
complete_movies_file =hdfs+ip+":54310/datasets/ml-latest/movies.csv"
complete_movies_raw_data = sc.textFile(complete_movies_file)
complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]
# Parse
complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]),x[1]))
print "There are %s movies in the complete dataset" % (complete_movies_titles.count())

#We need to get the number of rates per movie so we calculate it
def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

#Now we are Going to create a new user with ID = 0 beacuse it doesn't exist.
new_user_ID = 0
# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,260,5), # Star Wars (1977)
     (0,1,1), # Toy Story (1995)
     (0,16,2), # Casino (1995)
     (0,25,4), # Leaving Las Vegas (1995)
     (0,32,3), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,3), # Flintstones, The (1994)
     (0,379,3), # Timecop (1994)
     (0,296,5), # Pulp Fiction (1994)
     (0,858,1) , # Godfather, The (1972)
     (0,50,2) # Usual Suspects, The (1995)
    ]
new_user_ratings_RDD = sc.parallelize(new_user_ratings)
print 'New user ratings: %s' % new_user_ratings_RDD.take(10)

#Join our new ID user with their rates
complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)

#Train our modelbut adding new USER and print time
t0 = time()
new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed, 
                              iterations=iterations, lambda_=regularization_parameter)
tt = time() - t0
print "New model trained in %s seconds" % round(tt,3)

#get Movies that our new User didn't rate
t2 = time()
# get just movie IDs
new_user_ratings_ids = map(lambda x: x[1], new_user_ratings)
# keep just those not on the ID list
new_user_unrated_movies_RDD = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))
# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
tt2 = time() - t2
print "New user unrated predictions in %s seconds" % round(tt2,3)
#Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
t3 = time()
new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
#new_user_recommendations_rating_title_and_count_RDD.take(3)
tt3 = time() - t3
print "New user recommendations in %s seconds" % round(tt3,3)

#Reduce our RDD  to order (Title, Rating, Ratings Count)
t5 = time()
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
#Finally, get the highest rated recommendations for the new user, 
#filtering out movies with less than 25 ratings.
top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])
tt5 = time() - t5
print "TOP user recommendations in %s seconds" % round(tt5,3)
print ('TOP recommended movies (with more than 25 reviews):\n%s' %
        '\n'.join(map(str, top_movies)))
#END

print 'For testing data the RMSE is %s' % (error)
print "New model trained in %s seconds" % round(tt,3)
print "New user unrated predictions in %s seconds" % round(tt2,3)
print "New user recommendations in %s seconds" % round(tt3,3)
print "TOP user recommendations in %s seconds" % round(tt5,3)
tfinal = time() - tinicio
print "All Training with 20M data in %s seconds" % round(tfinal,3)
