#Create HDFS dir
hdfs dfs -mkdir -p /datasets
hdfs dfs -mkdir -p /datasets/ml-latest-small
hdfs dfs -mkdir -p /datasets/ml-latest

#Load HDFS files into /datasets/ml-latest-small

hdfs dfs -copyFromLocal datasets/ml-latest-small/movies.csv /datasets/ml-latest-small/movies.csv
hdfs dfs -copyFromLocal datasets/ml-latest-small/ratings.csv /datasets/ml-latest-small/ratings.csv

#Load HDFS files into /datasets/ml-latest

hdfs dfs -copyFromLocal datasets/ml-latest/movies.csv /datasets/ml-latest/movies.csv
hdfs dfs -copyFromLocal datasets/ml-latest/ratings.csv /datasets/ml-latest/ratings.csv

