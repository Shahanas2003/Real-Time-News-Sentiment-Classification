
import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.sql.types import StructType, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# --- 1. PySpark and Streaming Setup ---
# Initialize Spark session
spark = SparkSession.builder.appName("NewsSentimentDashboard").getOrCreate()

# Define schema for streaming JSON files
schema = StructType().add("title", StringType(), True)

# Set the path to the directory where news files are saved
SAVE_DIR = "/content/news_stream_folder"

# Labeled training data
data = [
    ("Stock markets crash amid economic fears", 0),
    ("Local community celebrates festival with joy", 1),
    ("Earthquake kills hundreds in city", 0),
    ("New vaccine brings hope to millions", 1),
    ("Government faces backlash over corruption scandal", 0),
    ("Breakthrough in clean energy technology announced", 1)
]
columns = ["title", "label"]
df = spark.createDataFrame(data, columns)

# Build and train the ML pipeline
df_clean = df.withColumn("clean_title", lower(regexp_replace(col("title"), "[^a-zA-Z0-9\\s]", "")))
tokenizer = Tokenizer(inputCol="clean_title", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
model = pipeline.fit(df_clean)

# Start the streaming query
# Read stream from the directory
news_stream = spark.readStream.schema(schema).json(SAVE_DIR)
news_stream_clean = news_stream.withColumn("clean_title", lower(regexp_replace(col("title"), "[^a-zA-Z0-9\\s]", "")))
predictions_stream = model.transform(news_stream_clean).select("title", "prediction")

# Stop any existing query
for q in spark.streams.active:
    q.stop()

# Start a new query to write to an in-memory table
query = predictions_stream.writeStream \
    .outputMode("append") \
    .format("memory") \
    .queryName("news_predictions") \
    .start()

# --- 2. Streamlit Dashboard Logic ---
st.title("Real-Time News Sentiment Dashboard")

# The main loop for the dashboard
while True:
    try:
        # Check if the query is active and new data has arrived
        if not query.status['isTerminated']:
            # Read the latest data from the in-memory table
            sdf = spark.sql("SELECT * FROM news_predictions ORDER BY title ASC") # Use a stable ordering
            df_pandas = sdf.toPandas()

            if not df_pandas.empty:
                # Map 0/1 to Negative/Positive
                df_pandas["Sentiment"] = df_pandas["prediction"].map({0: "Negative", 1: "Positive"})
                
                # Create a container to hold the updated elements
                with st.container():
                    st.header("Sentiment Distribution")
                    st.bar_chart(df_pandas["Sentiment"].value_counts())
                    
                    st.header("Latest News Headlines")
                    st.dataframe(df_pandas[["title", "Sentiment"]].tail(20))
            else:
                st.write("Waiting for news articles...")
    except Exception as e:
        st.error(f"Error reading from Spark table: {e}")

    # A better way to refresh is to use a time-based loop, not experimental_rerun()
    # Streamlit will handle the reruns automatically when state changes or input widgets are used.
    # A manual loop like this is one option, but it's not the "Streamlit way."
    # For a real app, you would rely on user input or a scheduled task outside of Streamlit to trigger a rerun.

    # This loop is for demonstration, in a real-world scenario you'd use a different approach.
