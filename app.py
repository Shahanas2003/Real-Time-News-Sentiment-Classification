import subprocess
subprocess.run(["pip", "install", "--upgrade", "pip"])

import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml import PipelineModel

# -----------------------------
# Initialize Spark
# -----------------------------
spark = SparkSession.builder.appName("NewsSentimentDashboard").getOrCreate()

# Load pre-trained model
model = PipelineModel.load("/content/news_sentiment_model")

# -----------------------------
# Streaming input (folder of JSON files)
# -----------------------------
news_stream = spark.readStream.schema("title STRING").json("/content/news_stream_folder")

# Preprocess headlines
news_clean = news_stream.withColumn(
    "clean_title", lower(regexp_replace(col("title"), r"[^a-zA-Z0-9\s]", ""))
)

# Apply model
predictions = model.transform(news_clean).select("title", "prediction")

# -----------------------------
# Streamlit dashboard
# -----------------------------
st.title("Real-Time News Sentiment Dashboard")

# Function to convert Spark DF to Pandas for visualization
def spark_to_pandas(sdf):
    try:
        return sdf.toPandas()
    except:
        return pd.DataFrame(columns=["title", "prediction"])

# Placeholder for dynamic updates
placeholder = st.empty()

# Stream loop
query = predictions.writeStream \
    .outputMode("append") \
    .format("memory") \
    .queryName("news_predictions") \
    .start()

import time

while True:
    # Read latest predictions from memory table
    sdf = spark.sql("SELECT * FROM news_predictions")
    df = spark_to_pandas(sdf)

    # Display counts of Positive / Negative
    if not df.empty:
        sentiment_counts = df["prediction"].value_counts()
        st.bar_chart(sentiment_counts)

        # Show latest headlines with sentiment
        df["Sentiment"] = df["prediction"].map({0: "Negative", 1: "Positive"})
        st.dataframe(df[["title", "Sentiment"]].tail(20))

    time.sleep(5)
