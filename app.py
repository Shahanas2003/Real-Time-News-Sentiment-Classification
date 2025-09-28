import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.sql.types import StructType, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# --- PySpark and Streaming Setup ---
@st.cache_resource
def get_spark_session():
    """Initializes and returns a Spark session."""
    return SparkSession.builder.appName("NewsSentimentDashboard").getOrCreate()

spark = get_spark_session()

@st.cache_resource
def get_ml_model():
    """Builds and trains the ML pipeline once."""
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
    df_clean = df.withColumn("clean_title", lower(regexp_replace(col("title"), "[^a-zA-Z0-9\\s]", "")))
    
    tokenizer = Tokenizer(inputCol="clean_title", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
    return pipeline.fit(df_clean)

model = get_ml_model()

# Ensure streaming query is started
if "query" not in st.session_state:
    schema = StructType().add("title", StringType(), True)
    news_stream = spark.readStream.schema(schema).json("/content/news_stream_folder")
    news_stream_clean = news_stream.withColumn("clean_title", lower(regexp_replace(col("title"), "[^a-zA-Z0-9\\s]", "")))
    predictions_stream = model.transform(news_stream_clean).select("title", "prediction")
    
    # Stop any existing query
    for q in spark.streams.active:
        q.stop()

    query = predictions_stream.writeStream \
        .outputMode("append") \
        .format("memory") \
        .queryName("news_predictions") \
        .start()
    
    st.session_state["query"] = query

# --- Streamlit Dashboard Logic ---
st.title("Real-Time News Sentiment Dashboard")

# Add a refresh button for manual updates
if st.button("Refresh Dashboard"):
    st.rerun()

try:
    sdf = spark.sql("SELECT * FROM news_predictions ORDER BY title ASC")
    df_pandas = sdf.toPandas()

    if not df_pandas.empty:
        df_pandas["Sentiment"] = df_pandas["prediction"].map({0: "Negative", 1: "Positive"})
        
        st.header("Sentiment Distribution")
        st.bar_chart(df_pandas["Sentiment"].value_counts())
        
        st.header("Latest News Headlines")
        st.dataframe(df_pandas[["title", "Sentiment"]].tail(20))
    else:
        st.info("No news processed yet. Waiting for data...")
except Exception as e:
    st.error(f"Could not retrieve data from Spark table: {e}")
