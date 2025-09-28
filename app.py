# app.py
import streamlit as st
import pandas as pd
import os
import json
import requests
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.sql.types import StructType, StringType
from pyspark.ml import PipelineModel
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# 1. Initialize Spark and load model
# -----------------------------
@st.experimental_singleton
def init_spark():
    spark = SparkSession.builder.appName("NewsSentimentDashboard").getOrCreate()
    model = PipelineModel.load("/content/news_sentiment_model")  # update path if needed
    return spark, model

spark, model = init_spark()

# -----------------------------
# 2. NewsData.io API
# -----------------------------
API_KEY = "pub_7c7f72f816dc47c28a889f0c0a5b371f"
BASE_URL = "https://newsdata.io/api/1/news"
SAVE_DIR = "news_stream_folder"
os.makedirs(SAVE_DIR, exist_ok=True)

VALID_CATEGORIES = ["business","entertainment","environment","food","health",
                    "politics","science","sports","technology","top"]

def fetch_news(category="top"):
    params = {"apikey": API_KEY, "language":"en", "category":category}
    r = requests.get(BASE_URL, params=params)
    if r.status_code == 200:
        return r.json().get("results", [])
    return []

def save_articles(articles):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"news_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        for article in articles:
            json.dump({"title": article.get("title","")}, f)
            f.write("\n")

# -----------------------------
# 3. Spark Structured Streaming
# -----------------------------
schema = StructType().add("title", StringType(), True)
news_stream = spark.readStream.schema(schema).json(SAVE_DIR)
predictions_stream = model.transform(news_stream).select("title", "prediction")

# Memory sink for dashboard
query = predictions_stream.writeStream \
    .outputMode("append") \
    .format("memory") \
    .queryName("news_predictions") \
    .start()

# -----------------------------
# 4. Streamlit Dashboard
# -----------------------------
st.title("Real-Time News Sentiment Dashboard")

category = st.selectbox("Select News Category:", VALID_CATEGORIES)
if st.button("Fetch Latest News"):
    articles = fetch_news(category)
    if articles:
        save_articles(articles)
        st.success(f"Fetched and saved {len(articles)} articles!")
    else:
        st.warning("No articles fetched.")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, limit=None, key="news_refresh")

# Function to read memory table
def get_latest_predictions():
    try:
        sdf = spark.sql("SELECT * FROM news_predictions")
        return sdf.toPandas()
    except:
        return pd.DataFrame(columns=["title", "prediction"])

df = get_latest_predictions()
if not df.empty:
    df["Sentiment"] = df["prediction"].map({0:"Negative", 1:"Positive"})
    st.bar_chart(df["Sentiment"].value_counts())
    st.dataframe(df[["title","Sentiment"]].tail(20))
else:
    st.write("No news processed yet.")
