import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd
import requests

# --------------------------
# 1️⃣ Initialize Spark
# --------------------------
spark = SparkSession.builder \
    .appName("Real-Time News Sentiment") \
    .getOrCreate()

# --------------------------
# 2️⃣ Load pre-trained PySpark ML model
# --------------------------
# Make sure your pipeline model is saved in 'models/news_sentiment_model'
model_path = "models/news_sentiment_model"
pipeline_model = PipelineModel.load(model_path)

# --------------------------
# 3️⃣ Fetch news headlines
# --------------------------
# Replace 'YOUR_API_KEY' and 'NEWS_API_URL' with your API details
NEWS_API_URL = "https://newsapi.org/v2/top-headlines?country=us&apiKey=YOUR_API_KEY"

def fetch_news():
    response = requests.get(NEWS_API_URL)
    data = response.json()
    articles = data.get("articles", [])
    headlines = [article["title"] for article in articles if article.get("title")]
    return headlines

headlines = fetch_news()

# Convert to Spark DataFrame
news_df = spark.createDataFrame(pd.DataFrame(headlines, columns=["headline"]))

# --------------------------
# 4️⃣ Make predictions
# --------------------------
predictions_df = pipeline_model.transform(news_df)
# Select only headline and predicted sentiment
results = predictions_df.select("headline", "prediction").toPandas()
results["prediction"] = results["prediction"].map({1: "Positive", 0: "Negative"})

# --------------------------
# 5️⃣ Streamlit Dashboard
# --------------------------
st.set_page_config(page_title="Real-Time News Sentiment Dashboard", layout="wide")
st.title("📰 Real-Time News Sentiment Dashboard")

# Refresh button
if st.button("Refresh News"):
    st.experimental_rerun()

st.subheader("Latest Headlines & Sentiment")
st.dataframe(results)

# Optional: Show counts
st.subheader("Sentiment Summary")
summary = results["prediction"].value_counts().reset_index()
summary.columns = ["Sentiment", "Count"]
st.bar_chart(summary.set_index("Sentiment"))
