import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Page title
# -----------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and the AI model will predict "
    "whether it is Positive or Negative."
)


# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")

    return df


df = load_data()


# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    return text


df["review"] = df["review"].apply(clean_text)


# -----------------------------
# Convert sentiment to numbers
# -----------------------------
df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})


# -----------------------------
# Train AI model
# -----------------------------
@st.cache_resource
def train_model(reviews, sentiments):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X = vectorizer.fit_transform(reviews)

    model = LogisticRegression(max_iter=1000)

    model.fit(X, sentiments)

    return vectorizer, model


vectorizer, model = train_model(
    df["review"],
    df["sentiment"]
)


# -----------------------------
# Review input box
# -----------------------------
review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was amazing and I really enjoyed it!"
)


# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        cleaned_review = clean_text(review)

        review_tfidf = vectorizer.transform(
            [cleaned_review]
        )

        prediction = model.predict(review_tfidf)[0]

        if prediction == 1:
            st.success("😊 Positive Review")

        else:
            st.error("😞 Negative Review")