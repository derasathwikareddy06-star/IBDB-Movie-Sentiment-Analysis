# 🎬 IMDB Movie Sentiment Analysis

A Machine Learning and Natural Language Processing (NLP) project that analyzes movie reviews and predicts whether a review is **Positive** or **Negative**.

## 📌 Project Overview

This project uses the IMDB movie review dataset to build a sentiment classification model.

The system:

- Takes a movie review as input
- Cleans and preprocesses the text
- Converts text into numerical features using TF-IDF
- Uses Logistic Regression for classification
- Predicts the sentiment as Positive or Negative
- Provides an interactive Streamlit web application

## 🚀 Features

- Movie review sentiment prediction
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression classification
- Interactive Streamlit interface
- Pre-trained model and vectorizer
- Jupyter Notebook for analysis

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF
- Logistic Regression
- Streamlit
- Jupyter Notebook
- Git
- GitHub

## 📂 Project Structure

```text
IMDB-Movie-Sentiment-Analysis/
│
├── app.py
├── streamlit_app.py
├── run_project.bat
├── requirements.txt
├── test_setup.py
│
├── dataset/
│   └── movie_reviews.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebook/
│   └── sentiment_analysis.ipynb
│
└── src/
    └── preprocess.py