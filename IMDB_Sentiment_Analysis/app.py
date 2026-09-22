import pandas as pd
import re

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.shape)
print(df.columns)
# Convert sentiment into numbers
df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

print(df.head())
# Clean review text
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    return text

df["review"] = df["review"].apply(clean_text)

print(df.head())
from sklearn.model_selection import train_test_split

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    df["review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text into numerical features
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)
from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_tfidf, y_train)

print("Model training completed!")
from sklearn.metrics import accuracy_score

# Make predictions on test data
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")
from sklearn.metrics import classification_report, confusion_matrix

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"]
))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Test your own review
review = input("\nEnter a movie review: ")

# Clean the review
review_clean = clean_text(review)

# Convert review into TF-IDF
review_tfidf = vectorizer.transform([review_clean])

# Predict sentiment
prediction = model.predict(review_tfidf)

if prediction[0] == 1:
    print("Predicted Sentiment: Positive 😊")
else:
    print("Predicted Sentiment: Negative 😞")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    