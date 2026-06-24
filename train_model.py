import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Data/Resume_Data.csv")

# REMOVE EMPTY VALUES
df = df.dropna(subset=["Feature", "Category"])

# reset index
df = df.reset_index(drop=True)

# CORRECT COLUMN USAGE
X = df["Feature"]
y = df["Category"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# FIT TF-IDF PROPERLY
tfidf = TfidfVectorizer(stop_words="english")
X_train_vec = tfidf.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save models
pickle.dump(model, open("resume_classifier.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))

print("Model has been trained successfully!")