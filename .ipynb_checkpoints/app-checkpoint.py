import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# Download stopwords
nltk.download('stopwords')

# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + " " + news_df['title']

# Initialize PorterStemmer
ps = PorterStemmer()

# Function for stemming and preprocessing
def stemming(content):
    # Remove non-alphabetic characters
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    # Convert to lowercase
    stemmed_content = stemmed_content.lower()
    # Tokenize and remove stopwords
    stemmed_content = [ps.stem(word) for word in stemmed_content.split() if word not in stopwords.words('english')]
    # Join words back into a single string
    return " ".join(stemmed_content)

# Apply stemming to 'content'
news_df['content'] = news_df['content'].apply(stemming)

# Define features and target
X = news_df['content'].values
y = news_df['label'].values

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# --- Streamlit Website Design ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #6c7bff, #3a82ff);
            font-family: 'Poppins', sans-serif;
        }
        .title {
            font-size: 48px;
            text-align: center;
            color: #ffffff;
            font-weight: bold;
            margin-top: 50px;
        }
        .header {
            text-align: center;
            font-size: 20px;
            color: #f1f1f1;
            margin-bottom: 30px;
        }
        .input-container {
            display: flex;
            justify-content: center;
            padding: 20px;
        }
        .input-box {
            width: 80%;
            height: 200px;
            padding: 10px;
            border-radius: 10px;
            font-size: 16px;
            border: none;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            background-color: #ffffff;
            resize: none;
        }
        .prediction {
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            padding: 15px;
            border-radius: 8px;
            margin-top: 30px;
            transition: all 0.3s ease-in-out;
        }
        .fake {
            background-color: #ff6f61;
            color: white;
            animation: fadeIn 0.5s ease-in-out;
        }
        .real {
            background-color: #4CAF50;
            color: white;
            animation: fadeIn 0.5s ease-in-out;
        }
        .button-container {
            text-align: center;
            margin-top: 30px;
        }
        .submit-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 18px;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
        .submit-button:hover {
            background-color: #45a049;
        }
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<div class="title">Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Check if a news article is real or fake by pasting it below.</div>', unsafe_allow_html=True)

# Input text area
input_text = st.text_area("", height=200, placeholder="Paste your news article here...", key="news_input", max_chars=1000)

def prediction(input_text):
    input_data = vectorizer.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

# Button to trigger prediction
if st.button("Predict", key="predict_button", help="Click to predict if the news is real or fake", use_container_width=True):
    if input_text:
        pred = prediction(input_text)
        if pred == 1:
            st.markdown('<div class="prediction fake">The News is Fake!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction real">The News is Real!</div>', unsafe_allow_html=True)
