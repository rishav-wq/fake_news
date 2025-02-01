import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, request, jsonify

# Download stopwords if not already downloaded
nltk.download("stopwords")

app = Flask(__name__)

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Initialize PorterStemmer
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Function for text preprocessing
def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", " ", text).lower()
    text = " ".join([ps.stem(word) for word in text.split() if word not in stop_words])
    return text

# Route for homepage
@app.route("/")
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No input text provided"}), 400

    input_text = data["text"]
    processed_text = preprocess_text(input_text)
    transformed_text = vectorizer.transform([processed_text])
    prediction = model.predict(transformed_text)[0]

    return jsonify({"prediction": int(prediction)})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
