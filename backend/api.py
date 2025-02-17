# from flask import Flask, request, jsonify
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# analyzer = SentimentIntensityAnalyzer()

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     text = data.get("text", "")

#     if not text:
#         return jsonify({"error": "No text provided"}), 400

#     sentiment_score = analyzer.polarity_scores(text)['compound']

#     if sentiment_score >= 0.05:
#         sentiment = "Positive"
#     elif sentiment_score <= -0.05:
#         sentiment = "Negative"
#     else:
#         sentiment = "Neutral"

#     return jsonify({"sentiment": sentiment, "score": sentiment_score})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

app = Flask(__name__)
CORS(app)

# Load VADER model
analyzer = SentimentIntensityAnalyzer()

# Check if trained model exists, otherwise use VADER
MODEL_PATH = "sentiment_model.pkl"
if os.path.exists(MODEL_PATH):
    model, vectorizer = joblib.load(MODEL_PATH)
else:
    model, vectorizer = None, None

# Analyze sentiment using either VADER or trained ML model
def analyze_sentiment(text):
    if model:
        transformed_text = vectorizer.transform([text])
        sentiment = model.predict(transformed_text)[0]
        return sentiment
    else:
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.1:
            return "Positive"
        elif score <= -0.1:
            return "Negative"
        else:
            return "Neutral"

# API endpoint for sentiment analysis
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentiment = analyze_sentiment(text)
    return jsonify({"sentiment": sentiment})

# API endpoint to save user feedback
@app.route('/feedback', methods=['POST'])
def save_feedback():
    data = request.json
    text = data.get("text", "")
    correct_sentiment = data.get("correct_sentiment", "")

    if not text or not correct_sentiment:
        return jsonify({"error": "Missing text or sentiment"}), 400

    feedback_df = pd.DataFrame([[text, correct_sentiment]], columns=["text", "sentiment"])
    feedback_df.to_csv("feedback_data.csv", mode='a', header=not os.path.exists("feedback_data.csv"), index=False)

    return jsonify({"message": "Feedback saved successfully"}), 200

# Retrain the model using feedback data
@app.route('/retrain', methods=['GET'])
def retrain_model():
    if not os.path.exists("feedback_data.csv"):
        return jsonify({"error": "No feedback data available"}), 400

    df = pd.read_csv("feedback_data.csv")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["sentiment"]

    model = MultinomialNB()
    model.fit(X, y)

    joblib.dump((model, vectorizer), MODEL_PATH)
    return jsonify({"message": "Model retrained successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)
