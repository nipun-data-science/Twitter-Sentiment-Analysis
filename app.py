import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pre-trained model from the pickle file
model = joblib.load('twitter_data.pkl')

@app.route('/')
def sentiment_form():
    return render_template('index.html')


# Define a route for sentiment analysis
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    # Get the tweet from the POST request
    tweet = request.json['tweet']
    tf=TfidfVectorizer()
    Fit = tf.fit([tweet])
    text_tf = Fit.transform([tweet])

    # Preprocess and vectorize the tweet (You may need to use the same preprocessing as when you trained the model)
    # Here's a simple example of preprocessing and vectorization using CountVectorizer:
    # from sklearn.feature_extraction.text import CountVectorizer
    # vectorizer = CountVectorizer()
    # tweet_vector = vectorizer.transform([tweet])

    # Make a sentiment prediction
    sentiment_prediction = model.predict([tweet])

    # Return the result as JSON
    result = {
        'tweet': tweet,
        'sentiment': sentiment_prediction[0]
    }

    return jsonify(result)

# Use this code to start the Flask app in a Jupyter Notebook
from werkzeug.serving import run_simple
run_simple('localhost', 5000, app)

