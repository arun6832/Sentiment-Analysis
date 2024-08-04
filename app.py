from flask import Flask, render_template, request, jsonify
import pickle
import re
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load models and tokenizer
with open('models/logistic_regression.pkl', 'rb') as f:
    log_reg = pickle.load(f)
with open('models/naive_bayes.pkl', 'rb') as f:
    nb = pickle.load(f)
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Define function to clean text
def clean_text(text):
    text = re.sub("wouldn't", 'would not', text)
    text = re.sub("they've", 'they have', text)
    text = re.sub("should've", 'should have', text)
    text = re.sub("could've", 'could have', text)
    text = re.sub("can't", 'can not', text)
    text = re.sub("couldn't", 'could not', text)
    text = re.sub("didn't", 'did not', text)
    text = re.sub("do've", 'do have', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower()

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the form
    review = request.form['review']
    review = clean_text(review)
    
    # Transform review
    review_tfidf = tfidf.transform([review])
    
    # Make predictions
    prediction_log_reg = log_reg.predict(review_tfidf)[0]
    prediction_nb = nb.predict(review_tfidf)[0]

    # Convert numerical predictions to sentiment labels
    sentiments = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment_log_reg = sentiments.get(prediction_log_reg, 'Unknown')
    sentiment_nb = sentiments.get(prediction_nb, 'Unknown')

    # Return results
    return jsonify({
        'log_reg': sentiment_log_reg,
        'nb': sentiment_nb,
        'log_reg_score': int(prediction_log_reg),
        'nb_score': int(prediction_nb)
    })

@app.route('/chart-data')
def chart_data():
    # Example chart data
    data = {
        'labels': ['Negative', 'Neutral', 'Positive'],
        'log_reg_values': [50, 30, 20],  # Example values
        'nb_values': [40, 35, 25]        # Example values
    }
    return jsonify(data)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
