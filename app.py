from flask import Flask, render_template, request, jsonify
import pandas as pd
import xgboost as xgb
from textblob import TextBlob
import os
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

app = Flask(__name__)

# Load trained XGBoost model
model_path = "xgboost_model.json"
if os.path.exists(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    print("✅ XGBoost Model loaded successfully from:", model_path)
else:
    model = None
    print("❌ Model could not be loaded!")

# Temporary storage for listing data (since we don't have a DB)
listing_records = []

# Function to extract sentiment score from review text
def get_sentiment(text):
    if not text:
        return 0  # Neutral sentiment if no text
    return TextBlob(text).sentiment.polarity  # Sentiment score between -1 and 1

# Route for HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not available"}), 500

    try:
        # Get user input
        listing_id = request.form['listing_id']
        price = float(request.form['price'])
        available = request.form['available']  # 't' or 'f'
        review_scores_communication = float(request.form['review_scores_communication'])
        review_scores_location = float(request.form['review_scores_location'])
        review_scores_value = float(request.form['review_scores_value'])
        instant_bookable = int(request.form['instant_bookable'])
        reviews_per_month = float(request.form['reviews_per_month'])
        review_text = request.form['review_text']

        # Convert 'available' (Label Encoding: t -> 1, f -> 0)
        available = 1 if available.lower() == 't' else 0

        # Store data (simulating database behavior)
        listing_records.append({
            "listing_id": listing_id,
            "price": price,
            "available": available
        })

        # Convert to DataFrame for computation
        df = pd.DataFrame(listing_records)

        # Compute avg_price & availability_rate dynamically per listing_id
        listing_group = df[df["listing_id"] == listing_id]
        avg_price = listing_group["price"].mean()  # Compute mean price
        availability_rate = listing_group["available"].mean()  # Compute mean availability

        # Convert review text to sentiment score
        sentiment_score = get_sentiment(review_text)

        # Create DataFrame with computed values
        input_data = pd.DataFrame([{
            "avg_price": avg_price,
            "availability_rate": availability_rate,
            "review_scores_communication": review_scores_communication,
            "review_scores_location": review_scores_location,
            "review_scores_value": review_scores_value,
            "instant_bookable": instant_bookable,
            "reviews_per_month": reviews_per_month,
            "sentiment_score": sentiment_score
        }])

        # Apply Log Transformation to all features except sentiment_score
        features_to_log = ["avg_price", "availability_rate", "review_scores_communication",
                           "review_scores_location", "review_scores_value", "reviews_per_month"]
        input_data[features_to_log] = input_data[features_to_log].apply(np.log1p)

        # Apply Polynomial Expansion (Disable Bias Term to Match Training)
        poly = PolynomialFeatures(degree=4, include_bias=False)
        input_poly = poly.fit_transform(input_data)

        # Check if feature count matches
        expected_features = model.num_features()
        actual_features = input_poly.shape[1]

        if actual_features != expected_features:
            return jsonify({"error": f"Feature count mismatch! Expected {expected_features}, but got {actual_features}"}), 400

        # Convert to XGBoost DMatrix
        dmatrix = xgb.DMatrix(input_poly)

        # Make prediction
        prediction = model.predict(dmatrix)[0]

        return render_template('index.html', prediction=f"Predicted Price: ${prediction:.2f}")

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
