from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load dataset and trained model
df = pd.read_csv("NFT_Top_Collections_Final.csv")
model = joblib.load("nft_model.pkl")  # Ensure this file is saved from nft_train.py
scaler = joblib.load("scaler.pkl")  # Ensure scaler is saved

# Features used in the model
features = ["Volume", "Sales", "Owners", "Average_Price"]

# Function to determine risk level based on probability
def get_risk_level(probability):
    if probability >= 80:
        return "Low Risk"
    elif probability >= 50:
        return "Medium Risk"
    else:
        return "High Risk"

# Function to provide reasons why NFT is risky/successful
def explain_prediction(probability, data):
    explanations = []
    
    if probability < 50:
        if data["Volume"] < 5000:
            explanations.append("Low trading volume—consider increasing market visibility.")
        if data["Sales"] < 500:
            explanations.append("Few total sales—focus on increasing community adoption.")
        if data["Owners"] < 200:
            explanations.append("Low number of unique owners—expand marketing efforts.")
        if data["Average_Price"] < 0.5:
            explanations.append("NFTs are undervalued—adjust pricing strategy.")

    else:
        explanations.append("NFT collection shows strong market potential based on past trends.")
    
    return explanations

@app.route('/predict-nft', methods=['POST'])
def predict_nft():
    data = request.get_json()

    # Validate input
    if 'collection_type' not in data:
        return jsonify({"error": "collection_type is required (new or existing)"}), 400

    collection_type = data["collection_type"].lower()

    if collection_type == "existing":
        # Ensure all necessary fields are present
        missing_fields = [f for f in features if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Convert input data into a DataFrame
        user_input = pd.DataFrame([[data[f] for f in features]], columns=features)

        # Scale input features
        user_input_scaled = scaler.transform(user_input)

        # Make prediction probability
        probability = model.predict_proba(user_input_scaled)[0][1] * 100  # Convert to percentage

        # Determine risk level
        risk_level = get_risk_level(probability)

        # Provide explanations
        reasons = explain_prediction(probability, data)

        return jsonify({
            "success_probability": f"{probability:.2f}%",
            "risk_level": risk_level,
            "analysis": reasons
        })

    else:
        return jsonify({"error": "This API currently supports only existing collections. The new collection model is under development."}), 400

if __name__ == '__main__':
    app.run(debug=True)
