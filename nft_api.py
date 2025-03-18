from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load models and scalers
existing_model = joblib.load("nft_xgb_model_optimized.pkl")  # Model for existing NFTs
existing_scaler = joblib.load("scaler.pkl")

new_model = joblib.load("nft_new_model.pkl")  # Model for new NFTs
new_scaler = joblib.load("scaler_new.pkl")
category_encoder = joblib.load("category_encoder.pkl")
marketing_encoder = joblib.load("marketing_encoder.pkl")

# Define expected features for both models
existing_features = ["Volume", "Sales", "Owners", "Average_Price", "NFT_Retention_Rate", "NFT_Price_Fluctuation", "Liquidity_Ratio"]
new_features = ["Category", "Roadmap_Strength", "Social_Media_Sentiment", "Whitelist_Count", "Marketing_Strategies"]

# Function to determine risk level based on probability
def get_risk_level(probability):
    if probability >= 80:
        return "Low Risk"
    elif probability >= 50:
        return "Medium Risk"
    else:
        return "High Risk"

# Function to provide reasons why NFT is risky/successful
def explain_prediction(probability, data, is_new):
    explanations = []
    
    if probability < 50:
        if is_new:
            if data["Roadmap_Strength"] < 5:
                explanations.append("Weak roadmap—consider refining and expanding the project roadmap.")
            if data["Social_Media_Sentiment"] < 0:
                explanations.append("Negative social sentiment—improve community engagement and marketing.")
            if data["Whitelist_Count"] < 500:
                explanations.append("Low early adopter interest—increase whitelist incentives.")
        else:
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
        missing_fields = [f for f in existing_features[:4] if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Compute additional features dynamically
        volume = float(data["Volume"])
        sales = float(data["Sales"])
        owners = float(data["Owners"])
        avg_price = float(data["Average_Price"])
        nft_retention_rate = (owners / sales) if sales > 0 else 0
        liquidity_ratio = (sales / volume) if volume > 0 else 0
        nft_price_fluctuation = ((avg_price - 0.01) / avg_price) if avg_price > 0 else 0

        # Convert input data into a DataFrame
        user_input = pd.DataFrame([[volume, sales, owners, avg_price, nft_retention_rate, nft_price_fluctuation, liquidity_ratio]], 
                                  columns=existing_features)

        # Scale input features
        user_input_scaled = existing_scaler.transform(user_input)

        # Make prediction probability
        probability = existing_model.predict_proba(user_input_scaled)[0][1] * 100  # Convert to percentage

        # Determine risk level
        risk_level = get_risk_level(probability)

        # Provide explanations
        reasons = explain_prediction(probability, data, is_new=False)

        return jsonify({
            "success_probability": f"{probability:.2f}%",
            "risk_level": risk_level,
            "analysis": reasons
        })

    elif collection_type == "new":
        # Ensure all necessary fields are present
        missing_fields = [f for f in new_features if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Encode categorical values
        category_encoded = category_encoder.transform([data["Category"]])[0]
        marketing_encoded = marketing_encoder.transform([data["Marketing_Strategies"]])[0]

        # Convert input data into a DataFrame
        user_input = pd.DataFrame([[category_encoded, data["Roadmap_Strength"], data["Social_Media_Sentiment"], 
                                    data["Whitelist_Count"], marketing_encoded]], columns=new_features)

        # Scale input features
        user_input_scaled = new_scaler.transform(user_input)

        # Make prediction probability
        probability = new_model.predict_proba(user_input_scaled)[0][1] * 100  # Convert to percentage

        # Determine risk level
        risk_level = get_risk_level(probability)

        # Provide explanations
        reasons = explain_prediction(probability, data, is_new=True)

        return jsonify({
            "success_probability": f"{probability:.2f}%",
            "risk_level": risk_level,
            "analysis": reasons
        })

    else:
        return jsonify({"error": "Invalid collection_type. Use 'new' or 'existing'"}), 400

if __name__ == '__main__':
    app.run(debug=True)
