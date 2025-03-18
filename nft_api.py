from flask import Flask, request, jsonify
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained models and scalers
existing_model = joblib.load("nft_xgb_model_optimized.pkl")
new_model = joblib.load("nft_new_model.pkl")
scaler_existing = joblib.load("scaler.pkl")
scaler_new = joblib.load("scaler_new.pkl")

# Define feature sets for both models
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

# Function to recommend marketing strategies with explanations
def recommend_marketing_strategies(data, is_existing):
    strategies = set()  # Use a set to prevent duplicate strategies
    reasons = []

    if is_existing:
        # Strategies for existing NFT collections
        if data["Volume"] < 5000:
            strategies.add("Increase NFT visibility through influencer marketing & collaborations.")
            reasons.append("Your trading volume is low. Partnering with influencers can increase exposure.")

        if data["Sales"] < 500:
            strategies.add("Run NFT giveaways and airdrops to boost adoption.")
            reasons.append("Low sales indicate a lack of interest. Airdrops help attract initial buyers.")

        if data["Owners"] < 200:
            strategies.add("Engage your community more actively on Twitter & Discord.")
            reasons.append("A small number of unique owners means limited market reach. Engage more users.")

        if not strategies:
            strategies.add("Invest in Twitter Ads & Discord community engagement.")
            reasons.append("Your collection is performing well, but paid ads can further enhance visibility.")
    
    else:
        # Strategies for new NFT collections
        if data["Social_Media_Sentiment"] < 0.5:
            strategies.add("Improve community engagement via Twitter & Discord.")
            reasons.append("Your project has low social media sentiment, indicating weak community support.")

        if data["Whitelist_Count"] < 100:
            strategies.add("Increase presale access & airdrop campaigns.")
            reasons.append("A low whitelist count suggests fewer early supporters. Offer exclusive presales.")

        if data["Roadmap_Strength"] < 5:
            strategies.add("Enhance the project roadmap to build investor confidence.")
            reasons.append("Your roadmap strength is weak, which may deter investors. Improve long-term vision.")

        if not strategies:
            strategies.add("Run targeted influencer and paid marketing campaigns.")
            reasons.append("Your project looks strong, but strategic marketing can accelerate growth.")

    # Ensure at least three strategies are returned
    while len(strategies) < 3:
        strategies.add("Leverage partnerships with established brands in the NFT space.")
        reasons.append("Collaborations help build trust and increase credibility.")

    return list(strategies), reasons  # ✅ FIXED INDENTATION HERE

@app.route('/predict-nft', methods=['POST'])
def predict_nft():
    data = request.get_json()

    if 'collection_type' not in data:
        return jsonify({"error": "collection_type is required (new or existing)"}), 400

    collection_type = data["collection_type"].lower()

    if collection_type == "existing":
        required_fields = existing_features[:4]  # First 4 features are required
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Compute additional features dynamically
        data["NFT_Retention_Rate"] = data["Owners"] / data["Sales"] if data["Sales"] > 0 else 0
        data["Liquidity_Ratio"] = data["Sales"] / data["Volume"] if data["Volume"] > 0 else 0
        data["NFT_Price_Fluctuation"] = (data["Average_Price"] - 0.01) / data["Average_Price"] if data["Average_Price"] > 0 else 0

        # Convert input to DataFrame and scale
        user_input = pd.DataFrame([[data[f] for f in existing_features]], columns=existing_features)
        user_input_scaled = scaler_existing.transform(user_input)

        # Predict success probability
        probability = existing_model.predict_proba(user_input_scaled)[0][1] * 100
        probability = min(100, max(0, probability))  # Ensure 0-100%
        risk_level = get_risk_level(probability)

        # Recommend marketing strategies
        marketing_strategies, reasons = recommend_marketing_strategies(data, is_existing=True)

        return jsonify({
            "success_probability": f"{probability:.2f}%",
            "risk_level": risk_level,
            "recommended_marketing_strategies": marketing_strategies,
            "strategy_explanation": reasons
        })

    elif collection_type == "new":
        # Ensure all required fields exist
        missing_fields = [f for f in new_features if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Convert input to DataFrame and scale
        user_input = pd.DataFrame([[data[f] for f in new_features]], columns=new_features)
        user_input_scaled = scaler_new.transform(user_input)

        # Predict success probability
        probability = new_model.predict_proba(user_input_scaled)[0][1] * 100
        probability = min(100, max(0, probability))  # Ensure 0-100%
        risk_level = get_risk_level(probability)

        # Recommend marketing strategies
        marketing_strategies, reasons = recommend_marketing_strategies(data, is_existing=False)

        return jsonify({
            "success_probability": f"{probability:.2f}%",
            "risk_level": risk_level,
            "recommended_marketing_strategies": marketing_strategies,
            "strategy_explanation": reasons
        })

    else:
        return jsonify({"error": "Invalid collection_type. Use 'new' or 'existing'."}), 400

if __name__ == '__main__':
    app.run(debug=True)
