from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging

app = Flask(__name__)

# Load trained models and scalers
existing_model = joblib.load("nft_xgb_model_optimized.pkl")
scaler_existing = joblib.load("scaler.pkl")

new_model = joblib.load("nft_new_model.pkl")
scaler_new = joblib.load("scaler_new.pkl")

# Load location-based NFT success model
location_model = joblib.load("location_model.pkl")
location_scaler = joblib.load("location_scaler.pkl")

# Define feature sets
existing_features = ["Volume", "Sales", "Owners", "Average_Price", "NFT_Retention_Rate", "NFT_Price_Fluctuation", "Liquidity_Ratio"]
new_collection_features = ["Category", "Roadmap_Strength", "Social_Media_Sentiment", "Whitelist_Count"]

# Category encoding
category_mapping = {"Art": 0, "Gaming": 1, "Collectibles": 2, "Utility": 3, "Metaverse": 4}
locations = ["USA", "Europe", "Asia", "Middle East", "Australia"]

logging.basicConfig(level=logging.DEBUG)

def get_risk_level(probability):
    if probability >= 80:
        return "Low Risk"
    elif probability >= 50:
        return "Medium Risk"
    else:
        return "High Risk"

def recommend_marketing_strategies(data, is_existing):
    strategies = []
    reasons = []

    if is_existing:
        # 🔹 Strategies for Low-Performing NFTs
        if data.get("Volume", 9999) < 5000:
            strategies.append("Increase NFT visibility through influencer marketing & collaborations.")
            reasons.append("Your trading volume is low. Partnering with influencers can increase exposure.")

        if data.get("Sales", 9999) < 500:
            strategies.append("Run NFT giveaways and airdrops to boost adoption.")
            reasons.append("Low sales indicate a lack of interest. Airdrops help attract initial buyers.")

        if data.get("Owners", 9999) < 200:
            strategies.append("Engage your community more actively on Twitter & Discord.")
            reasons.append("A small number of unique owners means limited market reach. Engage more users.")

        # 🔹 Strategies for High-Performing NFTs (Upper Limits)
        if data.get("Volume", 0) > 15000:
            strategies.append("Expand globally with exclusive partnerships & brand collaborations.")
            reasons.append("Your collection has high trading volume! Expansion into international markets is recommended.")

        if data.get("Sales", 0) > 3000:
            strategies.append("Offer premium membership & exclusive NFT perks for top buyers.")
            reasons.append("Your collection has strong sales! Creating VIP benefits can increase loyalty.")

        if data.get("Owners", 0) > 2000:
            strategies.append("Introduce NFT staking & rewards programs for long-term engagement.")
            reasons.append("Your collection has a high number of unique owners. Providing staking options will enhance retention.")

        # Ensure at least 1 recommendation is given
        if not strategies:
            strategies.append("Leverage Twitter Ads & Discord engagement for even greater exposure.")
            reasons.append("Your collection is performing well! Scaling efforts can further increase adoption.")

    else:
        # 🔹 Strategies for New Collections
        if data.get("Social_Media_Sentiment", 1) <= 0.9:
            strategies.append("Improve community engagement via Twitter & Discord.")
            reasons.append("Your project has moderate social media sentiment, indicating room for improvement.")

        if data.get("Whitelist_Count", 9999) <= 500:
            strategies.append("Increase presale access & airdrop campaigns.")
            reasons.append("A relatively low whitelist count suggests fewer early supporters.")

        if data.get("Roadmap_Strength", 9999) <= 9:
            strategies.append("Enhance the project roadmap to build investor confidence.")
            reasons.append("Your roadmap strength is decent but could be improved for long-term success.")

        # 🔹 If already strong, suggest scaling strategies
        if not strategies:
            strategies.append("Run targeted influencer and paid marketing campaigns.")
            reasons.append("Your project looks strong! Strategic marketing can accelerate growth.")

    return strategies, reasons


def predict_location_success(features_scaled):
    try:
        location_probabilities = location_model.predict_proba(features_scaled)[0]
        location_scores = dict(zip(locations, location_probabilities * 100))

        # Add longitude & latitude for each location
        location_coordinates = {
            "USA": {"latitude": 37.0902, "longitude": -95.7129},
            "Europe": {"latitude": 54.5260, "longitude": 15.2551},
            "Asia": {"latitude": 34.0479, "longitude": 100.6197},
            "Middle East": {"latitude": 29.2985, "longitude": 41.3133},
            "Australia": {"latitude": -25.2744, "longitude": 133.7751}
        }

        return [
            {
                "location": loc,
                "success_rate": f"{rate:.2f}%",
                "latitude": location_coordinates[loc]["latitude"],
                "longitude": location_coordinates[loc]["longitude"]
            }
            for loc, rate in location_scores.items()
        ]

    except ValueError as e:
        logging.error("Feature mismatch error: %s", str(e))
        return []


@app.route("/predict-nft", methods=["POST"])
def predict_nft():
    try:
        data = request.get_json()
        logging.debug("Received request data: %s", data)

        if "collection_type" not in data:
            return jsonify({"error": "collection_type is required (new or existing)"}), 400

        collection_type = data["collection_type"].lower()

        if collection_type == "existing":
            required_fields = existing_features[:4]
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

            data["NFT_Retention_Rate"] = data["Owners"] / data["Sales"] if data["Sales"] > 0 else 0
            data["Liquidity_Ratio"] = data["Sales"] / data["Volume"] if data["Volume"] > 0 else 0
            data["NFT_Price_Fluctuation"] = (data["Average_Price"] - 0.01) / data["Average_Price"] if data["Average_Price"] > 0 else 0

            user_input = pd.DataFrame([[data[f] for f in existing_features]], columns=existing_features)
            features_scaled = scaler_existing.transform(user_input)
            probability = existing_model.predict_proba(features_scaled)[0][1] * 100

            # Add missing fields for location model
            location_features = np.hstack((features_scaled, np.zeros((features_scaled.shape[0], 2))))
            location_features_scaled = location_scaler.transform(location_features)

        elif collection_type == "new":
            category = category_mapping.get(data.get("Category"), -1)
            roadmap_strength = float(data.get("Roadmap_Strength", 0))
            social_sentiment = float(data.get("Social_Media_Sentiment", 0))
            whitelist_count = int(data.get("Whitelist_Count", 0))

            if category == -1:
                return jsonify({"error": "Invalid Category"}), 400

            features_df = pd.DataFrame([[category, roadmap_strength, social_sentiment, whitelist_count]],
                                       columns=new_collection_features)
            features_scaled = scaler_new.transform(features_df)
            probability = float(new_model.predict_proba(features_scaled)[0][1] * 100)

            # Add missing fields for location model
            location_features = np.hstack((features_scaled, np.zeros((features_scaled.shape[0], 5))))
            location_features_scaled = location_scaler.transform(location_features)

        else:
            return jsonify({"error": "Invalid collection_type. Use 'new' or 'existing'."}), 400

        probability = min(100, max(0, probability))
        risk_level = get_risk_level(probability)

        location_success_rates = predict_location_success(location_features_scaled)
        marketing_strategies, reasons = recommend_marketing_strategies(data, is_existing=(collection_type == "existing"))

        return jsonify({
            "success_probability": f"{probability:.2f}%",
            "risk_level": risk_level,
            "recommended_marketing_strategies": marketing_strategies,
            "strategy_explanation": reasons,
            "location_success_rates": location_success_rates
        })

    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
