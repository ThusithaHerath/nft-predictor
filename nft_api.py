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

# Define marketing strategies based on weaknesses
def suggest_improvements(data):
    suggestions = []
    
    if data["Volume"] < 5000:
        suggestions.append("Increase trading volume by using promotional airdrops and collaborations.")

    if data["Sales"] < 500:
        suggestions.append("Boost sales by offering limited-time discounts or exclusive perks.")

    if data["Owners"] < 200:
        suggestions.append("Grow community engagement through Discord & Twitter campaigns.")

    if data["Average_Price"] < 0.5:
        suggestions.append("Adjust pricing strategy. Consider dynamic pricing based on demand.")

    return suggestions

# Define best marketing strategies based on category
def get_marketing_strategy(category):
    if category in ["Art", "Digital Art"]:
        return "Leverage Instagram and Twitter marketing, collaborate with digital artists."
    elif category in ["Gaming", "Metaverse"]:
        return "Engage in Discord communities, partner with gaming influencers, in-game events."
    elif category in ["Collectibles", "PFP"]:
        return "Use influencer marketing, NFT giveaways, and whitelist promotions."
    elif category in ["Music", "Audio"]:
        return "Use TikTok virality, collaborate with musicians, host NFT-based concerts."
    else:
        return "Community engagement, targeted social media ads, partnerships with brands."

# Define geographic targeting based on category
def get_best_regions(category):
    if category in ["Art", "Digital Art"]:
        return ["USA", "Europe", "Japan"]
    elif category in ["Gaming", "Metaverse"]:
        return ["Southeast Asia", "USA", "South Korea"]
    elif category in ["Collectibles", "PFP"]:
        return ["USA", "Europe", "Middle East"]
    elif category in ["Music", "Audio"]:
        return ["USA", "Europe", "Latin America"]
    else:
        return ["Global"]

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

        # Make prediction
        prediction = model.predict(user_input_scaled)
        result = "Yes" if prediction[0] == 1 else "No"

        response = {"success_prediction": result}

        # If prediction is "No", provide suggestions
        if result == "No":
            response["improvement_suggestions"] = suggest_improvements(data)
            response["best_marketing_strategy"] = get_marketing_strategy(data.get("Category", "General"))
            response["target_geographic_regions"] = get_best_regions(data.get("Category", "General"))

        return jsonify(response)

    else:
        return jsonify({"error": "This API currently supports only existing collections. The new collection model is under development."}), 400

if __name__ == '__main__':
    app.run(debug=True)
