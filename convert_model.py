import xgboost as xgb
import joblib

# Convert Existing JSON Model Back to Pickle
try:
    existing_model = xgb.Booster()
    existing_model.load_model("nft_xgb_model_optimized.json")
    
    # Convert to Sklearn-compatible model
    existing_model_sklearn = xgb.XGBClassifier()
    existing_model_sklearn._Booster = existing_model
    joblib.dump(existing_model_sklearn, "nft_xgb_model_optimized.pkl")

    print("✅ Converted nft_xgb_model_optimized.json back to .pkl")
except Exception as e:
    print(f"❌ Error converting existing model back: {e}")

# Convert New Collection JSON Model Back to Pickle
try:
    new_model = xgb.Booster()
    new_model.load_model("nft_new_model.json")
    
    # Convert to Sklearn-compatible model
    new_model_sklearn = xgb.XGBClassifier()
    new_model_sklearn._Booster = new_model
    joblib.dump(new_model_sklearn, "nft_new_model.pkl")

    print("✅ Converted nft_new_model.json back to .pkl")
except Exception as e:
    print(f"❌ Error converting new model back: {e}")
