import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

# Load dataset for new collections
df = pd.read_csv("NFT_New_Collections_Updated.csv")

features = ["Category", "Roadmap_Strength", "Social_Media_Sentiment", "Whitelist_Count"]

# Encode categorical values (if necessary)
df["Category"] = df["Category"].astype("category").cat.codes

df = df.dropna(subset=features + ["Success"])  # Ensure no missing values

# Target variable: Success probability
y = df["Success"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df[features], y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, eval_metric="logloss")
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "nft_new_model.pkl")
joblib.dump(scaler, "scaler_new.pkl")

print("✅ New NFT Model Trained & Saved Successfully Without!")
