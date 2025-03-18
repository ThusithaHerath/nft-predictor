import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("NFT_New_Collections_Updated.csv")

# Define features and target
features = ["Category", "Roadmap_Strength", "Social_Media_Sentiment", "Whitelist_Count", "Marketing_Strategies"]
target = "Success"  # 1 = Successful, 0 = Not Successful

# Encode categorical values
le_category = LabelEncoder()
df["Category"] = le_category.fit_transform(df["Category"])

le_marketing = LabelEncoder()
df["Marketing_Strategies"] = le_marketing.fit_transform(df["Marketing_Strategies"])

# Define X (features) and y (target)
X = df[features]
y = df[target]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("New NFT Model Accuracy:", accuracy)

# Save trained model & encoders
joblib.dump(model, "nft_new_model.pkl")
joblib.dump(scaler, "scaler_new.pkl")
joblib.dump(le_category, "category_encoder.pkl")
joblib.dump(le_marketing, "marketing_encoder.pkl")

print("✅ New NFT Model Trained & Saved Successfully!")
