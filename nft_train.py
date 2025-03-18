import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv("NFT_Top_Collections_Final.csv")

# Select features for training
features = ["Volume", "Sales", "Owners", "Average_Price"]
df = df.dropna(subset=features)

# Balance dataset
df_yes = df[df["Market_Cap"] > df["Market_Cap"].quantile(0.75)]
df_no = df[df["Market_Cap"] <= df["Market_Cap"].quantile(0.75)]
df_no_upsampled = resample(df_no, replace=True, n_samples=len(df_yes), random_state=42)
df_balanced = pd.concat([df_yes, df_no_upsampled])

# Fix division errors and replace infinity values
df_balanced["NFT_Retention_Rate"] = (df_balanced["Owners"] / df_balanced["Sales"]).replace([float('inf'), -float('inf')], 0)
df_balanced["Liquidity_Ratio"] = (df_balanced["Sales"] / df_balanced["Volume"]).replace([float('inf'), -float('inf')], 0)

# Fix price fluctuation (find max price per collection and avoid NaN)
df_balanced["Max_Price"] = df_balanced.groupby("Name")["Average_Price"].transform("max")
df_balanced["NFT_Price_Fluctuation"] = ((df_balanced["Max_Price"] - df_balanced["Average_Price"]) / df_balanced["Average_Price"]).replace([float('inf'), -float('inf')], 0)

# Drop temporary column
df_balanced.drop(columns=["Max_Price"], inplace=True)

# Replace NaN values with 0
df_balanced.fillna(0, inplace=True)

# Define Features (X) and Target (y)
features = ["Volume", "Sales", "Owners", "Average_Price", "NFT_Retention_Rate", "NFT_Price_Fluctuation", "Liquidity_Ratio"]
X = df_balanced[features]

# ✅ Define `y` Here
y = (df_balanced["Market_Cap"] > df_balanced["Market_Cap"].quantile(0.75)).astype(int)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7]
}

# Use GridSearch to find the best parameters
grid_search = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Train using best parameters
best_model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = best_model.predict(X_test)
print("Optimized Model Accuracy:", accuracy_score(y_test, y_pred))

# Save the optimized model and scaler
joblib.dump(best_model, "nft_xgb_model_optimized.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Optimized XGBoost model saved successfully!")