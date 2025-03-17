import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import joblib  # For saving the model

# Load dataset
df = pd.read_csv("NFT_Top_Collections_Final.csv")

# Select features for training
features = ["Volume", "Sales", "Owners", "Average_Price"]
df = df.dropna(subset=features)

# Balance the dataset
df_yes = df[df["Market_Cap"] > df["Market_Cap"].quantile(0.75)]  # Use top 25% for "Yes"
df_no = df[df["Market_Cap"] <= df["Market_Cap"].quantile(0.75)]
df_no_upsampled = resample(df_no, replace=True, n_samples=len(df_yes), random_state=42)
df_balanced = pd.concat([df_yes, df_no_upsampled])

# Prepare data
X = df_balanced[features]
y = (df_balanced["Market_Cap"] > df_balanced["Market_Cap"].quantile(0.75)).astype(int)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model and scaler
joblib.dump(model, "nft_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")
