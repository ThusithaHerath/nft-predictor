import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Sample Locations
locations = ["USA", "Europe", "Asia", "Middle East", "Australia"]

# Generate Sample Data
np.random.seed(42)
num_samples = 1000
data = {
    "Collection_Type": np.random.choice(["existing", "new"], num_samples),
    "Category": np.random.choice(["Gaming", "Art", "Collectibles", "Utility", "Metaverse"], num_samples),
    "Roadmap_Strength": np.random.randint(1, 10, num_samples),
    "Social_Media_Sentiment": np.random.uniform(0, 1, num_samples),
    "Whitelist_Count": np.random.randint(50, 1000, num_samples),
    "Volume": np.random.randint(1000, 10000, num_samples),
    "Sales": np.random.randint(50, 2000, num_samples),
    "Owners": np.random.randint(50, 2000, num_samples),
    "Average_Price": np.random.uniform(0.01, 1.0, num_samples),
    "Best_Location": np.random.choice(locations, num_samples)
}

df = pd.DataFrame(data)

# Encode Collection Type and Category
category_mapping = {"Art": 0, "Gaming": 1, "Collectibles": 2, "Utility": 3, "Metaverse": 4}
collection_type_mapping = {"existing": 0, "new": 1}
df["Category"] = df["Category"].map(category_mapping)
df["Collection_Type"] = df["Collection_Type"].map(collection_type_mapping)

# Prepare Features and Labels
X = df.drop(columns=["Best_Location"])
y = df["Best_Location"]

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model
location_model = RandomForestClassifier(n_estimators=100, random_state=42)
location_model.fit(X_train, y_train)

# Save Model & Scaler
joblib.dump(location_model, "location_model.pkl")
joblib.dump(scaler, "location_scaler.pkl")

print("✅ Location Model Trained and Saved for BOTH new & existing collections!")
