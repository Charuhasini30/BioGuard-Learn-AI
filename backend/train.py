import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

# Load Dataset
df = pd.read_csv("../data/BioGuardAI_Biodiversity_Dataset_10000.csv")

# Encode Forest Type
le = LabelEncoder()
df["Forest_Type"] = le.fit_transform(df["Forest_Type"])

# Features
X = df[
    [
        "Rainfall_mm",
        "Temperature_C",
        "Humidity_Percent",
        "Elevation_m",
        "Forest_Type",
        "NDVI",
        "Species_Count"
    ]
]

# Target
y = df["Biodiversity_Health_Score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# Save Model
joblib.dump(model, "model.pkl")

# Save Encoder
joblib.dump(le, "label_encoder.pkl")

print("Model Saved Successfully")