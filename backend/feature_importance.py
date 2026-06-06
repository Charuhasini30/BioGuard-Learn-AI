import joblib
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")

features = [
    "Rainfall",
    "Temperature",
    "Humidity",
    "Elevation",
    "Forest Type",
    "NDVI",
    "Species Count"
]

importance = model.feature_importances_

plt.figure(figsize=(8,5))
plt.barh(features, importance)

plt.xlabel("Importance")
plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

print("Feature Importance Saved")