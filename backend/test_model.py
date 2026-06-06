import pandas as pd
import joblib

model = joblib.load("model.pkl")
encoder = joblib.load("label_encoder.pkl")

forest = encoder.transform(["Evergreen"])[0]

sample = pd.DataFrame([{
    "Rainfall_mm": 1800,
    "Temperature_C": 24,
    "Humidity_Percent": 85,
    "Elevation_m": 1500,
    "Forest_Type": forest,
    "NDVI": 0.8,
    "Species_Count": 250
}])

prediction = model.predict(sample)

print(prediction)