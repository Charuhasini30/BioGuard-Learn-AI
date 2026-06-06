import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("../data/BioGuardAI_Biodiversity_Dataset_10000.csv")

# Load Encoder
le = joblib.load("label_encoder.pkl")
df["Forest_Type"] = le.transform(df["Forest_Type"])

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

# Load model
model = joblib.load("model.pkl")

# Use only 500 samples for SHAP
X_sample = X.sample(500, random_state=42)

explainer = shap.Explainer(
    model.predict,
    X_sample
)

shap_values = explainer(X_sample)

shap.plots.beeswarm(
    shap_values,
    show=False
)

plt.savefig(
    "shap_summary.png",
    bbox_inches="tight"
)

print("SHAP graph saved successfully")