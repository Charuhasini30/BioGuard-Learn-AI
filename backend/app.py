from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi import UploadFile, File
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoder
model = joblib.load("model.pkl")
encoder = joblib.load("label_encoder.pkl")


class BiodiversityInput(BaseModel):
    rainfall: float
    temperature: float
    humidity: float
    elevation: float
    forest_type: str
    ndvi: float
    species_count: int


@app.get("/")
def home():
    return {
        "message": "BioGuard AI API Running"
    }


@app.post("/predict")
def predict(data: BiodiversityInput):

    try:

        # Normalize forest type
        forest_mapping = {
            "evergreen": "Evergreen",
            "deciduous": "Deciduous",
            "semi-evergreen": "Semi-Evergreen",
            "grassland": "Grassland"
        }

        forest_name = data.forest_type.strip().lower()

        if forest_name not in forest_mapping:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid forest type. Use: {list(forest_mapping.keys())}"
            )

        encoded_forest = encoder.transform(
            [forest_mapping[forest_name]]
        )[0]

        input_df = pd.DataFrame([{
            "Rainfall_mm": data.rainfall,
            "Temperature_C": data.temperature,
            "Humidity_Percent": data.humidity,
            "Elevation_m": data.elevation,
            "Forest_Type": encoded_forest,
            "NDVI": data.ndvi,
            "Species_Count": data.species_count
        }])

        print("\nINPUT DATA")
        print(input_df)

        prediction = model.predict(input_df)

        print("\nMODEL PREDICTION")
        print(prediction)

        score = float(prediction[0])

        # Risk Level
        if score >= 80:
            risk = "Low"
        elif score >= 50:
            risk = "Medium"
        else:
            risk = "High"

        # Ecosystem Stability Index
        stability_index = min(
            100,
            round(
                score * 0.7 +
                data.ndvi * 20 +
                (data.humidity / 100) * 10,
                2
            )
        )

        # Biodiversity Resilience Index
        resilience_index = round(
            (
                data.ndvi * 40 +
                (data.species_count / 500) * 40 +
                (data.humidity / 100) * 20
            ),
            2
        )

        # Recommendation Engine
        recommendations = []

        if data.temperature > 30:
            recommendations.append(
                "Increase forest canopy restoration."
            )

        if data.rainfall < 1000:
            recommendations.append(
                "Implement water conservation measures."
            )

        if data.species_count < 100:
            recommendations.append(
                "Strengthen species protection programs."
            )

        if data.ndvi < 0.4:
            recommendations.append(
                "Improve vegetation cover through afforestation."
            )

        if not recommendations:
            recommendations.append(
                "Maintain current conservation efforts."
            )

        return {
            "biodiversity_health_score": round(score, 2),
            "species_risk_level": risk,
            "ecosystem_stability_index": stability_index,
            "biodiversity_resilience_index": resilience_index,
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):

    try:

        # Read CSV
        df = pd.read_csv(file.file)

        print("\nCSV HEAD")
        print(df.head())

        print("\nCSV COLUMNS")
        print(df.columns.tolist())

        # Encode Forest Type
        df["Forest_Type"] = (
            df["Forest_Type"]
            .astype(str)
            .str.strip()
        )

        df["Forest_Type"] = encoder.transform(
            df["Forest_Type"]
        )

        # Select only model features
        feature_df = df[
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

        # Predict
        predictions = model.predict(feature_df)

        # Add prediction column
        df["Predicted_Score"] = predictions

        # Average score
        avg_score = round(
            float(predictions.mean()),
            2
        )

        # Overall risk
        if avg_score >= 80:
            risk = "Low"
        elif avg_score >= 50:
            risk = "Medium"
        else:
            risk = "High"

        # Top 3 most vulnerable ecosystems
        high_risk = (
            df.sort_values(
                by="Predicted_Score"
            )
            .head(3)
        )

        high_risk_regions = []

        for index, row in high_risk.iterrows():

            high_risk_regions.append({
                "region": f"Region {index + 1}",
                "score": round(
                    float(row["Predicted_Score"]),
                    2
                )
            })

        # Medium risk ecosystems
        medium_risk = df[
            (df["Predicted_Score"] >= 50)
            &
            (df["Predicted_Score"] < 80)
        ]

        medium_risk_regions = []

        for index, row in medium_risk.head(3).iterrows():

            medium_risk_regions.append({
                "region": f"Region {index + 1}",
                "score": round(
                    float(row["Predicted_Score"]),
                    2
                )
            })

        return {
            "rows_analyzed": len(df),
            "average_health_score": avg_score,
            "risk_level": risk,
            "high_risk_regions": high_risk_regions,
            "medium_risk_regions": medium_risk_regions
        }

    except Exception as e:

        import traceback

        print("\nFULL ERROR")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )