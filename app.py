import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load your saved pipeline
model = joblib.load("credit_risk_pipeline.pkl")

app = FastAPI(title="Credit Risk Prediction API")

# Defined input schema matching model feature requirements
class ApplicantData(BaseModel):
    EmploymentType: str
    CreditScore: float
    Income: float
    LoanIncomeRatio: float
    AgeGroup: str
    LoanAmount: float
    City: str
    DebtToIncome: float

@app.post("/predict")
def predict_risk(data: ApplicantData):
    # Convert input to DataFrame (pipelines expect pandas DataFrames)
    input_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    input_df = pd.DataFrame([input_dict])

    # Predict class and probability
    prediction = model.predict(input_df)[0]
    probability = (
        model.predict_proba(input_df)[0][1]
        if hasattr(model, "predict_proba")
        else None
    )

    # Apply the tuned decision threshold of 0.70 from the analysis
    threshold = 0.70
    is_default_risk = int(probability >= threshold) if probability else int(prediction)

    return {
        "prediction": is_default_risk,
        "default_probability": round(float(probability), 4) if probability else None,
        "threshold_applied": threshold,
        "status": "High Risk / Rejected" if is_default_risk == 1 else "Approved",
    }
