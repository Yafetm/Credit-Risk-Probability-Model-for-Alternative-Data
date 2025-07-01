# src/api/main.py
from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Load model with raw string path
model = joblib.load(r"C:/Users/hp/Desktop/Kifiya AIM/week 5/Technical Content/workspace/models/rf_model.pkl")

@app.post("/predict")
async def predict(data: dict):
    # Convert input data to DataFrame (expecting RFM features)
    df = pd.DataFrame([data])
    X = df[['Recency', 'Frequency', 'Monetary']].values
    prediction = model.predict(X)[0]
    return {"credit_risk_probability": int(prediction)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)