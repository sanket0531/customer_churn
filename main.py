from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and preprocessors
model = load_model("churn_model.h5", compile=False)
transformer = joblib.load("transformer.pkl")
scaler = joblib.load("scaler.pkl")

class customerdata(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def pridection(data:customerdata):
    
    new_customer = pd.DataFrame([data.model_dump()])

     # Apply preprocessing
    new_customer_transformed = transformer.transform(new_customer)
    new_customer_scaled = scaler.transform(new_customer_transformed)

      # Predict
    probability = float(model.predict(new_customer_scaled)[0][0])
    prediction = 1 if probability > 0.5 else 0

    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "result": "Customer will churn" if prediction == 1 else "Customer will stay"
    }