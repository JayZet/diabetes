from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# Load the model and scaler
model = joblib.load('/home/jayzet/coba_data/diabetes/diabetes_model.pkl')
scaler = joblib.load('/home/jayzet/coba_data/diabetes/scaler.pkl')

app = FastAPI()
app.mount("/home/jayzet/coba_data/diabetes/static", StaticFiles(directory="static"), name="static")

class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int

@app.post('/predict')
def predict_diabetes(data: DiabetesInput):
    # Convert input to numpy array
    input_data = np.array([[
        data.pregnancies, data.glucose, data.blood_pressure,
        data.skin_thickness, data.insulin, data.bmi,
        data.diabetes_pedigree, data.age
    ]])

    # Scale the input data
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    return {
        'prediction': int(prediction[0]),
        'probability': float(probability[0])
    }

@app.get("/")
def read_index():
    return FileResponse('/home/jayzet/coba_data/diabetes/static/index.html')
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
