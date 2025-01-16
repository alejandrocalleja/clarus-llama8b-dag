import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn

# Definir el modelo FastAPI
app = FastAPI()

# Cargar el modelo desde un archivo local
model_path = "model/model.pkl"  # Reemplaza con la ruta a tu modelo
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Definir el esquema de entrada para las predicciones
class PredictionInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convertir los datos de entrada a un array NumPy
    input_values = np.array([[input_data.fixed_acidity, input_data.volatile_acidity,
                              input_data.citric_acid, input_data.residual_sugar,
                              input_data.chlorides, input_data.free_sulfur_dioxide,
                              input_data.total_sulfur_dioxide, input_data.density,
                              input_data.pH, input_data.sulphates, input_data.alcohol]])
    
    # Realizar la predicción
    prediction = model.predict(input_values)[0]
    
    # Retornar la predicción
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    # Iniciar el servidor FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
