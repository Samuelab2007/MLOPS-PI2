from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib
from io import StringIO
import numpy as np
import pandas as pd
import os

app = FastAPI(title="API de Inferencia MLOps", version="1.1")

MODEL_PATH = os.path.join("models", "test_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo: {e}")

@app.get("/")
def home():
    return {"message": "API de predicción activa. Usa POST /predict"}

@app.post("/predict")
def predict_csv(file: UploadFile = File(...)):
    try:
        # Leer el archivo como DataFrame
        contents = file.file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        # Validar que el dataset tenga columnas
        if df.empty or df.shape[1] == 0:
            raise ValueError("El archivo CSV está vacío o sin columnas.")

        # Hacer predicciones
        X = df.values
        y_pred = model.predict(X)

        return {"predictions": y_pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
