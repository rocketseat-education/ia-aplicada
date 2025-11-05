from typing import List, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras

MODEL_PATH = 'model/best_model.keras'
PREPROCESSOR_PATH = 'model/preprocessor.pkl'
MAXMINSCALER_PATH = 'model/minmaxscaler.pkl'

# Instanciar a API
app = FastAPI(title="Preço de Veículos - API Inferência")

# Modelo de dados de entrada
class Payload(BaseModel):
    records: List[Dict[str, Any]]

# Carregar artefatos
def carregar_artefatos():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    maxminscaler = joblib.load(MAXMINSCALER_PATH)
    modelo = keras.models.load_model(MODEL_PATH)

    return preprocessor, maxminscaler, modelo

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: Payload):

    # Carregar artefatos
    preprocessor, maxminscaler, modelo = carregar_artefatos()

    # Carregar dados do Payload
    records = request.records

    # Carregar Dataframe com Payload
    df_veiculos = pd.DataFrame(records)

    # Preprocessar dados de entrada
    X_proc = preprocessor.transform(df_veiculos)

    # Fazer previsões
    outputs = modelo.predict(X_proc)

    print(outputs)

    # Inverter a escala dos resultados
    preds = maxminscaler.inverse_transform(outputs).reshape(-1).tolist()

    return {"predictions": preds}