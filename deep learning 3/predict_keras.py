#  mlflow models serve -m "models:/passagens_classifier_keras/Production" -p 5002 --env-manager local --no-conda
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
import requests

PREPROCESSOR_PATH = 'model/preprocessor.pkl'
MODEL_API_URL = 'http://localhost:5002/invocations'

# Modelo de dados de entrada
class Payload(BaseModel):
    records: List[Dict[str, Any]]

# Carregar o preprocessador
def carregar_preprocessors():
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return preprocessor

def predict(request: Payload):

    # Carregar artefatos
    preprocessor = carregar_preprocessors()

    # Carregar dados do Payload
    records = request.records

    # Carregar Dataframe com Payload
    df_passagens = pd.DataFrame(records)

    # Preprocessar dados de entrada
    X_proc = preprocessor.transform(df_passagens)

    # Chamar o modelo treinado usando requests
    response = requests.post(MODEL_API_URL, json={"inputs": X_proc.tolist()})
    probs = response.json()
    predicted_classes = np.argmax(probs)
    classes = ['Comprou', 'Não comprou', 'Preferiu aguardar e comprou depois']
    preds = classes[predicted_classes]
    return {"predictions": preds}

if __name__ == "__main__":
    # Exemplo de uso
    sample_payload = Payload(
        records=[
            {   "Localizador": "ABC123",
                "Cia Aérea": "DELTA",
                "Origem": "LHR",
                "Destino": "JFK",
                "Classe da Passagem": 2,
                "Qtde de Paradas": 1,
                "Tipo de Destino": "negócios",
                "Estação do Ano": "inverno",
                "Preço Atual da Passagem": 500.0,
                "Preço da Passagem na última semana": 520.0,
                "Preço da Passagem no último mês": 550.0
            }
        ]
    )

    resultado = predict(sample_payload)
    print(resultado)