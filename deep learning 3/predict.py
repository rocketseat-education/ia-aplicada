from typing import List, Dict, Any

import pandas as pd
import joblib
import torch
from pydantic import BaseModel
import requests

PREPROCESSOR_PATH = 'model/preprocessor.pkl'
MODEL_API_URL = 'http://localhost:5001/invocations'

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
    logits = response.json()
    print(f"Logits:{logits}")

    # Transformar os logits de volta para classes originais, se necessário
    tensor_preds = torch.tensor(logits['predictions'], dtype=torch.float32)
    probs = torch.softmax(tensor_preds, dim=1)
    print(f"Probs:{probs}")
    predicted_classes = torch.argmax(probs, dim=1).tolist()
    classes = ['Comprou', 'Não comprou', 'Preferiu aguardar e comprou depois']
    preds = [classes[i] for i in predicted_classes]
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