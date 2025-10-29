from typing import List, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

MODEL_PATH = "model/best_model.pth"
PREPROCESSOR_PATH = "model/preprocessor.pkl"
MAXMIN_SCALER_PATH = "model/minmaxscaler.pkl"

# Instaciar a API

app = FastAPI(title="Preço de Veículos - API Inferência")

# Modelo de dados de entrada
class Payload(BaseModel):
  records: List[Dict[str, Any]]

# Criar uma arquitetura de rede neural com 3 camadas ocultas, usando ReLU como função de ativação
class NeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_layer_sizes=[128, 64, 32, 16], output_size=1, dropout_rate=0.2):
    super(NeuralNetwork, self).__init__()
    self.layer1 = nn.Linear(input_size, hidden_layer_sizes[0])
    self.bn1 = nn.BatchNorm1d(hidden_layer_sizes[0])
    self.dropout1 = nn.Dropout(dropout_rate)

    self.layer2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
    self.bn2 = nn.BatchNorm1d(hidden_layer_sizes[1])
    self.dropout2 = nn.Dropout(dropout_rate)

    self.layer3 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
    self.bn3 = nn.BatchNorm1d(hidden_layer_sizes[2])
    self.dropout3 = nn.Dropout(dropout_rate)

    self.layer4 = nn.Linear(hidden_layer_sizes[2], hidden_layer_sizes[3])
    self.bn4 = nn.BatchNorm1d(hidden_layer_sizes[3])
    self.dropout4 = nn.Dropout(dropout_rate)

    self.output_layer = nn.Linear(hidden_layer_sizes[3], output_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.bn1(self.layer1(x)))
    x = self.dropout1(x)

    x = self.relu(self.bn2(self.layer2(x)))
    x = self.dropout2(x)

    x = self.relu(self.bn3(self.layer3(x)))
    x = self.dropout3(x)

    x = self.relu(self.bn4(self.layer4(x)))
    x = self.dropout4(x)

    x = self.output_layer(x)
    return x
    
# Carregar pré-processor
def carregar_preprocessor():
  preprocessor = joblib.load(PREPROCESSOR_PATH)
  maxminscaler = joblib.load(MAXMIN_SCALER_PATH)
  return preprocessor, maxminscaler

@app.get("/health")
def health():
  return {"status": "ok"}

@app.post("/predict")
def predict(request: Payload):
  # Carregar artefatos
  preprocessor, maxminscaler = carregar_preprocessor()

  # Carregar dados do payload
  records = request.records

  # Carregar Dataframe com Payload
  df_veiculos = pd.DataFrame(records)

  # Preprocessor dados de entrada
  X_proc = preprocessor.transform(df_veiculos)

  # Carregar o modelo treinado
  input_size = X_proc.shape[1]
  modelo = NeuralNetwork(input_size=input_size, hidden_layer_sizes=[64, 32, 16, 8],
                          output_size=1)
  modelo.load_state_dict(torch.load(MODEL_PATH))

   # Converter dados para tensor
  X_tensor = torch.tensor(X_proc, dtype=torch.float32)
  modelo.eval()
  with torch.no_grad():
    outputs = modelo(X_tensor)
  
  print(outputs)

  # Inverter a escala dos resultados
  preds = maxminscaler.inverse_transform(outputs).reshape(-1).tolist()

  return {"predictions": preds}
