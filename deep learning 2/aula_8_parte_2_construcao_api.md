# Laboratório 8 (Parte 2) — Construindo a API de Inferência (Deep Learning 2)

> **Continuação da Parte 1.** Lá nós preparamos os artefatos (`preprocessor.pkl`,
> `minmaxscaler.pkl`, `best_model.pth`), montamos o projeto local e deixamos o `api.py`
> com os imports e as constantes de caminho. Agora vamos **escrever a API de fato**.

Retomando, o `api.py` estava assim no fim da Parte 1:

```python
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
```

Tudo o que vem a seguir é continuação **deste mesmo arquivo**.

---

## Instanciar a aplicação

A primeira coisa é criar o objeto que **é** a API. Por convenção, chamamos essa variável
de `app` e a instanciamos com a classe `FastAPI`:

```python
# Instanciar a API
app = FastAPI(title="Preço de Veículos - API Inferência")
```

- O `title` é só um rótulo da API — ele aparece, por exemplo, na documentação automática
  que o FastAPI gera.
- É esse objeto `app` que o servidor (Uvicorn) vai procurar para **colocar a API no ar**.
  Guarde esse nome: ele vai ser referenciado tanto pelos endpoints (`@app.get`,
  `@app.post`) quanto pela linha de comando que sobe o servidor (na próxima aula).

---

## O modelo de dados de entrada (Pydantic)

Antes de receber requisições, definimos **o formato** dos dados que a API aceita. Fazemos
isso com uma classe que herda de `BaseModel`, a classe principal do Pydantic:

```python
# Modelo de dados de entrada
class Payload(BaseModel):
  records: List[Dict[str, Any]]
```

Lendo o tipo da direita para a esquerda: `records` é uma **lista** (`List`) de
**dicionários** (`Dict`) cuja chave é texto (`str`) e cujo valor pode ser **qualquer
coisa** (`Any`). Isso casa exatamente com o `payload.json` da Parte 1:

```json
{
  "records": [
    { "Categoria": 4, "Cor": "azul", "Potencia": 143, ... }
  ]
}
```

> **Por que `Any` e não tipos específicos?** Aqui estamos sendo **propositalmente pouco
> restritos**. Poderíamos validar campo a campo (`Categoria` é inteiro, `Cor` é texto,
> etc.), e aí o Pydantic **recusaria** uma requisição com tipo errado, retornando erro
> antes de o código rodar. Neste exemplo deixamos solto (`Any`) para focar no fluxo da
> inferência — a validação detalhada fica para exemplos futuros. Mas perceba o poder da
> ferramenta: com o Pydantic, a validação acontece "de graça" só por declarar os tipos.

---

## Trazer a classe da rede neural

Aqui está um ponto importante que vem lá da Parte 1: quando salvamos o modelo com
`torch.save(model.state_dict(), ...)`, gravamos **apenas os pesos**, não a arquitetura.
Os pesos sozinhos são só uma "caixa de números" — eles precisam de uma estrutura onde
encaixar.

Por isso a API precisa **conhecer a mesma arquitetura** usada no treino. Trazemos a
classe `NeuralNetwork` exatamente como ela estava no notebook:

```python
# Mesma arquitetura usada no treino (3+ camadas ocultas, ReLU como ativação)
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
```

> 💡 **Tem como evitar essa duplicação?** Sim. O ideal, num projeto real, seria colocar a
> classe `NeuralNetwork` em um arquivo separado (ex.: `model.py`) e **importá-la** tanto
> no notebook quanto na API. Assim existe **uma única definição** da arquitetura, sem
> risco de as duas versões saírem do sincronismo. Aqui, por uma questão **didática**,
> copiamos a mesma classe para dentro do `api.py`.

A regra de ouro: **a arquitetura na API tem que ser idêntica à do treino**. Se as camadas
não baterem, os pesos do `best_model.pth` não vão encaixar.

---

## Uma função para carregar os artefatos do scikit-learn

Uma pequena função utilitária que carrega o `preprocessor` e o `minmaxscaler` do disco:

```python
# Carregar pré-processador e scaler do alvo
def carregar_preprocessor():
  preprocessor = joblib.load(PREPROCESSOR_PATH)
  maxminscaler = joblib.load(MAXMIN_SCALER_PATH)
  return preprocessor, maxminscaler
```

Ela apenas faz dois `joblib.load` e devolve os dois objetos. Vamos chamá-la dentro do
`/predict`:

- o **`preprocessor`** será aplicado nos dados de **entrada**;
- o **`maxminscaler`** será usado para **inverter** a escala da predição (lembre da
  Parte 1: a rede prevê em escala 0–1 e precisamos voltar para reais).

---

## Definindo os endpoints — o `/health`

Agora começa a parte de **expor** funções como endpoints REST. No FastAPI isso é feito com
um **decorator** (o `@`) em cima da função, referenciando o `app` que criamos:

```python
@app.get("/health")
def health():
  return {"status": "ok"}
```

Lendo o decorator:

- `@app` → referencia a aplicação criada lá em cima.
- `.get` → o **verbo REST** desse endpoint (uma consulta simples).
- `"/health"` → a **URL** que dispara essa função. Ao acessar
  `http://127.0.0.1:8000/health`, o FastAPI executa `health()` e devolve `{"status":
  "ok"}`.

Esse é um **health check**: um endpoint padrão para responder "estou no ar?". Aqui ele só
retorna `ok`, mas poderia ter uma lógica de verdade — checar se os artefatos carregam, se
o modelo responde, etc. — para garantir que a API está realmente pronta para uso.

---

## O endpoint principal — o `/predict`

Esse é o coração da API. Ele recebe os dados de um carro e devolve o preço previsto.

```python
@app.post("/predict")
def predict(request: Payload):
  ...
```

Dois detalhes da assinatura:

- `@app.post("/predict")` → usamos o verbo **POST** (estamos **enviando** dados no corpo
  da requisição) na URL `/predict`.
- `request: Payload` → o `request` é **forçado** a ter o tipo `Payload` (aquela classe do
  Pydantic). Esse `nome: Tipo` é o *type enforcing* do Python: se a requisição não vier no
  formato esperado, o Pydantic acusa o erro **antes** de a função rodar.

### O fluxo interno é o mesmo do notebook

A grande sacada é que o `/predict` apenas **repete o caminho do treino**, só que sem a
parte de treinar. Vamos por partes.

**1) Carregar os artefatos:**

```python
  # Carregar artefatos
  preprocessor, maxminscaler = carregar_preprocessor()
```

**2) Extrair os registros do payload e virar um DataFrame:**

```python
  # Carregar dados do payload
  records = request.records

  # Carregar DataFrame com o payload
  df_veiculos = pd.DataFrame(records)
```

Aqui pegamos a lista de dicionários de dentro do `Payload` e a transformamos num
`DataFrame` do pandas — exatamente o mesmo tipo de estrutura que alimentava o
pré-processamento no notebook. A diferença é que agora só temos o **X** (os carros), não
o **y** (o preço) — afinal, o preço é o que queremos prever.

**3) Aplicar o pré-processamento (só `transform`):**

```python
  # Pré-processar os dados de entrada
  X_proc = preprocessor.transform(df_veiculos)
```

Como o `preprocessor` **já foi treinado** (o `fit` aconteceu lá no notebook), aqui usamos
**só `transform`**. É o equivalente a gerar um `X_val`/`X_test`: aplicamos as mesmas
transformações de coluna (one-hot + min-max) que o modelo espera.

**4) Recriar o modelo e carregar os pesos:**

```python
  # Carregar o modelo treinado
  input_size = X_proc.shape[1]
  modelo = NeuralNetwork(input_size=input_size, hidden_layer_sizes=[64, 32, 16, 8],
                          output_size=1)
  modelo.load_state_dict(torch.load(MODEL_PATH))
```

Passo a passo:

- `input_size = X_proc.shape[1]` → o número de **colunas** após o pré-processamento. Esse
  número **tem que bater** com o do treino: como aplicamos a mesma transformação no mesmo
  formato de entrada, a quantidade de colunas é a mesma — e a primeira camada da rede
  precisa ter esse tamanho de entrada.
- Instanciamos a `NeuralNetwork` com a **mesma arquitetura treinada** (mesmas camadas
  ocultas e `output_size`). Nesse momento o modelo existe, mas com pesos aleatórios.
- `modelo.load_state_dict(torch.load(MODEL_PATH))` → encadeamos duas operações:
  `torch.load` lê o dicionário de pesos do `best_model.pth`, e `load_state_dict` os
  injeta no modelo. Agora o modelo tem **a arquitetura certa + os melhores pesos**
  (aqueles escolhidos pelo early stopping na Parte 1).

> ⚠️ **Atenção:** o `hidden_layer_sizes` aqui deve refletir **exatamente** a arquitetura
> com que o `best_model.pth` foi treinado. Se você treinou com camadas diferentes, ajuste
> esse parâmetro — senão os pesos não encaixam e o `load_state_dict` falha.

**5) Converter para tensor e fazer a inferência:**

```python
  # Converter dados para tensor
  X_tensor = torch.tensor(X_proc, dtype=torch.float32)
  modelo.eval()
  with torch.no_grad():
    outputs = modelo(X_tensor)

  print(outputs)
```

- `torch.tensor(X_proc, dtype=torch.float32)` → no treino, o `DataLoader` convertia os
  dados em tensores automaticamente; aqui fazemos isso "na mão".
- `modelo.eval()` → coloca o modelo em **modo de avaliação**. Isso é importante: camadas
  como `Dropout` e `BatchNorm1d` se comportam diferente em treino e em inferência. Sem o
  `eval()`, a previsão sairia instável.
- `with torch.no_grad():` → desliga o cálculo de gradientes (não vamos treinar), deixando
  a inferência mais leve e rápida.
- `outputs = modelo(X_tensor)` → aplica o input no modelo e gera a saída.
- `print(outputs)` → opcional, só para depurar. Esse valor vai aparecer **no console do
  Uvicorn** quando a API for chamada. Lembre que essa saída está na **escala 0–1**.

**6) Inverter a escala e responder:**

```python
  # Inverter a escala dos resultados
  preds = maxminscaler.inverse_transform(outputs).reshape(-1).tolist()

  return {"predictions": preds}
```

- `maxminscaler.inverse_transform(outputs)` → traz a previsão de volta para a **escala de
  preço** (de reais), usando o scaler do alvo treinado na Parte 1.
- `.reshape(-1)` → achata para uma única dimensão.
- `.tolist()` → vira uma lista Python comum (um array do NumPy/tensor não é serializável
  direto em JSON).
- `return {"predictions": preds}` → como é uma API REST, devolvemos um **dicionário**, que
  o FastAPI serializa em **JSON**. A chave `predictions` carrega o(s) preço(s)
  previsto(s).

---

## O `api.py` completo

Juntando tudo, este é o arquivo final desta aula:

```python
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

# Instanciar a API
app = FastAPI(title="Preço de Veículos - API Inferência")

# Modelo de dados de entrada
class Payload(BaseModel):
  records: List[Dict[str, Any]]

# Mesma arquitetura usada no treino
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

# Carregar pré-processador e scaler do alvo
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

  # Carregar DataFrame com o payload
  df_veiculos = pd.DataFrame(records)

  # Pré-processar os dados de entrada
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
```

---

## Recapitulando e próximos passos

Nesta aula você construiu a API completa:

- [x] Instanciou a aplicação com `FastAPI(title=...)`.
- [x] Definiu o **modelo de dados de entrada** com Pydantic (`Payload`), entendendo o
      papel da validação.
- [x] Trouxe a **mesma arquitetura** da rede (`NeuralNetwork`) — necessária porque só
      salvamos os pesos.
- [x] Criou a função `carregar_preprocessor()` para ler os artefatos `.pkl`.
- [x] Implementou o endpoint **`/health`** (health check).
- [x] Implementou o endpoint **`/predict`**, repetindo o caminho do notebook: DataFrame →
      `transform` → tensor → inferência → `inverse_transform` → resposta JSON.

O código que expõe o modelo como API está pronto. **Na próxima aula** vamos colocar isso
no ar com o **Uvicorn** e fazer a primeira chamada de verdade com **curl**, enviando o
`payload.json` e recebendo o preço previsto em JSON.
