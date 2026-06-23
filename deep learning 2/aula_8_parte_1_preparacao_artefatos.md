# Laboratório 8 (Parte 1) — Preparação dos Artefatos para a API (Deep Learning 2)

> **Pré-requisito:** você já treinou a rede neural de previsão de preço de veículos
> nas aulas anteriores. Aqui **não** vamos treinar nada de novo — vamos preparar o
> terreno para colocar esse modelo no ar como uma **API**.

---

## Contexto e objetivo

Até agora todo o trabalho aconteceu dentro do notebook: você carregou os dados,
pré-processou, treinou a rede e avaliou os resultados. Mas um modelo que só roda no
notebook não serve para ninguém de fora — para que outras aplicações usem suas
previsões, ele precisa ser exposto como um **serviço**: uma API.

Esta aula é a etapa **preparatória** desse processo. Ainda **não** vamos escrever a
lógica da API (isso fica para a próxima aula). O objetivo aqui é responder a uma
pergunta simples:

> *"O que eu preciso tirar do notebook e levar para fora dele para conseguir fazer
> uma previsão sem precisar treinar o modelo de novo?"*

A resposta são os **artefatos**: arquivos salvos em disco que carregam tudo o que o
modelo aprendeu. No fim desta aula você terá três artefatos prontos, um projeto local
montado e o esqueleto do `api.py` com os imports e as constantes.

---

## A ideia central: a API tem que reproduzir o treino

Esse é o conceito que amarra a aula inteira, então vale parar nele.

Quando você treinou o modelo, o dado **bruto** (um carro com `Cor = "azul"`,
`Potencia = 143`, etc.) passou por uma sequência de transformações antes de chegar na
rede neural:

1. Texto virou número (one-hot encoding em `Cor`, `Pais de Origem`, ...).
2. Números foram colocados numa escala comum (min-max scaling em `Potencia`,
   `Kilometragem`, ...).
3. O preço (o alvo) também foi escalado para o intervalo de 0 a 1.

A rede **nunca viu o dado bruto** — ela só entende esse formato transformado. Logo, na
hora de prever, o dado que chega na API **precisa passar pela mesma transformação,
com os mesmos parâmetros**. Se você transformar de um jeito diferente, está
alimentando a rede com algo que ela não reconhece, e a previsão sai errada.

E "os mesmos parâmetros" é a parte crucial: o min-max scaler, por exemplo, **aprendeu**
qual é o mínimo e o máximo de cada coluna **a partir do seu dataset de treino**. Esses
números fazem parte do objeto. Por isso não basta criar um scaler novo na API — ele
estaria "vazio", sem ter aprendido nada. Você precisa **exatamente o mesmo objeto**
que foi treinado.

É daí que vem a necessidade de **exportar** (salvar em disco) esses objetos.

---

## Por que exportar o `preprocessor`

O `preprocessor` é um `ColumnTransformer` do scikit-learn. Ele empacota, num único
objeto, todas as transformações de **entrada** (as features `X`):

```python
# Trecho do notebook (célula de pré-processamento)
numeric_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, reduced_categorical_features)
    ]
)
```

Repare que ele faz duas coisas ao mesmo tempo:

- **`OneHotEncoder`** nas colunas categóricas — transforma texto (`"azul"`,
  `"Alemanha"`, `"Híbrido"`) em colunas numéricas 0/1.
- **`MinMaxScaler`** nas colunas numéricas — coloca os números numa escala comum.

Quando você chamou `preprocessor.fit_transform(X_train)`, ele **aprendeu**:
- quais categorias existem em cada coluna (para montar as colunas do one-hot);
- o mínimo e o máximo de cada coluna numérica.

Esse "conhecimento" fica guardado **dentro** do objeto `preprocessor`. Na API, você vai
receber um carro em formato bruto e chamar `preprocessor.transform(...)` para colocá-lo
exatamente no formato que a rede espera. Por isso ele precisa ser salvo e carregado tal
como ficou após o treino.

> ⚠️ **Atenção:** use sempre `transform` (e **nunca** `fit_transform`) na hora de prever.
> `fit` reaprende os parâmetros a partir do dado novo — e o dado novo pode ser um único
> carro. Você quer aplicar o que foi aprendido no treino, não reaprender.

---

## Por que exportar o `scaler_y` (MinMaxScaler do alvo)

Esse é o ponto que costuma gerar mais dúvida, então vamos com calma.

No treino, além de escalar as features, você também escalou o **alvo** (o preço de
venda) para o intervalo de 0 a 1:

```python
# Trecho do notebook (mesma célula dos splits)
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_val   = scaler_y.transform(y_val.reshape(-1, 1))
y_test  = scaler_y.transform(y_test.reshape(-1, 1))
```

Consequência: o modelo aprendeu a prever **nessa escala de 0 a 1**, não em reais. Quando
a API rodar a rede, o número que sai não é um preço — é algo como `0.37`.

Para transformar esse `0.37` de volta em "R$ 87.000", usamos o `scaler_y` no sentido
inverso:

- **`transform`** → pega o valor original (preço) e o leva para a escala 0–1.
- **`inverse_transform`** → pega o valor na escala 0–1 e o traz de volta para a escala
  original (preço).

> 🔁 **Analogia:** pense numa tradução. `transform` traduz de "reais" para o "idioma do
> modelo" (0 a 1). `inverse_transform` traduz de volta do "idioma do modelo" para
> "reais". Para traduzir de volta corretamente, você precisa do **mesmo dicionário** que
> usou na ida — e esse dicionário é o `scaler_y` treinado, que conhece o mínimo e o
> máximo dos preços do seu dataset.

Por isso o `scaler_y` também precisa ser exportado. Instanciar um `MinMaxScaler` novo na
API não funcionaria: ele não saberia qual é o preço mínimo e máximo aprendidos, e a
inversão devolveria um número sem sentido.

---

## Salvando os artefatos no notebook

Com o "porquê" entendido, salvar é a parte fácil. São duas chamadas com `joblib` para os
objetos do scikit-learn:

```python
# Salvar pré-processador e scaler do alvo para uso futuro
import joblib

joblib.dump(preprocessor, 'model/preprocessor.pkl')
joblib.dump(scaler_y,     'model/minmaxscaler.pkl')
```

O `joblib.dump` faz a **serialização**: transforma o objeto Python (com tudo o que ele
aprendeu) em um arquivo `.pkl` no disco. Depois é só `joblib.load(...)` para trazê-lo de
volta intacto.

### E o modelo treinado?

O modelo PyTorch é salvo durante o próprio loop de treino, junto com o **early stopping**.
A cada época em que a perda de validação melhora, guardamos o melhor estado do modelo:

```python
# Trecho do loop de treino — parte do early stopping
if best_val_loss - epoch_val_loss > min_delta:
    best_val_loss = epoch_val_loss
    best_epoch = epoch
    epochs_no_improve = 0
    # Salvar o melhor modelo até agora
    torch.save(model.state_dict(), 'model/best_model.pth')
else:
    epochs_no_improve += 1
```

Dois detalhes importantes aqui:

- **`model.state_dict()`** guarda apenas os **pesos** aprendidos (os parâmetros), não a
  classe da rede. É a forma recomendada no PyTorch: é mais leve e portável. O preço a
  pagar é que, na API, você precisará **recriar a mesma arquitetura** da rede para depois
  carregar os pesos nela (veremos isso na próxima aula).
- O `torch.save` está **dentro do `if`** do early stopping de propósito: assim o arquivo
  `best_model.pth` sempre reflete a **melhor** época, e não a última (que pode já estar
  pior por overfitting).

---

## Resultado: 3 artefatos em `model/`

Depois de rodar o notebook, a pasta `model/` deve conter:

| Arquivo                 | O que é                              | Para que serve na API                          |
| ----------------------- | ------------------------------------ | ---------------------------------------------- |
| `preprocessor.pkl`      | `ColumnTransformer` treinado         | Transformar o dado bruto de entrada (one-hot + min-max nas features) |
| `minmaxscaler.pkl`      | `MinMaxScaler` do alvo (`scaler_y`)  | `inverse_transform` da previsão para a escala de preço |
| `best_model.pth`        | Pesos da rede (`state_dict`)         | Carregar o modelo treinado para fazer a previsão |

Esses três arquivos são tudo o que carrega o "conhecimento" do treino. Com eles em mãos,
você nunca mais precisa do dataset original nem de treinar de novo para fazer uma
previsão.

---

## Levando os artefatos para a máquina local

No curso, o treino aconteceu no **Google Colab**. A API, porém, vamos construir e rodar
**localmente**. Por quê?

- A API precisa de um **servidor web** rodando continuamente (o Uvicorn, veja adiante),
  algo que o Colab — pensado para notebooks — não foi feito para hospedar de forma
  confortável. Rodar local evita "gambiarras" para manter o serviço no ar.
- Localmente você tem controle total sobre o ambiente, os arquivos e o terminal, que é
  como uma API normalmente é desenvolvida e testada.

Então o passo manual é: **baixar do Colab** os três artefatos e o arquivo de dados, e
montar a estrutura do projeto local mais ou menos assim:

```
deep learning 2/
├── api.py                 # o código da API (vamos construir)
├── payload.json           # exemplo de requisição para testar
├── requirements.txt       # dependências do projeto
├── data/                  # o arquivo de veículos usado no curso
└── model/
    ├── preprocessor.pkl
    ├── minmaxscaler.pkl
    └── best_model.pth
```

Os caminhos dentro do `api.py` (próxima seção) assumem exatamente essa organização — os
artefatos dentro de uma pasta `model/`.

---

## O payload de teste (`payload.json`)

Para testar a API sem precisar montar a requisição na mão toda vez, deixamos um exemplo
pronto: o `payload.json`. Ele representa o **corpo (body)** que será enviado para a API.

```json
{
  "records": [
    {
      "Categoria": 4,
      "Cor": "azul",
      "Pais de Origem": "Alemanha",
      "Ano Modelo": 2026,
      "Ano Fabricação": 2025,
      "Potencia": 143,
      "Quantidade de lugares": 4,
      "Unico dono?": 0,
      "Ja teve sinistro?": 0,
      "Ja foi carro de aplicativo?": 1,
      "Revisoes em dia?": 1,
      "Sistema avancado de Multimidia?": 1,
      "Tipo de Motorizacao": "Híbrido",
      "Kilometragem": 60459,
      "Tipo de Transmissao": 3,
      "Tamanho do porta malas": 430
    }
  ]
}
```

Pontos a observar:

- **`records` é uma lista.** Isso permite mandar **vários carros** de uma vez. Aqui há só
  um registro, mas o formato já está pronto para lotes.
- **Os campos são exatamente os mesmos** do dataset de treino, com os mesmos nomes. Isso
  não é coincidência: lembre que o `preprocessor` espera reconhecer essas colunas para
  aplicar as transformações corretas. Este exemplo é, inclusive, o **primeiro registro**
  do próprio dataset.
- O valor que queremos prever (o preço de venda) **não** está no payload — é justamente o
  que a API vai devolver.

---

## O `requirements.txt` — as dependências

Para o projeto rodar em qualquer máquina, fixamos as versões das bibliotecas:

```txt
pandas==2.3.3
numpy==2.3.4
fastapi==0.120.1
uvicorn[standard]==0.38.0
joblib==1.5.2
scikit-learn==1.6.1
torch==2.8.0
```

Instale com:

```bash
pip install -r requirements.txt
```

Em relação ao que você já usava no treino, há **três pacotes novos** que merecem
explicação:

- **`fastapi`** — o framework que permite expor funções do seu código como uma **API
  REST**. É ele que transforma uma função Python comum em um endpoint HTTP que outras
  aplicações podem chamar.
- **`uvicorn`** — o **servidor web** (ASGI) que mantém a API no ar. O FastAPI define
  *o que* a API faz; o Uvicorn é quem efetivamente **fica escutando** as requisições e
  responde. Repare que ele aparece no `requirements`, mas **não** será importado no
  código — ele é chamado pelo terminal para subir o servidor.

E os pacotes que você já conhece, mas que continuam necessários:

- **`pandas`** — usado para transformar o payload recebido em um `DataFrame`, do mesmo
  jeito que no treino, antes de aplicar o `preprocessor`.
- **`joblib`** — carrega os artefatos `.pkl` (`preprocessor` e `minmaxscaler`).
- **`scikit-learn`** — necessário porque os objetos dentro dos `.pkl` são do sklearn;
  sem ele instalado, o `joblib.load` não consegue reconstruí-los.
- **`torch`** — recria a rede e carrega os pesos do `best_model.pth`.
- **`numpy`** — dependência de base usada por trás de tudo isso.

> Há também o **`pydantic`**, que será importado no código mas não aparece listado
> explicitamente: ele vem **junto com o FastAPI** como dependência. É a biblioteca de
> **validação de dados** — definiremos com ela o "modelo de dados" que a API aceita.

---

## O começo do `api.py` — imports e constantes

Com tudo preparado, criamos o arquivo `api.py`. Nesta aula ele ainda fica só no começo:
os **imports** e as **constantes** com os caminhos dos artefatos.

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

Por que cada import está aqui:

- **`typing` (`List`, `Dict`, `Any`)** — para descrever o formato dos dados de entrada na
  validação com o Pydantic (por exemplo: "uma lista de dicionários").
- **`joblib`** — para carregar os artefatos `.pkl` (`preprocessor` e `minmaxscaler`).
- **`pandas`** — para transformar o payload recebido em um `DataFrame`, repetindo o mesmo
  fluxo do treino (DataFrame → `preprocessor.transform` → tensor).
- **`fastapi` (`FastAPI`)** — a classe que instancia a aplicação/API.
- **`pydantic` (`BaseModel`)** — a base para definir o **modelo de dados** que a API
  recebe (o formato do payload).
- **`torch`** — para carregar os pesos (`torch.load`) e rodar a inferência com tensores.
- **`torch.nn` (`nn`)** — para recriar a arquitetura da rede neural (`NeuralNetwork`) na
  qual os pesos serão carregados.

E as três **constantes** apenas centralizam os caminhos dos artefatos, seguindo a
estrutura de pastas montada na seção 7:

- `MODEL_PATH` → os pesos do modelo (`best_model.pth`).
- `PREPROCESSOR_PATH` → o `ColumnTransformer` (one-hot + min-max das **features**).
- `MAXMIN_SCALER_PATH` → o `MinMaxScaler` do **alvo**, usado para o `inverse_transform`
  da previsão.

> 💡 **Observação do professor:** seria natural pensar em importar `MinMaxScaler` direto
> do scikit-learn aqui. Mas **não é preciso**: como o scaler já está salvo no `.pkl`, o
> `joblib.load` o reconstrói por completo, com a classe e tudo. Importar a classe
> manualmente só faria sentido se fôssemos **instanciar** um scaler novo — e não é o
> caso.

---

## Recapitulando e próximos passos

Nesta aula você:

- [x] Entendeu **por que** a API precisa reproduzir exatamente o pré-processamento do
      treino.
- [x] Exportou o **`preprocessor`** (transformação das features de entrada).
- [x] Exportou o **`scaler_y`** (`minmaxscaler.pkl`) para inverter a escala da previsão.
- [x] Salvou os **pesos do modelo** (`best_model.pth`) via `torch.save(state_dict)` junto
      ao early stopping.
- [x] Levou os **3 artefatos** para um projeto **local** com a estrutura de pastas
      correta.
- [x] Preparou o **`payload.json`** de teste e o **`requirements.txt`** com as
      dependências.
- [x] Criou o começo do **`api.py`** com imports e constantes, entendendo o papel de cada
      um.

Tudo o que carrega o conhecimento do treino já está em disco e o esqueleto da API está de
pé. **Na próxima aula** vamos preencher o `api.py` de fato: recriar a arquitetura da rede,
carregar os pesos, definir os endpoints (`/health` e `/predict`), subir o servidor com o
**Uvicorn** e fazer a primeira chamada com **curl** usando o `payload.json`.
