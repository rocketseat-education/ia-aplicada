# Laboratório 8 (Parte 3) — Subindo a API e Testando (Deep Learning 2)

> **Continuação das Partes 1 e 2.** Já preparamos os artefatos e escrevemos o `api.py`
> completo. Agora vamos **colocar a API no ar** com o Uvicorn e fazer a **primeira chamada
> de verdade** com o `curl`, usando o `payload.json`.
>
> 💡 Esta etapa não tem código novo — é tudo no **terminal**. Por isso ela não gera um
> commit: nenhum arquivo do projeto muda, apenas executamos comandos.

Para testar, vamos trabalhar com **dois terminais** abertos lado a lado (no VS Code:
*Terminal → New Terminal*):

- **Terminal 1** → roda o servidor (fica ocupado, "preso", segurando a API no ar).
- **Terminal 2** → faz as chamadas de teste.

---

## Subir o servidor com o Uvicorn (Terminal 1)

Lembre da Parte 1: o **Uvicorn** é o servidor web que mantém a API no ar. Ele não é
importado no código — é chamado pela linha de comando, apontando para a nossa aplicação.

A partir da pasta do projeto (`deep learning 2/`, onde está o `api.py`):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Decifrando o comando:

- **`uvicorn`** → o servidor.
- **`api:app`** → o padrão é `<módulo>:<variável>`. Ou seja: "no arquivo `api.py`, suba o
  objeto chamado `app`" — aquela variável que criamos na Parte 2 com
  `app = FastAPI(...)`.
- **`--host 0.0.0.0`** → em qual endereço escutar. `0.0.0.0` aceita conexões locais (e da
  rede); para testes locais você acessa via `127.0.0.1` (localhost).
- **`--port 8000`** → a porta em que a API vai responder.
- **`--reload`** → **hot reload**: toda vez que você salvar uma alteração no `api.py`, o
  Uvicorn reinicia sozinho. Não precisa derrubar e subir o servidor a cada mudança. (Útil
  em desenvolvimento; em produção, não se usa `--reload`.)

> 📁 **Sobre o caminho `api:app`.** No vídeo, o instrutor roda a partir de uma pasta raiz
> com o código dentro de `src/`, por isso usa `uvicorn src.api:app`. O padrão é sempre o
> mesmo — `<caminho.do.módulo>:<variável>` — muda só o prefixo conforme a sua estrutura de
> pastas. Como aqui o `api.py` está direto na pasta do projeto, usamos `api:app`.

Ao executar, o Uvicorn mostra algo como "Application startup complete" e fica
**escutando**. A API está no ar e pronta para receber requisições. **Deixe esse terminal
rodando** e abra um segundo.

---

## Testar com `curl` (Terminal 2)

O `curl` é uma ferramenta de linha de comando para fazer requisições HTTP — perfeita para
testar uma API rapidinho. No segundo terminal (também na pasta do projeto, onde está o
`payload.json`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @payload.json
```

Decifrando:

- **`-X POST`** → o **verbo** da requisição. Tem que ser `POST`, porque foi assim que
  declaramos o endpoint (`@app.post("/predict")`).
- **`http://127.0.0.1:8000/predict`** → o endereço: localhost, porta 8000, endpoint
  `/predict`.
- **`-H "Content-Type: application/json"`** → o **header** dizendo que estamos enviando
  dados em formato JSON.
- **`-d @payload.json`** → os **dados** (o corpo da requisição). O `@` indica que o
  conteúdo vem de um **arquivo** (`payload.json`), em vez de ser digitado direto na linha.

> 🔧 O `curl` é a forma mais simples, mas não a única. Você poderia testar com **Postman**,
> **Thunder Client** (extensão do VS Code) ou outras ferramentas. Todas fazem a mesma
> coisa: enviar a requisição e mostrar a resposta.

### A resposta

Ao dar *Enter*, a API responde com um JSON parecido com este:

```json
{"predictions": [388840.88]}
```

Esse é o **preço de venda estimado** para o carro do `payload.json`: cerca de
**R$ 388.840,88**.

---

## Acompanhando os dois lados

É instrutivo olhar os dois terminais ao mesmo tempo:

- **No Terminal 2 (curl)** você vê a resposta final, já na escala de preço:
  `{"predictions": [388840.88]}`.
- **No Terminal 1 (Uvicorn)** aparece o `print(outputs)` que deixamos no `/predict` — o
  tensor **antes** da inversão de escala, algo como:

  ```
  tensor([[0.0381]])
  ```

Isso fecha o ciclo que discutimos na Parte 1: a rede prevê na **escala 0–1** (`0.0381`) e
o `inverse_transform` do `minmaxscaler` traz esse número de volta para a **escala de
preço** (`388840.88`). Os dois lados conversando confirmam que toda a cadeia está
funcionando.

> Você também pode checar o health check no navegador ou via curl:
> `curl http://127.0.0.1:8000/health` → `{"status": "ok"}`.

---

## O quadro completo — o que construímos no Laboratório 8

Com a API respondendo, fechamos o caminho inteiro, do treino à entrega:

1. **Salvamos os artefatos** (Parte 1): os pesos do modelo (`best_model.pth`) e os
   pré-processadores que precisamos reutilizar (`preprocessor.pkl` e `minmaxscaler.pkl`).
2. **Codificamos a API** com FastAPI (Parte 2), lembrando de:
   - **validar** o modelo de dados de entrada com o **Pydantic**;
   - **trazer a arquitetura** da rede neural para o código (já que só salvamos os pesos);
   - **repetir o mesmo processo do notebook** para transformar o dado bruto recebido no
     **tensor** que o modelo espera, passando pelos pré-processadores;
   - **inverter** a saída com o mesmo `MinMaxScaler` do alvo, para devolver um preço na
     escala real.
3. **Subimos e testamos** a API (Parte 3) com Uvicorn + curl.

> 💭 **Por que escalar o `y` (o alvo)?** Poderíamos ter deixado o preço na escala original.
> Optamos por colocá-lo também em min-max, no **mesmo range** das demais variáveis, para o
> treino ficar mais estável (os valores não "dançarem" em magnitudes muito diferentes). O
> preço dessa escolha é justamente ter que **inverter** a escala no final — o que fazemos
> com o `minmaxscaler` salvo.

---

## Conclusão

Com isso encerramos o **Laboratório 8** e o módulo de **Deep Learning 2**, em que o foco
foi usar o **PyTorch** para um problema de **regressão**: partimos do dataset e do
problema, desenvolvemos a rede ao longo do módulo e finalizamos **entregando o modelo como
uma API** — pronto para ser consumido por outras aplicações.

Da próxima vez que precisar de uma previsão, não há notebook nem treino: basta a API no ar
e uma requisição com o payload. 🚗💸
