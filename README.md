# Legal RAG Backend API

Backend API per il sistema RAG legale basato su Google Vertex AI, Gemini e Discovery Engine.

## Funzionalità

- Ricerca avanzata con query expansion e re-ranking semantico
- Risposta RAG con citazioni inline delle fonti
- Interfaccia REST per integrazioni con frontend

## Prerequisiti

- Progetto Google Cloud con API abilitate:
  - Vertex AI API
  - Discovery Engine API
  - Cloud Run
  - Cloud Build
- Data store configurato in Vertex AI Search
- Autorizzazioni di accesso alle API

## Configurazione Ambiente di Sviluppo

1. **Installazione dipendenze**

```bash
pip install -r requirements.txt
```

2. **Configurazione Variabili d'Ambiente**

```bash
# Configurazione Google Cloud
export PROJECT_ID="tuo-project-id"
export PROJECT_NUM="tuo-project-number"
export DISCOVERY_LOCATION="eu"
export DISCOVERY_ENGINE_ID="tuo-engine-id"
export DATA_STORE_ID="tuo-data-store-id"

# Configurazione Gemini
export GEMINI_MODEL="gemini-2.5-pro"
export GEMINI_LOCATION="europe-west4"
```

3. **Avvio dell'applicazione in sviluppo**

```bash
uvicorn main:app --reload
```

## Deployment su Cloud Run

### Opzione 1: Deployment automatico con Cloud Build

1. Configura il repository con Cloud Build:

```bash
./setup-cloud-run.sh
```

2. Modifica il file `cloudbuild.yaml` con i tuoi parametri specifici.

3. Effettua un push dei cambiamenti sul branch collegato per attivare la build.

### Opzione 2: Deployment manuale

1. Costruisci l'immagine Docker localmente:

```bash
docker build -t gcr.io/[PROJECT_ID]/legal-rag-api .
```

2. Push dell'immagine su Container Registry:

```bash
docker push gcr.io/[PROJECT_ID]/legal-rag-api
```

3. Deploy su Cloud Run:

```bash
gcloud run deploy legal-rag-api \
  --image=gcr.io/[PROJECT_ID]/legal-rag-api \
  --region=europe-west1 \
  --platform=managed \
  --allow-unauthenticated
```

## Configurazione delle Variabili d'Ambiente su Cloud Run

Le seguenti variabili d'ambiente devono essere configurate nel servizio Cloud Run:

- `PROJECT_ID`: ID del tuo progetto Google Cloud
- `PROJECT_NUM`: Numero del tuo progetto Google Cloud
- `DISCOVERY_LOCATION`: Regione della Discovery Engine (default: "eu")
- `DISCOVERY_ENGINE_ID`: ID del tuo Engine
- `DISCOVERY_COLLECTION`: Nome della collezione (default: "default_collection")
- `DATA_STORE_ID`: ID del tuo Data Store
- `BRANCH_ID`: ID del branch (default: "default_branch")
- `GEMINI_MODEL`: Nome del modello Gemini (default: "gemini-2.5-pro")
- `GEMINI_LOCATION`: Regione del modello Gemini (default: "europe-west4")

## Variabili per Configurazioni Avanzate

- `USE_RERANKING`: Attiva/disattiva il re-ranking semantico (default: "true")
- `USE_QUERY_EXPANSION`: Attiva/disattiva l'espansione delle query (default: "true")
- `INITIAL_RESULTS_COUNT`: Numero di risultati da recuperare per il re-ranking (default: 20)
- `QUERY_EXPANSION_COUNT`: Numero di variazioni della query da generare (default: 3)
- `EMBEDDING_MODEL`: Modello di embedding da utilizzare (default: "textembedding-gecko@003")

## Endpoint API

### `GET /health`

Verifica lo stato dell'API.

### `POST /chat`

Endpoint principale per il chatbot RAG.

Payload:
```json
{
  "question": "Quali sono gli obblighi GDPR?",
  "include_sources": true,
  "max_tokens": 4096,
  "temperature": 0.2,
  "debug": false
}
```

### `GET /search` e `POST /search`

Effettua una ricerca diretta sui documenti.

### `GET /debug/*`

Endpoint di debug per testare le diverse componenti del sistema.

## Contribuire

Per contribuire al progetto, crea un fork del repository, implementa le tue modifiche e invia una pull request.