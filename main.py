# main.py
import os
import time
import re
import json
import hashlib
import threading
import html2text
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import NotFound
from google.cloud import discoveryengine_v1 as de
from google.protobuf.json_format import MessageToDict
import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform_v1.services.prediction_service import PredictionServiceClient
from google.cloud.aiplatform_v1.types.prediction_service import PredictRequest
from vertexai.generative_models import GenerativeModel # Aggiungi questa linea
import google.auth
from google.auth.transport.requests import AuthorizedSession
import uvicorn

# ========= Env =========
PROJECT_ID = os.getenv("PROJECT_ID")
PROJECT_NUM = os.getenv("PROJECT_NUM")  # es. "250932538739"
LOCATION = os.getenv("DISCOVERY_LOCATION", "eu")
ENGINE_ID = os.getenv("DISCOVERY_ENGINE_ID")
COLLECTION = os.getenv("DISCOVERY_COLLECTION", "default_collection")
DATA_STORE_ID = os.getenv("DATA_STORE_ID", "")
BRANCH_ID = os.getenv("BRANCH_ID", "default_branch")
MAX_PAGE_SIZE = int(os.getenv("MAX_PAGE_SIZE", "10"))  # tienilo basso per la quota

# ========= Gemini AI Config =========
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "europe-west4")  # "eu" non è supportato, usa una regione specifica
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))
GEMINI_TOP_K = int(os.getenv("GEMINI_TOP_K", "40"))

# ========= RAG Config =========
RAG_MAX_CHUNKS = int(os.getenv("RAG_MAX_CHUNKS", "5"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0"))  # Soglia minima di pertinenza
RAG_CHUNK_SEPARATOR = os.getenv("RAG_CHUNK_SEPARATOR", "\n---\n")

if not PROJECT_ID or not ENGINE_ID or not PROJECT_NUM:
    raise RuntimeError("Missing env: PROJECT_ID, PROJECT_NUM, DISCOVERY_ENGINE_ID")

API_ENDPOINT = f"{LOCATION}-discoveryengine.googleapis.com"
SERVING = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}"
    f"/collections/{COLLECTION}/engines/{ENGINE_ID}/servingConfigs/default_search"
)
DOC_PARENT = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}"
    f"/collections/{COLLECTION}/dataStores/{DATA_STORE_ID}/branches/{BRANCH_ID}"
    if DATA_STORE_ID else ""
)

# ========= Clients =========
# SDK per getDocument (lettura diretta)
doc_client = de.DocumentServiceClient(client_options=ClientOptions(api_endpoint=API_ENDPOINT))

# Inizializzazione del client Vertex AI per Gemini
try:
    aiplatform.init(project=PROJECT_ID, location=GEMINI_LOCATION)
except Exception as e:
    print(f"Warning: Vertex AI initialization failed: {e}")
    # Non blocchiamo l'avvio dell'app se l'inizializzazione fallisce
    # L'errore verrà gestito quando si tenterà di utilizzare il modello

# ========= Cache & Rate =========
_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = int(os.getenv("CACHE_TTL_SEC", "600"))

_rate_lock = threading.Lock()
_last_calls_ts: List[float] = []
_RATE_WINDOW = float(os.getenv("RATE_WINDOW_SEC", "1.2"))
_RATE_MAX = int(os.getenv("RATE_MAX_CALLS", "2"))

# ========= Schemas =========
class SearchIn(BaseModel):
    query: str
    page_size: int = 3
    mode: str = "CHUNKS"  # default utile

class ChatIn(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list, 
        description="Lista opzionale di messaggi precedenti nel formato [{role: 'user|assistant', content: '...'}]")
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    include_sources: bool = True  # Se true, include i riferimenti alle fonti nella risposta
    debug: bool = False # <-- AGGIUNGI QUESTA RIGA

class Source(BaseModel):
    title: str
    link: str
    score: float = 0.0

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)
    debug: Optional[Dict[str, Any]] = None
    sources: List[Source] = Field(default_factory=list)

# ========= App =========
app = FastAPI(title="vertex-mini", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monta i file statici
app.mount("/static", StaticFiles(directory="static"), name="static")

# ========= Helpers =========
def _rate_allow() -> bool:
    now = time.time()
    with _rate_lock:
        while _last_calls_ts and now - _last_calls_ts[0] > _RATE_WINDOW:
            _last_calls_ts.pop(0)
        if len(_last_calls_ts) >= _RATE_MAX:
            return False
        _last_calls_ts.append(now)
        return True

def _cache_key(q: str, page_size: int, mode: str) -> str:
    s = f"{q}|{page_size}|{mode}|v=rest+extractive"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _get_cache(k: str):
    with _cache_lock:
        v = _cache.get(k)
        if not v:
            return None
        if time.time() - v["ts"] > _CACHE_TTL:
            _cache.pop(k, None)
            return None
        return v["data"]

def _set_cache(k: str, data: Any):
    if not data:
        return
    with _cache_lock:
        _cache[k] = {"ts": time.time(), "data": data}

def _search_rest(query: str, page_size: int, mode: str) -> Dict[str, Any]:
    """Chiama l'endpoint REST :search (stesso della Console)."""
    creds, _ = google.auth.default(quota_project_id=PROJECT_ID)
    authed = AuthorizedSession(creds)

    url = (
        f"https://{LOCATION}-discoveryengine.googleapis.com/v1alpha/"
        f"projects/{PROJECT_NUM}/locations/{LOCATION}/collections/{COLLECTION}"
        f"/engines/{ENGINE_ID}/servingConfigs/default_search:search"
    )

    body = {
        "query": query,
        "pageSize": max(1, min(int(page_size or 3), MAX_PAGE_SIZE)),
        "languageCode": "it",
        "queryExpansionSpec": {"condition": "AUTO"},
        "spellCorrectionSpec": {"mode": "AUTO"},
        "contentSearchSpec": {
            # DOCUMENTS o CHUNKS
            "searchResultMode": (mode.upper() if mode else "CHUNKS"),
            "snippetSpec": {"returnSnippet": True, "maxSnippetCount": 3},
            "extractiveContentSpec": {
                "maxExtractiveAnswerCount": 1,
                "maxExtractiveSegmentCount": 3,
                "returnExtractiveSegmentScore": True,
            },
        },
        "userInfo": {"timeZone": "Europe/Rome"},
    }

    resp = authed.post(url, data=json.dumps(body), headers={"Content-Type": "application/json"})
    if resp.status_code == 429:
        raise HTTPException(status_code=429, detail="Quota Search esaurita (REST).")
    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()

def _pack_rest_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalizza i risultati REST in {title, link, text, score, id}."""
    print("\n=== ESTRAZIONE TESTO DAI RISULTATI ===")
    print(f"• Elaborazione di {len(items)} risultati grezzi")
    
    out: List[Dict[str, Any]] = []
    for idx, j in enumerate(items or [], 1):
        print(f"\n• RISULTATO {idx}/{len(items)}:")
        
        doc = j.get("document") or {}
        derived = doc.get("derivedStructData") or {}
        structd = doc.get("structData") or {}

        # Score e ID per tracciamento
        score = float(j.get("relevanceScore", j.get("matchingScore", 0.0)) or 0.0)
        doc_id = doc.get("id", f"doc-{idx}")
        
        print(f"  - ID: {doc_id}")
        print(f"  - Score: {score:.4f}")

        # Estrazione di titolo e link
        title = (
            derived.get("title") or structd.get("title") or
            derived.get("file_name") or structd.get("file_name") or
            doc.get("name") or doc.get("id") or str(idx)
        )
        print(f"  - Titolo: {title[:60]}")
        
        link = (
            derived.get("link") or structd.get("link") or
            derived.get("uri") or structd.get("uri") or
            derived.get("url") or structd.get("url") or
            doc.get("id") or ""
        )
        print(f"  - Link: {link[:60]}")

        # Estrazione del testo con log dettagliati
        text = ""
        extraction_source = "NESSUNO"
        
        # 1) Priorità al contenuto del chunk (extractive_answers)
        extractive_answers = derived.get("extractive_answers")
        if not text and extractive_answers and isinstance(extractive_answers, list) and len(extractive_answers) > 0:
            print("  - Tentativo 1: Cerco in derived.extractive_answers...")
            answer_content = extractive_answers[0].get("content")
            if answer_content and isinstance(answer_content, str):
                text = answer_content
                extraction_source = "derived.extractive_answers"
                print(f"    ✅ Trovato testo ({len(text)} caratteri)")
            else:
                print("    ❌ Nessun testo trovato")

        # 2) Fallback sul campo 'content' in derivedStructData
        if not text:
            print("  - Tentativo 2: Cerco in derived.content...")
            chunk_content = derived.get("content")
            if chunk_content and isinstance(chunk_content, str):
                text = chunk_content
                extraction_source = "derived.content"
                print(f"    ✅ Trovato testo ({len(text)} caratteri)")
            else:
                print("    ❌ Nessun testo trovato")

        # 3) Fallback su snippet
        if not text:
            print("  - Tentativo 3: Cerco nei snippets...")
            snippets = j.get("snippets")  # A volte è a livello radice del risultato
            if not snippets:
                snippets = (j.get("snippetInfo", {}) or {}).get("snippets", [])
            
            if snippets and isinstance(snippets, list):
                print(f"    Trovati {len(snippets)} snippets")
                for i, s in enumerate(snippets):
                    if s and isinstance(s, dict) and s.get("snippet"):
                        text = s["snippet"]
                        extraction_source = f"snippet[{i}]"
                        print(f"    ✅ Trovato testo in snippet {i} ({len(text)} caratteri)")
                        break
            if not text:
                print("    ❌ Nessun testo trovato nei snippets")

        # 4) Fallback su extractive answers/segments a livello radice
        if not text:
            print("  - Tentativo 4: Cerco in extractiveAnswers/Segments...")
            eas = j.get("extractiveAnswers") or []
            ess = j.get("extractiveSegments") or []
            
            if eas and isinstance(eas, list) and len(eas) > 0 and eas[0].get("content"):
                text = eas[0]["content"]
                extraction_source = "extractiveAnswers"
                print(f"    ✅ Trovato testo in extractiveAnswers ({len(text)} caratteri)")
            elif ess and isinstance(ess, list) and len(ess) > 0 and ess[0].get("content"):
                text = ess[0]["content"]
                extraction_source = "extractiveSegments"
                print(f"    ✅ Trovato testo in extractiveSegments ({len(text)} caratteri)")
            else:
                print("    ❌ Nessun testo trovato")

        # 5) Fallback finale sul contenuto del documento intero
        if not text:
            print("  - Tentativo 5: Cerco in document.content...")
            cont = doc.get("content") or {}
            raw = cont.get("rawText") or cont.get("content") or ""
            if isinstance(raw, str) and raw:
                text = raw
                extraction_source = "document.content"
                print(f"    ✅ Trovato testo in document.content ({len(text)} caratteri)")
            else:
                print("    ❌ Nessun testo trovato")
        
        # Risultato finale
        if text:
            clean_text = text.strip()[:800]  # Limita la lunghezza
            preview = clean_text[:100].replace("\n", " ")
            preview += "..." if len(clean_text) > 100 else ""
            
            print(f"  - ✅ TESTO ESTRATTO: {len(clean_text)} caratteri da {extraction_source}")
            print(f"    Preview: {preview}")
            
            out.append({
                "id": doc_id,
                "title": title,
                "link": link,
                "text": clean_text,
                "score": score,
            })
        else:
            print(f"  - ❌ RISULTATO SCARTATO: Nessun testo trovato in nessuna fonte")
    
    # Riepilogo finale
    scartati = len(items) - len(out)
    percentuale_scartata = (scartati / len(items) * 100) if items else 0
    
    print(f"\n• RIEPILOGO ESTRAZIONE TESTO:")
    print(f"  - Totale risultati processati: {len(items)}")
    print(f"  - Risultati con testo: {len(out)}")
    print(f"  - Risultati scartati: {scartati} ({percentuale_scartata:.1f}%)")
    
    # Ordina per score
    out.sort(key=lambda x: x.get("score", 0), reverse=True)
    print(f"  - Risultati finali ordinati per score")
    print("=== FINE ESTRAZIONE TESTO ===\n")
    
    return out


# ========= RAG Helper Functions =========
def _clean_html(text: str) -> str:
    """Pulisce il testo dai tag HTML e da altre formattazioni non desiderate."""
    if not text:
        return ""
    
    # Utilizza html2text per una conversione più accurata da HTML a testo
    h = html2text.HTML2Text()
    h.ignore_links = False  # Conserva i link come [testo](url)
    h.ignore_images = True
    h.ignore_tables = False
    h.body_width = 0  # Disabilita il wrapping
    
    # Prima conversione base con html2text
    text = h.handle(text)
    
    # Rimuovi entità HTML comuni che potrebbero essere rimaste
    html_entities = {
        "&#39;": "'", "&quot;": '"', "&lt;": "<", "&gt;": ">", 
        "&amp;": "&", "&nbsp;": " ", "\\n": "\n"
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)
    
    # Rimuovi il bold marker ** che html2text inserisce
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    return text.strip()


def _prepare_context(search_results: List[Dict[str, Any]]) -> tuple:
    """
    Prepara il contesto per l'LLM dai risultati di ricerca.
    
    Returns:
        tuple: (context_text, sources_list)
            - context_text: testo concatenato dei chunk
            - sources_list: lista delle fonti utilizzate
    """
    print("\n=== PREPARAZIONE CONTESTO ===")
    
    if not search_results:
        print("ERRORE: Nessun risultato di ricerca ricevuto")
        print("=== FINE PREPARAZIONE CONTESTO ===\n")
        return "", []
    
    # Stampa informazioni sui risultati grezzi
    print(f"1. Risultati totali ricevuti: {len(search_results)}")
    total_chars = sum(len(r.get("text", "")) for r in search_results)
    print(f"2. Totale caratteri nei risultati grezzi: {total_chars}")
    
    # Analizza i punteggi per debug
    scores = [r.get("score", 0) for r in search_results]
    if scores:
        print(f"3. Distribuzione score: min={min(scores):.4f}, max={max(scores):.4f}, media={sum(scores)/len(scores):.4f}")
    
    # Filtra i risultati per score minimo e limita il numero di chunk
    filtered_results = [
        r for r in search_results 
        if r.get("score", 0) >= RAG_MIN_SCORE
    ][:RAG_MAX_CHUNKS]
    
    print(f"4. Filtro applicato: score >= {RAG_MIN_SCORE}, max chunks = {RAG_MAX_CHUNKS}")
    print(f"5. Risultati dopo filtro: {len(filtered_results)}/{len(search_results)}")
    
    # Estrai e pulisci i testi
    chunks = []
    sources = []
    
    print(f"\n6. Elaborazione di ogni risultato:")
    for i, result in enumerate(filtered_results):
        # Pulisci il testo
        text = result.get("text", "")
        clean_text = _clean_html(text)
        
        print(f"\n   Chunk {i+1}/{len(filtered_results)}:")
        print(f"   • Score: {result.get('score', 0):.4f}")
        print(f"   • Titolo: {result.get('title', 'N/A')}")
        print(f"   • Link: {result.get('link', '')[:60]}{'...' if len(result.get('link', '')) > 60 else ''}")
        print(f"   • ID documento: {result.get('id', 'N/A')}")
        print(f"   • Testo originale: {len(text)} caratteri")
        print(f"   • Testo pulito: {len(clean_text)} caratteri")
        
        # Log dei primi caratteri del testo pulito
        if clean_text:
            preview = clean_text[:100].replace('\n', ' ') + ("..." if len(clean_text) > 100 else "")
            print(f"   • Preview: {preview}")
            chunks.append(clean_text)
            
            # Aggiungi la fonte
            sources.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "score": result.get("score", 0.0)
            })
        else:
            print(f"   • ATTENZIONE: Testo vuoto dopo la pulizia - chunk ignorato")
    
    # Concatena i chunk con un separatore
    context = RAG_CHUNK_SEPARATOR.join(chunks)
    
    print(f"\n7. Riepilogo finale:")
    print(f"   • Chunks utilizzati: {len(chunks)}/{len(filtered_results)}")
    print(f"   • Lunghezza contesto finale: {len(context)} caratteri")
    print(f"   • Separatore chunk: '{RAG_CHUNK_SEPARATOR}'")
    print(f"   • Fonti incluse: {len(sources)}")
    
    if context:
        start_preview = context[:150].replace('\n', ' ')
        end_preview = context[-150:].replace('\n', ' ') if len(context) > 300 else ""
        print(f"\n8. Preview contesto:")
        print(f"   Inizio: {start_preview}...")
        if end_preview:
            print(f"   Fine: ...{end_preview}")
    
    print("=== FINE PREPARAZIONE CONTESTO ===\n")
    
    return context, sources


def _build_prompt(question: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Costruisce il prompt completo per l'LLM.
    
    Args:
        question: La domanda dell'utente
        context: Il contesto estratto dai documenti
        history: Cronologia della conversazione (opzionale)
        
    Returns:
        str: Il prompt completo
    """
    # Sistema le istruzioni per il modello
    system_prompt = """Sei un assistente legale esperto che risponde a domande utilizzando solo le informazioni fornite nel CONTESTO.
Se l'informazione non è presente nel CONTESTO, rispondi "Non ho informazioni sufficienti per rispondere a questa domanda."
Non inventare o inferire informazioni che non sono esplicitamente indicate nel CONTESTO.
Struttura la tua risposta in modo chiaro e professionale, citando quando possibile la fonte specifica dell'informazione.
Rispondi in italiano."""
    
    print(f"\n=== COSTRUZIONE PROMPT ===")
    print(f"1. System Prompt: {len(system_prompt)} caratteri")
    print(f"2. Contesto: {len(context)} caratteri")
    print(f"3. Domanda: '{question}'")
    
    # Gestione della cronologia delle conversazioni, se presente
    history_text = ""
    if history and isinstance(history, list) and len(history) > 0:
        print(f"4. Storia della conversazione: {len(history)} messaggi")
        # La storia viene ignorata nell'implementazione attuale, ma possiamo loggare
        for i, msg in enumerate(history):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            print(f"   Messaggio {i+1}: {role} - {len(content)} caratteri")
    else:
        print("4. Nessuna storia di conversazione")
    
    # Aggiungi il contesto e la domanda al prompt
    prompt = f"{system_prompt}\n\nCONTESTO:\n{context}\n\nDOMANDA: {question}\n\nRISPOSTA:"
    
    print(f"5. Prompt finale: {len(prompt)} caratteri totali")
    # Stampa una preview del prompt completo
    prompt_preview = f"{prompt[:300]}...{prompt[-300:] if len(prompt) > 600 else ''}"
    print(f"6. Preview del prompt:\n{prompt_preview}")
    print(f"=== FINE COSTRUZIONE PROMPT ===\n")
    
    return prompt


def _generate_response(prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
    """
    Genera una risposta utilizzando il modello Gemini.
    
    Args:
        prompt: Il prompt completo
        max_tokens: Numero massimo di token nella risposta (opzionale)
        temperature: Temperatura per la generazione (opzionale)
        
    Returns:
        str: La risposta generata
    """
    print("\n=== GENERAZIONE RISPOSTA ===")
    start_time = time.time()
    
    # Log dei parametri di generazione
    actual_max_tokens = max_tokens or GEMINI_MAX_OUTPUT_TOKENS
    actual_temperature = temperature or GEMINI_TEMPERATURE
    
    print(f"1. Parametri di generazione:")
    print(f"   • Modello: {GEMINI_MODEL}")
    print(f"   • Regione: {GEMINI_LOCATION}")
    print(f"   • Max tokens output: {actual_max_tokens}")
    print(f"   • Temperature: {actual_temperature}")
    print(f"   • Top-p: {GEMINI_TOP_P}")
    print(f"   • Top-k: {GEMINI_TOP_K}")
    print(f"   • Lunghezza prompt: {len(prompt)} caratteri")
    
    try:
        print(f"\n2. Inizializzazione modello...")
        
        # Configura i parametri per il modello
        generation_config = {
            "max_output_tokens": actual_max_tokens,
            "temperature": actual_temperature,
            "top_p": GEMINI_TOP_P,
            "top_k": GEMINI_TOP_K
        }
        
        # Crea un'istanza del modello Gemini
        model = GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=generation_config
        )
        
        # Log prima della generazione
        print(f"\n3. Invio prompt al modello...")
        generation_start = time.time()
        
        # Genera la risposta
        response = model.generate_content(prompt)
        
        generation_time = time.time() - generation_start
        print(f"\n4. Risposta ricevuta in {generation_time:.2f} secondi")
        
        # Estrai il testo della risposta
        answer = response.text
        
        # Log della risposta
        answer_preview = answer[:300] + ("..." if len(answer) > 300 else "")
        print(f"\n5. Risposta generata ({len(answer)} caratteri):")
        print(f"   {answer_preview}")
        
        total_time = time.time() - start_time
        print(f"\n6. Tempo totale di elaborazione: {total_time:.2f} secondi")
        print("=== FINE GENERAZIONE RISPOSTA ===\n")
        
        return answer
    
    except Exception as e:
        error_time = time.time() - start_time
        print(f"\nERRORE dopo {error_time:.2f} secondi: {str(e)}")
        print(f"Traceback completo:")
        import traceback
        print(traceback.format_exc())
        print("=== FINE GENERAZIONE RISPOSTA (CON ERRORE) ===\n")
        
        return "Mi dispiace, ho riscontrato un problema nel generare una risposta. Riprova più tardi."
def _get_document_text(doc_id: str) -> Dict[str, Any]:
    """Legge il documento direttamente dal Data Store (richiede Serve content attivo)."""
    if not DATA_STORE_ID:
        raise HTTPException(status_code=400, detail="DATA_STORE_ID mancante")
    name = f"{DOC_PARENT}/documents/{doc_id}"
    try:
        doc = doc_client.get_document(name=name)
    except NotFound:
        raise HTTPException(status_code=404, detail="Documento non trovato")

    body = {"id": getattr(doc, "id", doc_id), "title": "", "link": "", "text": ""}

    try:
        derived = MessageToDict(getattr(doc, "derived_struct_data", {}))
        structd = MessageToDict(getattr(doc, "struct_data", {}))
    except Exception:
        derived, structd = {}, {}

    body["title"] = derived.get("title") or structd.get("title") or body["id"]
    body["link"] = derived.get("link") or structd.get("link") or ""

    try:
        cont = MessageToDict(getattr(doc, "content", {}))
        if isinstance(cont, dict):
            raw = cont.get("rawText") or cont.get("content") or ""
            if isinstance(raw, str):
                body["text"] = raw[:1200]
    except Exception:
        pass

    if not body["text"]:
        sn = derived.get("snippets") or structd.get("snippets") or []
        if isinstance(sn, list) and sn:
            x = sn[0]
            if isinstance(x, str):
                body["text"] = x[:1200]
            elif isinstance(x, dict) and "snippet" in x:
                body["text"] = x["snippet"][:1200]

    return body

def _search(query: str, page_size: int, mode: str) -> List[Dict[str, Any]]:
    print("\n=== RICERCA DISCOVERY ENGINE ===")
    print(f"• Query: '{query}'")
    print(f"• Modalità: {mode}")
    print(f"• Page size: {page_size}")
    search_start = time.time()
    
    if not _rate_allow():
        print("ERRORE: Rate limit superato!")
        raise HTTPException(status_code=429, detail="Rate limited, riprova tra un attimo")

    qn = re.sub(r"\s+", " ", (query or "").strip())[:500]
    pmode = (mode or "CHUNKS").upper()
    ps = max(1, min(int(page_size or 3), MAX_PAGE_SIZE))
    print(f"• Query normalizzata: '{qn}'")
    print(f"• Modalità effettiva: {pmode}")
    print(f"• Page size effettiva: {ps}")

    # Verifica cache
    ck = _cache_key(qn, ps, pmode)
    cached = _get_cache(ck)
    if cached is not None:
        cache_items = len(cached)
        print(f"• RISULTATO DALLA CACHE: {cache_items} items")
        # Log dettagliato dei risultati cached
        for i, result in enumerate(cached):
            print(f"  [{i+1}] Score: {result.get('score', 0):.4f} | Titolo: {result.get('title', 'N/A')[:60]}")
            print(f"      Link: {result.get('link', 'N/A')[:60]}")
            print(f"      ID: {result.get('id', 'N/A')}")
            text_preview = (result.get('text', '')[:100].replace('\n', ' ') + '...') if result.get('text') else 'No text'
            print(f"      Testo: {text_preview}")
        print(f"=== FINE RICERCA (CACHE) ===\n")
        return cached

    # 1) prova REST con mode richiesto
    print("• Esecuzione ricerca REST...")
    api_start = time.time()
    rest = _search_rest(qn, ps, pmode)
    api_time = time.time() - api_start
    print(f"• Chiamata API completata in {api_time:.2f} secondi")
    
    # Numero di risultati grezzi
    raw_results = rest.get("results", [])
    print(f"• Risultati grezzi ricevuti: {len(raw_results)}")
    
    # Log dettagliato dei risultati grezzi prima del processing
    print("\n• RISULTATI GREZZI DISCOVERY:")
    for i, result in enumerate(raw_results):
        doc = result.get("document", {})
        score = float(result.get("relevanceScore", result.get("matchingScore", 0.0)) or 0.0)
        doc_id = doc.get("id", "N/A")
        derived = doc.get("derivedStructData", {}) or {}
        title = derived.get("title", doc.get("name", f"Doc {i+1}"))
        print(f"  [{i+1}] Score: {score:.4f} | ID: {doc_id} | Titolo: {title[:60]}")
    
    # Elaborazione risultati
    pack_start = time.time()
    packed = _pack_rest_items(raw_results)
    pack_time = time.time() - pack_start
    
    # 2) fallback: se CHUNKS ha dato vuoto, prova DOCUMENTS
    if not packed and pmode == "CHUNKS":
        print("\n• FALLBACK: Nessun risultato in modalità CHUNKS, provo con DOCUMENTS")
        fallback_start = time.time()
        rest2 = _search_rest(qn, ps, "DOCUMENTS")
        fallback_time = time.time() - fallback_start
        print(f"• Fallback completato in {fallback_time:.2f} secondi")
        
        # Numero di risultati fallback
        raw_fallback = rest2.get("results", [])
        print(f"• Risultati fallback ricevuti: {len(raw_fallback)}")
        
        # Log dei risultati fallback
        print("\n• RISULTATI FALLBACK (DOCUMENTS):")
        for i, result in enumerate(raw_fallback):
            doc = result.get("document", {})
            score = float(result.get("relevanceScore", result.get("matchingScore", 0.0)) or 0.0)
            doc_id = doc.get("id", "N/A")
            derived = doc.get("derivedStructData", {}) or {}
            title = derived.get("title", doc.get("name", f"Doc {i+1}"))
            print(f"  [{i+1}] Score: {score:.4f} | ID: {doc_id} | Titolo: {title[:60]}")
        
        packed = _pack_rest_items(raw_fallback)

    # Log dei risultati elaborati finali
    print("\n• RISULTATI FINALI ELABORATI:")
    for i, result in enumerate(packed):
        print(f"  [{i+1}] Score: {result.get('score', 0):.4f} | Titolo: {result.get('title', 'N/A')[:60]}")
        print(f"      Link: {result.get('link', 'N/A')[:60]}")
        print(f"      ID: {result.get('id', 'N/A')}")
        text_preview = (result.get('text', '')[:100].replace('\n', ' ') + '...') if result.get('text') else 'No text'
        print(f"      Testo: {text_preview}")

    # Metti in cache i risultati
    if packed:
        _set_cache(ck, packed)
        print(f"• Risultati salvati in cache (chiave: {ck[:8]}...)")
    else:
        print("• Nessun risultato da salvare in cache")

    total_time = time.time() - search_start
    print(f"• Tempo totale ricerca: {total_time:.2f} secondi")
    print(f"• Totale risultati finali: {len(packed)}")
    print(f"=== FINE RICERCA ===\n")
    
    return packed

# ========= Routes =========
@app.post("/search")
def search_post(body: SearchIn):
    try:
        packed = _search(body.query, body.page_size, body.mode)
        # fallback ‘getdoc’ di 3 doc se packed vuoto o senza testo
        if DATA_STORE_ID and (not packed or all(not (x.get("text") or "").strip() for x in packed)):
            # usa una query corta per identificare id doc
            rest = _search_rest(re.sub(r"\s+", " ", body.query)[:200], 3, "DOCUMENTS")
            ids = [((i.get("document") or {}).get("id") or "") for i in rest.get("results", [])]
            out = []
            for i in ids[:3]:
                if i:
                    try:
                        out.append(_get_document_text(i))
                    except Exception:
                        continue
            if out:
                return {"ok": True, "results": out}
        return {"ok": True, "results": packed}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/search")
def search_get(q: str, page_size: int = 3, mode: str = "CHUNKS"):
    try:
        packed = _search(q, page_size, mode)
        if DATA_STORE_ID and (not packed or all(not (x.get("text") or "").strip() for x in packed)):
            rest = _search_rest(re.sub(r"\s+"," ",q)[:200], 3, "DOCUMENTS")
            ids = [((i.get("document") or {}).get("id") or "") for i in rest.get("results", [])]
            out = []
            for i in ids[:3]:
                if i:
                    try:
                        out.append(_get_document_text(i))
                    except Exception:
                        continue
            if out:
                return {"ok": True, "results": out}
        return {"ok": True, "results": packed}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/debug/raw")
def debug_raw(q: str, mode: str = "CHUNKS", page_size: int = 3):
    """Ritorna la risposta REST grezza per ispezione."""
    rest = _search_rest(q.strip(), max(1, min(page_size, MAX_PAGE_SIZE)), mode)
    return {"ok": True, "raw": rest}

@app.get("/debug/getdoc")
def debug_getdoc(id: str):
    if not DATA_STORE_ID:
        raise HTTPException(status_code=400, detail="DATA_STORE_ID mancante")
    return _get_document_text(id)

@app.get("/debug/serving")
def debug_serving():
    return {
        "serving": SERVING,
        "project": PROJECT_ID,
        "project_num": PROJECT_NUM,
        "location": LOCATION,
        "engine": ENGINE_ID,
        "collection": COLLECTION,
        "data_store": DATA_STORE_ID,
        "branch": BRANCH_ID
    }

@app.post("/debug/gemini")
async def debug_gemini(body: dict):
    """Endpoint per testare direttamente il modello Gemini."""
    try:
        # Configura i parametri del modello
        generation_config = {
            "max_output_tokens": body.get("max_tokens") or GEMINI_MAX_OUTPUT_TOKENS,
            "temperature": body.get("temperature") or GEMINI_TEMPERATURE,
            "top_p": GEMINI_TOP_P,
            "top_k": GEMINI_TOP_K
        }
        
        # Crea un'istanza del modello
        model = GenerativeModel(
            model_name=GEMINI_MODEL, 
            generation_config=generation_config
        )
        
        # Genera la risposta
        prompt = body.get("prompt", "Ciao, sono un assistente AI.")
        response = model.generate_content(prompt)
        
        return {"ok": True, "response": response.text, "model": GEMINI_MODEL, "location": GEMINI_LOCATION}
    
    except Exception as e:
        return {
            "ok": False, 
            "error": str(e), 
            "model": GEMINI_MODEL, 
            "location": GEMINI_LOCATION,
            "project": PROJECT_ID
        }

@app.get("/health")
def health():
    return {"ok": True, "project": PROJECT_ID, "location": LOCATION, "engine": ENGINE_ID}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatIn):
    """
    Endpoint per il chatbot RAG che combina ricerca e generazione.
    
    Accetta una domanda dell'utente, cerca le informazioni pertinenti e 
    genera una risposta basata sul contesto recuperato.
    """
    print("\n==================================================")
    print(f"NUOVA RICHIESTA CHAT: '{body.question}'")
    print("==================================================")
    
    start_time_total = time.time()
    
    try:
        # Creiamo un dizionario per tracciare tutto il processo
        rag_debug = {
            "question": body.question,
            "steps": [],
            "timestamps": {
                "start": start_time_total
            }
        }
        
        # Log dei parametri della richiesta
        print(f"\n=== PARAMETRI RICHIESTA ===")
        print(f"• Domanda: '{body.question}'")
        print(f"• Include fonti: {body.include_sources}")
        print(f"• Max tokens: {body.max_tokens or 'default'}")
        print(f"• Temperature: {body.temperature or 'default'}")
        print(f"• History messaggi: {len(body.history) if body.history else 0}")
        if body.history and len(body.history) > 0:
            for i, msg in enumerate(body.history):
                print(f"  - Messaggio {i+1}: {msg.get('role', 'unknown')} ({len(msg.get('content', ''))} caratteri)")
        
        # 1. Fase di Retrieval: Utilizza la funzione esistente di ricerca
        print(f"\n=== FASE 1: RETRIEVAL (CHUNKS) ===")
        retrieval_start = time.time()
        
        search_results = _search(body.question, RAG_MAX_CHUNKS, "CHUNKS")
        
        retrieval_time = time.time() - retrieval_start
        print(f"Tempo di retrieval: {retrieval_time:.2f} secondi")
        
        # Aggiungi al log
        rag_debug["steps"].append({
            "step": "retrieval_chunks",
            "time": retrieval_time,
            "results_count": len(search_results),
            "results": [{"id": r.get("id", ""), "title": r.get("title", ""), "score": r.get("score", 0)} for r in search_results]
        })
        
        # 2. Preparazione del contesto e delle fonti
        context_start = time.time()
        context, sources = _prepare_context(search_results)
        context_time = time.time() - context_start
        
        # Aggiungi al log
        rag_debug["steps"].append({
            "step": "context_preparation_chunks", 
            "time": context_time,
            "context_length": len(context),
            "sources_count": len(sources)
        })
        
        # Se non abbiamo trovato contenuto rilevante, cerchiamo di nuovo con modalità DOCUMENTS
        if not context:
            print(f"\n=== FASE 1b: RETRY RETRIEVAL (DOCUMENTS) ===")
            retry_start = time.time()
            
            search_results = _search(body.question, RAG_MAX_CHUNKS, "DOCUMENTS")
            
            retry_time = time.time() - retry_start
            print(f"Tempo di retrieval (DOCUMENTS): {retry_time:.2f} secondi")
            
            # Aggiungi al log
            rag_debug["steps"].append({
                "step": "retrieval_documents",
                "time": retry_time,
                "results_count": len(search_results),
                "results": [{"id": r.get("id", ""), "title": r.get("title", ""), "score": r.get("score", 0)} for r in search_results]
            })
            
            # Prepara il contesto dai nuovi risultati
            retry_context_start = time.time()
            context, sources = _prepare_context(search_results)
            retry_context_time = time.time() - retry_context_start
            
            # Aggiungi al log
            rag_debug["steps"].append({
                "step": "context_preparation_documents", 
                "time": retry_context_time,
                "context_length": len(context),
                "sources_count": len(sources)
            })
        
        # Se ancora non abbiamo contenuto, restituiamo un messaggio appropriato
        if not context:
            print(f"\n=== ERRORE: NESSUN CONTESTO TROVATO ===")
            
            # Aggiungi al log
            rag_debug["steps"].append({
                "step": "no_context_found"
            })
            rag_debug["timestamps"]["end"] = time.time()
            rag_debug["total_time"] = rag_debug["timestamps"]["end"] - rag_debug["timestamps"]["start"]
            
            # Stampa un sommario completo
            print(f"\n=== RIEPILOGO FINALE ===")
            print(f"• Totale passaggi: {len(rag_debug['steps'])}")
            print(f"• Tempo totale: {rag_debug['total_time']:.2f} secondi")
            print(f"• Risultato: FALLITO - Nessun contesto pertinente trovato")
            print("==================================================\n")
            
            return ChatResponse(
                answer="Non ho trovato informazioni pertinenti per rispondere alla tua domanda.",
                sources=[]
            )
        
        # 3. Costruzione del prompt
        prompt_start = time.time()
        prompt = _build_prompt(body.question, context, body.history)
        prompt_time = time.time() - prompt_start
        
        # Aggiungi al log
        rag_debug["steps"].append({
            "step": "prompt_construction",
            "time": prompt_time,
            "prompt_length": len(prompt)
        })
        
        # 4. Generazione della risposta
        print(f"\n=== FASE 3: LLM GENERATION ===")
        generation_start = time.time()
        
        answer = _generate_response(
            prompt, 
            max_tokens=body.max_tokens, 
            temperature=body.temperature
        )
        
        generation_time = time.time() - generation_start
        print(f"Tempo di generazione: {generation_time:.2f} secondi")
        
        # Aggiungi al log
        rag_debug["steps"].append({
            "step": "response_generation",
            "time": generation_time,
            "answer_length": len(answer)
        })
        
        # 5. Preparazione della risposta
        source_objects = [Source(**source) for source in sources] if body.include_sources else []
        
        # Completa il log
        rag_debug["timestamps"]["end"] = time.time()
        rag_debug["total_time"] = rag_debug["timestamps"]["end"] - rag_debug["timestamps"]["start"]
        
        # Stampa un sommario completo
        print(f"\n=== RIEPILOGO FINALE ===")
        print(f"• Totale passaggi: {len(rag_debug['steps'])}")
        print(f"• Tempo totale: {rag_debug['total_time']:.2f} secondi")
        print(f"• Fonti utilizzate: {len(sources)}")
        print(f"• Lunghezza risposta: {len(answer)} caratteri")
        print(f"• Risultato: SUCCESSO - Risposta generata")
        print("==================================================\n")
        
        return ChatResponse(
            answer=answer,
            sources=source_objects,
            debug=rag_debug if body.debug else None
        )
        
    except HTTPException as e:
        error_time = time.time() - start_time_total
        print(f"\n=== ERRORE HTTP ===")
        print(f"Tempo: {error_time:.2f} secondi")
        print(f"Status: {e.status_code}")
        print(f"Dettaglio: {e.detail}")
        print("==================================================\n")
        raise e
    except Exception as e:
        error_time = time.time() - start_time_total
        print(f"\n=== ERRORE GENERICO ===")
        print(f"Tempo: {error_time:.2f} secondi")
        print(f"Tipo: {type(e).__name__}")
        print(f"Messaggio: {str(e)}")
        print(f"Traceback:")
        import traceback
        print(traceback.format_exc())
        print("==================================================\n")
        raise HTTPException(status_code=502, detail=f"Errore durante l'elaborazione: {str(e)}")

@app.get("/", response_class=FileResponse)
def index():
    """Serve il file HTML statico dalla cartella static."""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), workers=1)