# main.py
import os
import time
import re
import json
import hashlib
import threading
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import NotFound
from google.cloud import discoveryengine_v1 as de
from google.protobuf.json_format import MessageToDict

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

# ========= App =========
app = FastAPI(title="vertex-mini", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    out: List[Dict[str, Any]] = []
    for idx, j in enumerate(items or [], 1):
        doc = j.get("document") or {}
        derived = doc.get("derivedStructData") or {}
        structd = doc.get("structData") or {}

        # Estrazione di titolo e link (il tuo codice andava già bene)
        title = (
            derived.get("title") or structd.get("title") or
            derived.get("file_name") or structd.get("file_name") or
            doc.get("name") or doc.get("id") or str(idx)
        )
        link = (
            derived.get("link") or structd.get("link") or
            derived.get("uri") or structd.get("uri") or
            derived.get("url") or structd.get("url") or
            doc.get("id") or ""
        )

        text = ""
        
        # --- INIZIO MODIFICHE ---

        # 1) Priorità al contenuto del chunk, che è la fonte più probabile in modalità CHUNKS
        # Il contenuto del chunk si trova spesso in derivedStructData.extractive_answers o content
        if not text:
            # Prova con 'extractive_answers' che spesso contiene il testo del chunk
            extractive_answers = derived.get("extractive_answers")
            if extractive_answers and isinstance(extractive_answers, list) and len(extractive_answers) > 0:
                answer_content = extractive_answers[0].get("content")
                if answer_content and isinstance(answer_content, str):
                    text = answer_content

        # 2) Fallback sul campo 'content' direttamente dentro derivedStructData
        if not text:
            chunk_content = derived.get("content")
            if chunk_content and isinstance(chunk_content, str):
                text = chunk_content

        # 3) Fallback su snippet (utile per modalità DOCUMENTS)
        if not text:
            snippets = j.get("snippets")  # A volte è a livello radice del risultato
            if not snippets:
                snippets = (j.get("snippetInfo", {}) or {}).get("snippets", [])
            
            for s in snippets or []:
                if s.get("snippet"):
                    text = s["snippet"]
                    break

        # 4) Fallback su extractive answers/segments a livello radice
        if not text:
            eas = j.get("extractiveAnswers") or []
            ess = j.get("extractiveSegments") or []
            if eas and eas[0].get("content"):
                text = eas[0]["content"]
            elif ess and ess[0].get("content"):
                text = ess[0]["content"]

        # 5) Fallback finale sul contenuto del documento intero (raro)
        if not text:
            cont = doc.get("content") or {}
            raw = cont.get("rawText") or cont.get("content") or ""
            if isinstance(raw, str) and raw:
                text = raw

        # --- FINE MODIFICHE ---

        score = float(j.get("relevanceScore", j.get("matchingScore", 0.0) or 0.0))
        
        # Aggiungi solo se è stato trovato del testo
        if text:
            out.append({
                "id": doc.get("id", ""),
                "title": title,
                "link": link,
                "text": text.strip()[:800],  # Limita la lunghezza
                "score": score,
            })
    return out
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
    if not _rate_allow():
        raise HTTPException(status_code=429, detail="Rate limited, riprova tra un attimo")

    qn = re.sub(r"\s+", " ", (query or "").strip())[:500]
    pmode = (mode or "CHUNKS").upper()
    ps = max(1, min(int(page_size or 3), MAX_PAGE_SIZE))

    ck = _cache_key(qn, ps, pmode)
    cached = _get_cache(ck)
    if cached is not None:
        return cached

    # 1) prova REST con mode richiesto
    rest = _search_rest(qn, ps, pmode)
    packed = _pack_rest_items(rest.get("results", []))

    # 2) fallback: se CHUNKS ha dato vuoto, prova DOCUMENTS
    if not packed and pmode == "CHUNKS":
        rest2 = _search_rest(qn, ps, "DOCUMENTS")
        packed = _pack_rest_items(rest2.get("results", []))

    if packed:
        _set_cache(ck, packed)
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

@app.get("/health")
def health():
    return {"ok": True, "project": PROJECT_ID, "location": LOCATION, "engine": ENGINE_ID}

@app.get("/", response_class=HTMLResponse)
def index():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width,initial-scale=1"/>
      <title>Vertex Mini Search</title>
      <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;margin:24px;line-height:1.4}
        .row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
        input,select,button{font-size:16px;padding:10px;border:1px solid #ccc;border-radius:10px}
        input{flex:1;min-width:260px}
        button{cursor:pointer}
        .card{border:1px solid #eee;border-radius:16px;padding:16px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,.05)}
        .title{font-weight:600;margin-bottom:6px}
        .meta{opacity:.7;font-size:12px;margin-bottom:8px}
        .text{white-space:pre-wrap}
      </style>
    </head>
    <body>
      <h1>Vertex Mini Search</h1>
      <div class="row">
        <input id="q" placeholder="Cerca..." value="GDPR fornitore AI"/>
        <select id="mode">
          <option value="DOCUMENTS">DOCUMENTS</option>
          <option value="CHUNKS" selected>CHUNKS</option>
        </select>
        <select id="size">
          <option selected>3</option>
          <option>5</option>
          <option>8</option>
          <option>12</option>
        </select>
        <button id="go">Cerca</button>
      </div>
      <div id="out"></div>
      <script>
        const q=document.getElementById("q")
        const mode=document.getElementById("mode")
        const size=document.getElementById("size")
        const out=document.getElementById("out")
        const go=document.getElementById("go")
        async function run() {
          out.innerHTML="Caricamento..."
          const u=new URL(window.location.origin+"/search")
          u.searchParams.set("q",q.value||"")
          u.searchParams.set("mode",mode.value)
          u.searchParams.set("page_size",size.value)
          const r=await fetch(u.toString())
          if(!r.ok){out.innerHTML="Errore: "+r.status;return}
          const j=await r.json()
          if(!j.ok||!j.results||!j.results.length){out.innerHTML="Nessun risultato";return}
          out.innerHTML=j.results.map(x=>`
            <div class="card">
              <div class="title">${x.title||""}</div>
              <div class="meta">
                score: ${x.score?.toFixed?x.score.toFixed(4):x.score}
                ${x.link?`· <a href="${x.link}" target="_blank" rel="noopener">apri fonte</a>`:""}
                ${x.id?`· id: ${x.id}`:""}
              </div>
              <div class="text">${(x.text||"").replace(/</g,"&lt;")}</div>
            </div>
          `).join("")
        }
        go.addEventListener("click",run)
        q.addEventListener("keydown",e=>{if(e.key==="Enter")run()})
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), workers=1)