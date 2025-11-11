import os
import time
import re
import hashlib
import threading
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import discoveryengine_v1 as de
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import ResourceExhausted, NotFound
from google.protobuf.json_format import MessageToDict
import uvicorn

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("DISCOVERY_LOCATION", "eu")
ENGINE_ID = os.getenv("DISCOVERY_ENGINE_ID")
COLLECTION = os.getenv("DISCOVERY_COLLECTION", "default_collection")
DATA_STORE_ID = os.getenv("DATA_STORE_ID", "")
BRANCH_ID = os.getenv("BRANCH_ID", "default_branch")
API_ENDPOINT = f"{LOCATION}-discoveryengine.googleapis.com"
if not PROJECT_ID or not ENGINE_ID:
    raise RuntimeError("Missing env: PROJECT_ID, DISCOVERY_ENGINE_ID")

search_client = de.SearchServiceClient(client_options=ClientOptions(api_endpoint=API_ENDPOINT))
doc_client = de.DocumentServiceClient(client_options=ClientOptions(api_endpoint=API_ENDPOINT))
SERVING = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION}/engines/{ENGINE_ID}/servingConfigs/default_search"
DOC_PARENT = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION}/dataStores/{DATA_STORE_ID}/branches/{BRANCH_ID}" if DATA_STORE_ID else ""

_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = int(os.getenv("CACHE_TTL_SEC", "600"))
_rate_lock = threading.Lock()
_last_calls_ts = []
_RATE_WINDOW = float(os.getenv("RATE_WINDOW_SEC", "1"))
_RATE_MAX = int(os.getenv("RATE_MAX_CALLS", "3"))

class SearchIn(BaseModel):
    query: str
    page_size: int = 5
    mode: str = "DOCUMENTS"

app = FastAPI(title="vertex-mini", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _packs(results) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, r in enumerate(results, 1):
        d = r.document
        data = {}
        try:
            if getattr(d, "derived_struct_data", None):
                data = MessageToDict(getattr(d, "derived_struct_data"))
        except Exception:
            pass
        try:
            if not data and getattr(d, "struct_data", None):
                data = MessageToDict(getattr(d, "struct_data"))
        except Exception:
            pass
        title = data.get("title") or data.get("file_name") or getattr(d, "id", str(idx))
        link = data.get("link") or data.get("uri") or data.get("url") or getattr(d, "id", "")
        text = ""
        if getattr(r, "snippet_info", None) and getattr(r.snippet_info, "snippets", None):
            sn = r.snippet_info.snippets
            if sn:
                text = sn[0].snippet or ""
        if not text:
            sn2 = data.get("snippets")
            if isinstance(sn2, list) and sn2:
                x = sn2[0]
                if isinstance(x, str):
                    text = x
                elif isinstance(x, dict) and "snippet" in x:
                    text = x["snippet"]
        if not text and getattr(d, "content", None):
            try:
                cont = MessageToDict(d.content)
                if isinstance(cont, dict):
                    raw = cont.get("rawBytes") or cont.get("rawText") or cont.get("content") or ""
                    if isinstance(raw, str) and raw:
                        text = raw[:800]
            except Exception:
                pass
        score = float(getattr(r, "matching_score", 0.0) or 0.0)
        if text:
            out.append({"title": title, "link": link, "text": text, "score": score, "id": getattr(d, "id", "")})
    return out

def _cache_key(q: str, page_size: int, mode: str) -> str:
    s = f"{q}|{page_size}|{mode}"
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

def _rate_allow() -> bool:
    now = time.time()
    with _rate_lock:
        while _last_calls_ts and now - _last_calls_ts[0] > _RATE_WINDOW:
            _last_calls_ts.pop(0)
        if len(_last_calls_ts) >= _RATE_MAX:
            return False
        _last_calls_ts.append(now)
        return True

def _search(query: str, page_size: int, mode: str) -> List[Dict[str, Any]]:
    if not _rate_allow():
        raise HTTPException(status_code=429, detail="Rate limited, riprova tra un attimo")
    qn = re.sub(r"\s+", " ", (query or "").strip())[:500]
    pmode = (mode or "DOCUMENTS").upper()
    ps = min(max(int(page_size or 5), 1), 5)
    ck = _cache_key(qn, ps, pmode)
    cached = _get_cache(ck)
    if cached is not None:
        return cached
    cs = de.SearchRequest.ContentSearchSpec(
        search_result_mode=getattr(de.SearchRequest.ContentSearchSpec.SearchResultMode, pmode),
        snippet_spec=de.SearchRequest.ContentSearchSpec.SnippetSpec(return_snippet=True, max_snippet_count=3),
    )
    req = de.SearchRequest(
        serving_config=SERVING,
        query=qn,
        page_size=ps,
        content_search_spec=cs,
        query_expansion_spec=de.SearchRequest.QueryExpansionSpec(condition=de.SearchRequest.QueryExpansionSpec.Condition.AUTO),
        spell_correction_spec=de.SearchRequest.SpellCorrectionSpec(mode=de.SearchRequest.SpellCorrectionSpec.Mode.AUTO),
    )
    try:
        primary = list(search_client.search(request=req))
    except ResourceExhausted:
        prev = _get_cache(ck)
        if prev is not None:
            return prev
        raise HTTPException(status_code=429, detail="Quota Search esaurita. Riduci richieste o riprova più tardi.")
    packs = _packs(primary)
    if not packs and pmode == "CHUNKS":
        cs2 = de.SearchRequest.ContentSearchSpec(
            search_result_mode=de.SearchRequest.ContentSearchSpec.SearchResultMode.DOCUMENTS,
            snippet_spec=de.SearchRequest.ContentSearchSpec.SnippetSpec(return_snippet=True, max_snippet_count=3),
        )
        req2 = de.SearchRequest(
            serving_config=SERVING,
            query=qn,
            page_size=ps,
            content_search_spec=cs2,
            query_expansion_spec=de.SearchRequest.QueryExpansionSpec(condition=de.SearchRequest.QueryExpansionSpec.Condition.AUTO),
            spell_correction_spec=de.SearchRequest.SpellCorrectionSpec(mode=de.SearchRequest.SpellCorrectionSpec.Mode.AUTO),
        )
        try:
            secondary = list(search_client.search(request=req2))
        except ResourceExhausted:
            prev = _get_cache(ck)
            if prev is not None:
                return prev
            raise HTTPException(status_code=429, detail="Quota Search esaurita. Riduci richieste o riprova più tardi.")
        packs = _packs(secondary)
    if packs:
        _set_cache(ck, packs)
    return packs

def _get_document_text(doc_id: str) -> Dict[str, Any]:
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

@app.post("/search")
def search(body: SearchIn):
    try:
        packs = _search(body.query, body.page_size, body.mode)
        if not packs and DATA_STORE_ID:
            ids = []
            try:
                cs = de.SearchRequest.ContentSearchSpec(search_result_mode=de.SearchRequest.ContentSearchSpec.SearchResultMode.DOCUMENTS)
                req = de.SearchRequest(serving_config=SERVING, query=re.sub(r"\s+"," ",body.query)[:200], page_size=min(max(body.page_size,1),5), content_search_spec=cs)
                for r in search_client.search(request=req):
                    ids.append(getattr(r.document, "id", ""))
            except Exception:
                pass
            out = []
            for i in ids[:3]:
                if i:
                    try:
                        out.append(_get_document_text(i))
                    except Exception:
                        continue
            if out:
                return {"ok": True, "results": out}
        return {"ok": True, "results": packs}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/search")
def search_get(q: str, page_size: int = 5, mode: str = "DOCUMENTS"):
    try:
        packs = _search(q, page_size, mode)
        if not packs and DATA_STORE_ID:
            ids = []
            try:
                cs = de.SearchRequest.ContentSearchSpec(search_result_mode=de.SearchRequest.ContentSearchSpec.SearchResultMode.DOCUMENTS)
                req = de.SearchRequest(serving_config=SERVING, query=re.sub(r"\s+"," ",q)[:200], page_size=min(max(page_size,1),5), content_search_spec=cs)
                for r in search_client.search(request=req):
                    ids.append(getattr(r.document, "id", ""))
            except Exception:
                pass
            out = []
            for i in ids[:3]:
                if i:
                    try:
                        out.append(_get_document_text(i))
                    except Exception:
                        continue
            if out:
                return {"ok": True, "results": out}
        return {"ok": True, "results": packs}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/debug/raw")
def debug_raw(q: str, mode: str = "DOCUMENTS", page_size: int = 5):
    cs = de.SearchRequest.ContentSearchSpec(
        search_result_mode=getattr(de.SearchRequest.ContentSearchSpec.SearchResultMode, mode.upper()),
        snippet_spec=de.SearchRequest.ContentSearchSpec.SnippetSpec(return_snippet=True, max_snippet_count=3),
    )
    req = de.SearchRequest(
        serving_config=SERVING,
        query=q.strip(),
        page_size=min(max(page_size,1),5),
        content_search_spec=cs,
        query_expansion_spec=de.SearchRequest.QueryExpansionSpec(condition=de.SearchRequest.QueryExpansionSpec.Condition.AUTO),
        spell_correction_spec=de.SearchRequest.SpellCorrectionSpec(mode=de.SearchRequest.SpellCorrectionSpec.Mode.AUTO),
    )
    rows = []
    for r in search_client.search(request=req):
        doc = r.document
        derived = {}
        structd = {}
        try:
            if getattr(doc, "derived_struct_data", None):
                derived = MessageToDict(getattr(doc, "derived_struct_data"))
        except Exception:
            pass
        try:
            if getattr(doc, "struct_data", None):
                structd = MessageToDict(getattr(doc, "struct_data"))
        except Exception:
            pass
        ci = getattr(r, "chunk_info", None)
        ci_dict = {}
        if ci:
            try:
                ci_dict = MessageToDict(ci)
            except Exception:
                ci_dict = {}
        rows.append({
            "id": getattr(doc, "id", ""),
            "score": float(getattr(r, "matching_score", 0.0) or 0.0),
            "snippet_info_snippets": [s.snippet for s in getattr(r, "snippet_info", {}).snippets] if getattr(r, "snippet_info", None) and r.snippet_info.snippets else [],
            "derived_keys": list(derived.keys()) if isinstance(derived, dict) else [],
            "derived_snippets": derived.get("snippets", []) if isinstance(derived, dict) else [],
            "struct_keys": list(structd.keys()) if isinstance(structd, dict) else [],
            "has_doc_content": bool(getattr(doc, "content", None)),
            "doc_content_preview": (MessageToDict(getattr(doc, "content", {})).get("rawText") if getattr(doc, "content", None) else "")[:200] if getattr(doc, "content", None) else "",
            "chunk_info_has_content": bool(ci_dict.get("content") or (ci_dict.get("chunk") or {}).get("content")),
            "chunk_info_preview": (ci_dict.get("content") or (ci_dict.get("chunk") or {}).get("content") or "")[:200],
        })
    return {"ok": True, "items": rows}

@app.get("/debug/getdoc")
def debug_getdoc(id: str):
    if not DATA_STORE_ID:
        raise HTTPException(status_code=400, detail="DATA_STORE_ID mancante")
    return _get_document_text(id)

@app.get("/debug/serving")
def debug_serving():
    return {"serving": SERVING, "project": PROJECT_ID, "location": LOCATION, "engine": ENGINE_ID, "collection": COLLECTION, "data_store": DATA_STORE_ID, "branch": BRANCH_ID}

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
        <input id="q" placeholder="Cerca..."/>
        <select id="mode">
          <option value="DOCUMENTS" selected>DOCUMENTS</option>
          <option value="CHUNKS">CHUNKS</option>
        </select>
        <select id="size">
          <option selected>5</option>
          <option>8</option>
          <option>12</option>
          <option>20</option>
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
              <div class="meta">score: ${x.score?.toFixed?x.score.toFixed(4):x.score} ${x.link?`· <a href="${x.link}" target="_blank">apri fonte</a>`:""} ${x.id?`· id: ${x.id}`:""}</div>
              <div class="text">${x.text||""}</div>
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
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), workers=1)