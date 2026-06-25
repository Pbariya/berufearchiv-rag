# ENVIRONMENT SETUP
import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env file in the current directory

def _get_secret(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        print(f"⚠  {key} not set in environment / .env file")
    return val

# IMPORTS

import json, re, csv, uuid, time, sqlite3, threading, traceback, hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import faiss

from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline as hf_pipeline
from groq import Groq

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("⚠  mistralai not installed — Mistral models disabled.")
except Exception as _mistral_err:
    # Package is installed but import failed (e.g. version conflict or broken dependency).
    # Mistral models will be disabled for this session.
    MISTRAL_AVAILABLE = False
    print(f"⚠  mistralai import error — Mistral models disabled: {_mistral_err}")

try:
    from openai import OpenAI as OpenAIClient
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    print("⚠  openai package not installed — Cerebras fallback disabled.")

try:
    from rank_bm25 import BM25Okapi
    BM25_IMPORTABLE = True
except ImportError:
    BM25_IMPORTABLE = False
    print("⚠  rank-bm25 not installed — dense-only retrieval.")

# PATHS

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

FAISS_PATH  = os.path.join(BASE_DIR, "Faiss_Metadata", "index.faiss")
META_PATH   = os.path.join(BASE_DIR, "Faiss_Metadata", "metadata.json")
SQLITE_PATH = os.path.join(BASE_DIR, "rag_evaluation_berufearchiv.sqlite")
EXPORT_DIR  = os.path.join(BASE_DIR, "exports_v6")
CKPT_DIR    = os.path.join(BASE_DIR, "ckpts")
CSV_PATH    = os.path.join(EXPORT_DIR, "rag_evaluation_berufearchiv.csv")

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

# MODEL REGISTRY

MODEL_OPTIONS: Dict[str, Dict[str, str]] = {
    "Llama-3.1-8B":  {"provider": "groq",     "id": "llama-3.1-8b-instant"},
    "Llama-3.3-70B":   {"provider": "groq", "id": "llama-3.3-70b-versatile",},
    "Qwen3-32B":       {"provider": "groq", "id": "qwen/qwen3-32b"},
    "Mistral-Small":         {"provider": "mistral",  "id": "mistral-small-latest"},
    "Mistral-Large":         {"provider": "mistral",  "id": "mistral-large-latest"},
    "Cerebras-Qwen3-32B":    {"provider": "cerebras", "id": "qwen3-32b",    "fallback_only": True},
    "Cerebras-Llama3.3-70B": {"provider": "cerebras", "id": "llama3.3-70b", "fallback_only": True},
    "Cerebras-Llama3.1-8B":  {"provider": "cerebras", "id": "llama3.1-8b",  "fallback_only": True},
}
DEFAULT_MODEL = "Llama-3.3-70B"

# FALLBACK CHAIN

FALLBACK_CHAIN: Dict[str, List[str]] = {
    "Mistral-Large":         ["Mistral-Small",        "Qwen3-32B",          "Llama-3.3-70B",
                              "Cerebras-Qwen3-32B",   "Cerebras-Llama3.3-70B", "Llama-3.1-8B",
                              "Cerebras-Llama3.1-8B"],
    "Mistral-Small":         ["Mistral-Large",         "Qwen3-32B",          "Llama-3.3-70B",
                              "Cerebras-Qwen3-32B",   "Cerebras-Llama3.3-70B", "Llama-3.1-8B",
                              "Cerebras-Llama3.1-8B"],
    "Qwen3-32B":             ["Mistral-Small",         "Mistral-Large",      "Llama-3.3-70B",
                              "Cerebras-Qwen3-32B",   "Cerebras-Llama3.3-70B", "Llama-3.1-8B",
                              "Cerebras-Llama3.1-8B"],
    "Llama-3.3-70B":         ["Mistral-Small",         "Qwen3-32B",          "Mistral-Large",
                              "Cerebras-Llama3.3-70B","Cerebras-Qwen3-32B", "Llama-3.1-8B",
                              "Cerebras-Llama3.1-8B"],
    "Llama-3.1-8B":          ["Qwen3-32B",             "Mistral-Small",      "Llama-3.3-70B",
                              "Cerebras-Qwen3-32B",   "Cerebras-Llama3.3-70B", "Cerebras-Llama3.1-8B"],
    "Cerebras-Qwen3-32B":    ["Qwen3-32B",             "Mistral-Small",      "Llama-3.3-70B",
                              "Cerebras-Llama3.3-70B","Mistral-Large",      "Llama-3.1-8B",
                              "Cerebras-Llama3.1-8B"],
    "Cerebras-Llama3.3-70B": ["Llama-3.3-70B",         "Mistral-Small",      "Qwen3-32B",
                              "Cerebras-Qwen3-32B",   "Mistral-Large",      "Llama-3.1-8B",
                              "Cerebras-Llama3.1-8B"],
    "Cerebras-Llama3.1-8B":  ["Llama-3.1-8B",          "Qwen3-32B",          "Mistral-Small",
                              "Cerebras-Qwen3-32B",   "Llama-3.3-70B",      "Mistral-Large"],
}

_rate_limited_until: Dict[str, float] = {}
_rate_limit_lock = threading.Lock()

EMBED_MODEL_ID    = "BAAI/bge-m3"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MDEBERTA_ID   = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_MINICHECK_ID  = "roberta-large"
NLI_XLMR_ID       = "joeddav/xlm-roberta-large-xnli"

# TUNING CONSTANTS

TOP_K              = 5
FETCH_K_MULTIPLIER = 4
RERANKER_THRESHOLD = 0.0
RRF_K              = 60
MAX_NEW_TOKENS     = 450
TEMPERATURE        = 0.1
TOP_P              = 0.9
TOPIC_BROAD_K      = 300
TOPIC_SIM_THRESH   = 0.45
TOPIC_MAX_DISPLAY  = 25

MINICHECK_THRESHOLD = 0.35
MDEBERTA_THRESHOLD  = 0.35
XLMR_THRESHOLD      = 0.50

ACTIVE_EVAL_BACKEND = "mdeberta"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# API CLIENTS

groq_client = Groq(api_key=_get_secret("GROQ_API_KEY"))

mistral_client = (
    Mistral(api_key=_get_secret("MISTRAL_API_KEY"))
    if MISTRAL_AVAILABLE else None
)

cerebras_key = _get_secret("CEREBRAS_API_KEY") if CEREBRAS_AVAILABLE else None
cerebras_client = (
    OpenAIClient(
        api_key=cerebras_key,
        base_url="https://api.cerebras.ai/v1",
    )
    if (CEREBRAS_AVAILABLE and cerebras_key)
    else None
)

if cerebras_client:
    print("✅ Cerebras client ready (1M tokens/day free tier)")
else:
    print("⚠  Cerebras client not configured — add CEREBRAS_API_KEY to .env")

print("✅ API clients ready")


# DATABASE

def init_db() -> None:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, created_at TEXT,
            question TEXT, answer TEXT, gen_model TEXT, top_k INTEGER,
            eval_backend TEXT,
            lat_total REAL, lat_embed REAL, lat_retrieve REAL,
            lat_generate REAL, lat_eval REAL, topic_doc_count INTEGER
        );
        CREATE TABLE IF NOT EXISTS retrieved_docs (
            run_id TEXT, rank INTEGER, score_raw REAL, score_sim REAL,
            heading TEXT, link TEXT, content TEXT, chunk_id INTEGER, chunked INTEGER
        );
        CREATE TABLE IF NOT EXISTS topic_coverage (
            run_id TEXT, source TEXT, heading TEXT, best_sim REAL
        );
        CREATE TABLE IF NOT EXISTS evaluation (
            run_id TEXT PRIMARY KEY, eval_backend TEXT,
            answer_relevance REAL, faithfulness REAL,
            context_precision REAL, context_recall REAL
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, created_at TEXT, rating INTEGER, feedback_text TEXT
            -- Blind evaluation protocol (thesis Section 5.3): the UI hides all
            -- automated NLI scores until the annotator submits a rating (thesis
            -- Section 3.4.3). Enforcement is structural — the inline-eval panel
            -- has display:none and _revealEval() is only called inside fb().
            -- No extra column is needed: the frontend gate guarantees blindness.
        );
    """)
    conn.commit(); conn.close()
    print("✅ DB ready:", SQLITE_PATH)

init_db()


# FAISS + METADATA

print("Loading FAISS …")
faiss_index: faiss.Index = faiss.read_index(FAISS_PATH)
with open(META_PATH, "r", encoding="utf-8") as _f:
    faiss_meta: List[dict] = json.load(_f)
if isinstance(faiss_meta, dict):
    for _k in ("metadata", "docs", "items", "data"):
        if _k in faiss_meta and isinstance(faiss_meta[_k], list):
            faiss_meta = faiss_meta[_k]; break
FAISS_IS_IP = (faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT)
print(f"✅ FAISS: {faiss_index.ntotal} vectors | Metadata: {len(faiss_meta)}")
if faiss_index.ntotal != len(faiss_meta):
    print("⚠  WARNING: vector count != metadata rows")


# SOURCE DOCUMENT REGISTRY

_source_chunk_counts: Dict[str, int]  = {}
_source_headings:     Dict[str, str]  = {}
_source_keywords:     Dict[str, str]  = {}

for _m in faiss_meta:
    _src = (_m.get("source") or _m.get("file_name") or "").strip()
    if not _src: continue
    _source_chunk_counts[_src] = _source_chunk_counts.get(_src, 0) + 1
    if _src not in _source_headings:
        _hdg = (_m.get("Context_Heading") or "").strip()
        if _hdg:
            _source_headings[_src] = _hdg
    _content = (_m.get("Context_Content") or _m.get("text") or "")
    _heading  = (_m.get("Context_Heading") or "")
    _source_keywords[_src] = (
        _source_keywords.get(_src, "") + " " + _content + " " + _heading
    ).lower()

print(f"✅ Source registry: {len(_source_chunk_counts)} unique source documents")


# EMBEDDER

print(f"Loading embedder on {DEVICE} …")
embed_model = SentenceTransformer(EMBED_MODEL_ID, device=DEVICE)
print(f"✅ Embedder: {EMBED_MODEL_ID} on {DEVICE}")

def embed_text(text: str) -> np.ndarray:
    return embed_model.encode(
        [text], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")

def raw_to_sim(raw: float) -> float:
    if FAISS_IS_IP:
        return float((raw + 1.0) / 2.0)
    return float(1.0 / (1.0 + abs(raw)))


# BM25

BM25_READY = False; _bm25 = None
if BM25_IMPORTABLE:
    print("Building BM25 index …")
    _corpus_tokens = [
        (m.get("text") or m.get("Context_Content") or "").lower().split()
        for m in faiss_meta
    ]
    _bm25 = BM25Okapi(_corpus_tokens)
    BM25_READY = True
    print(f"✅ BM25: {len(_corpus_tokens)} docs")


# CROSS-ENCODER RERANKER

print("Loading reranker on CPU …")
_reranker = CrossEncoder(RERANKER_MODEL_ID, device="cpu")
print(f"✅ Reranker: {RERANKER_MODEL_ID} on CPU")


# HYBRID RETRIEVAL

def _doc_key(meta: dict) -> str:
    source   = (meta.get("source") or meta.get("file_name") or "").strip()
    chunk_id = meta.get("chunk_id")
    if chunk_id:
        return f"{source}__c{chunk_id}"
    content_prefix = (meta.get("text") or meta.get("Context_Content") or "")[:200]
    content_key = hashlib.md5(content_prefix.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{source}__h{content_key}"

def retrieve(query: str, k: int = TOP_K) -> List[dict]:
    fetch_k = max(k * FETCH_K_MULTIPLIER, 20)
    q_vec   = embed_text(query)

    raw_scores, ids = faiss_index.search(q_vec, fetch_k)
    dense = [(faiss_meta[int(i)], float(s))
             for i, s in zip(ids[0], raw_scores[0]) if i >= 0]

    faiss_sim: Dict[str, float] = {}
    for meta, raw in dense:
        key = _doc_key(meta)
        sim = raw_to_sim(raw)
        if key not in faiss_sim or sim > faiss_sim[key]:
            faiss_sim[key] = sim

    if BM25_READY and _bm25:
        bm_query  = _extract_profession_keyword(query)
        bm_scores = _bm25.get_scores(bm_query.split())
        bm_top    = sorted(range(len(bm_scores)),
                           key=lambda i: bm_scores[i], reverse=True)[:fetch_k]
        sparse = [(faiss_meta[i], bm_scores[i]) for i in bm_top]
    else:
        sparse = []

    rrf, doc_map = {}, {}
    for rank, (meta, _) in enumerate(dense, 1):
        key = _doc_key(meta)
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank)
        doc_map[key] = meta
    for rank, (meta, _) in enumerate(sparse, 1):
        key = _doc_key(meta)
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank)
        doc_map[key] = meta

    ranked     = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:fetch_k]
    candidates = [doc_map[k] for k in ranked]

    reranker_scores: Dict[str, float] = {}
    if candidates:
        texts  = [(query, (m.get("text") or m.get("Context_Content") or "")[:512])
                  for m in candidates]
        scores = _reranker.predict(texts)
        for meta, score in zip(candidates, scores):
            reranker_scores[_doc_key(meta)] = float(score)

        filtered = [(s, m) for m, s in
                    [(m, reranker_scores[_doc_key(m)]) for m in candidates]
                    if s >= RERANKER_THRESHOLD]
        if not filtered:
            filtered = [(reranker_scores.get(_doc_key(m), 0.0), m)
                        for m in candidates]
        filtered.sort(key=lambda x: x[0], reverse=True)
        candidates = [m for _, m in filtered[:k]]

    docs_out = []
    for rank, meta in enumerate(candidates, 1):
        content = (meta.get("text") or meta.get("Context_Content") or "").strip()
        key     = _doc_key(meta)
        sim     = faiss_sim.get(key, raw_to_sim(rrf.get(key, 0.0)))
        docs_out.append({
            "rank":           rank,
            "score_raw":      round(rrf.get(key, 0.0), 6),
            "score_sim":      round(sim, 4),
            "reranker_score": round(reranker_scores.get(key, 0.0), 4),
            "heading":        meta.get("Context_Heading") or meta.get("source") or "",
            "link":           meta.get("Context_Links")   or meta.get("source") or "",
            "content":        content,
            "char_count":     meta.get("char_count", len(content)),
            "page_count":     meta.get("page_count", 1),
            "chunked":        bool(meta.get("chunked", False)),
            "chunk_id":       int(meta.get("chunk_id", 0)),
        })
    return docs_out

def build_context(docs: List[dict], max_chars: int = 8000) -> str:
    ctx = ""
    for d in docs:
        block = f"[Beruf: {d['heading']} | Quelle: {d['link']}]\n{d['content']}\n\n"
        if len(ctx) + len(block) > max_chars: break
        ctx += block
    return ctx


# DATASET-WIDE TOPIC COVERAGE

def _extract_profession_keyword(query: str) -> str:
    _STOP = (
        r"\b(?:aus|dem|den|der|des|die|das|im|in|vom|von|zum|zur|"
        r"seit|bis|nach|über|unter|für|bei|an|auf|ab|"
        r"Jahr|Jahren|Zeit|und|oder|sowie|mit|hat|haben)\b"
    )
    q = query.strip()
    pattern = (
        r"(?:zum\s+Beruf|über\s+den\s+Beruf|über\s+Beruf|"
        r"Berufsbezeichnung|Berufsfeld|Ausbildung\s+zum|"
        r"Ausbildung\s+zur|über|zum)\s+"
        r"([A-ZÄÖÜ][a-zA-ZäöüÄÖÜß\-]+"
        r"(?:\s+(?:und|oder)\s+[A-ZÄÖÜ][a-zA-ZäöüÄÖÜß\-]+)*"
        r"(?:\s+[A-ZÄÖÜ][a-zA-ZäöüÄÖÜß\-]+)*)"
    )
    m = re.search(pattern, q, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        trimmed = re.split(_STOP, raw, maxsplit=1)[0].strip()
        if trimmed:
            return trimmed.lower()
        return raw.lower()
    cap_words = re.findall(r"[A-ZÄÖÜ][a-zA-ZäöüÄÖÜß]{4,}", q)
    if cap_words:
        return max(cap_words, key=len).lower()
    words = re.findall(r"[A-Za-zäöüÄÖÜß]{5,}", q)
    return max(words, key=len).lower() if words else q.lower()


def count_topic_coverage(query: str, q_vec: Optional[np.ndarray] = None) -> dict:
    keyword = _extract_profession_keyword(query)

    keyword_sources: set = {
        src for src, text in _source_keywords.items() if keyword in text
    }
    keyword_count = len(keyword_sources)

    if q_vec is None: q_vec = embed_text(query)
    search_k = min(TOPIC_BROAD_K, faiss_index.ntotal)
    raw_scores, ids = faiss_index.search(q_vec, search_k)

    best_sim: Dict[str, float] = {}
    for raw, idx in zip(raw_scores[0], ids[0]):
        if idx < 0 or idx >= len(faiss_meta): continue
        sim = raw_to_sim(float(raw))
        if sim < TOPIC_SIM_THRESH: continue
        meta   = faiss_meta[int(idx)]
        source = (meta.get("source") or meta.get("file_name") or "").strip()
        if not source: continue
        if source not in best_sim or sim > best_sim[source]:
            best_sim[source] = sim

    ranked     = sorted(best_sim.items(), key=lambda x: x[1], reverse=True)
    top_sources = [
        {
            "source":      src,
            "heading":     _source_headings.get(src, src),
            "best_sim":    round(sim, 4),
            "chunk_count": _source_chunk_counts.get(src, 1),
        }
        for src, sim in ranked[:TOPIC_MAX_DISPLAY]
    ]

    print(f"  [topic] keyword='{keyword}' → {keyword_count} sources")

    return {
        "total_unique_docs":  keyword_count,
        "keyword":            keyword,
        "total_dataset_docs": len(_source_chunk_counts),
        "sim_threshold":      TOPIC_SIM_THRESH,
        "candidates_scanned": search_k,
        "top_sources":        top_sources,
    }


# NLI MODELS

MINICHECK_READY = False; _minicheck = None
try:
    from minicheck.minicheck import MiniCheck
    print("Loading MiniCheck (roberta-large) …")
    _minicheck   = MiniCheck(model_name=NLI_MINICHECK_ID, cache_dir=CKPT_DIR)
    MINICHECK_READY = True
    print("✅ MiniCheck ready")
except Exception as _e:
    print(f"⚠  MiniCheck load failed: {_e}")

print(f"Loading mDeBERTa-v3 (2mil7, 26 langs) on {DEVICE} …")
_mdeberta = hf_pipeline(
    "text-classification",
    model=NLI_MDEBERTA_ID,
    device=0 if DEVICE == "cuda" else -1,
    truncation=True,
)
print("✅ mDeBERTa (2mil7) ready")

_xlmr = None
XLMR_READY = False
def _get_xlmr():
    global _xlmr, XLMR_READY
    if _xlmr is None:
        try:
            print(f"  [xlmr] Lazy-loading {NLI_XLMR_ID} on {DEVICE} …")
            _xlmr = hf_pipeline(
                "zero-shot-classification",
                model=NLI_XLMR_ID,
                device=0 if DEVICE == "cuda" else -1,
            )
            XLMR_READY = True
            print("  [xlmr] ✅ xlm-roberta-large ready")
        except Exception as e:
            print(f"  [xlmr] Load failed: {e}")
    return _xlmr


# OCR NORMALISATION FOR NLI

def _clean_for_nli(text: str) -> str:
    if not text:
        return text
    t = re.sub(r"-\s*\n\s*", "", text)
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"[^\x20-\x7E\u00C0-\u024F\u1E00-\u1EFF]", " ", t)
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r'([a-zäöüß])O([a-zäöüß])', 'ö'.join(['\\1', '\\2']), t)
    t = re.sub(r'([a-zäöüß])U([a-zäöüß])', 'ü'.join(['\\1', '\\2']), t)
    t = re.sub(r'([a-zäöüß])A([a-zäöüß])', 'ä'.join(['\\1', '\\2']), t)
    return t.lower().strip()


# UNIFIED NLI MATRIX

def _matrix_minicheck(chunks: List[str], claims: List[str]) -> List[List[float]]:
    nc, nch  = len(claims), len(chunks)
    prob_mat = [[0.0] * nch for _ in range(nc)]
    for ci, chunk in enumerate(chunks):
        try:
            _, raw_probs, _, _ = _minicheck.score(
                docs=[chunk] * nc, claims=claims)
            for ki, p in enumerate(raw_probs):
                prob_mat[ki][ci] = float(p)
        except Exception as e:
            print(f"  [MiniCheck] chunk {ci+1}/{nch} error: {e}")
    return prob_mat

def _matrix_mdeberta(chunks: List[str], claims: List[str]) -> List[List[float]]:
    nc, nch  = len(claims), len(chunks)
    prob_mat = [[0.0] * nch for _ in range(nc)]
    for ci, chunk in enumerate(chunks):
        try:
            batch = [{"text": chunk[:3000], "text_pair": c} for c in claims]
            for ki, out in enumerate(_mdeberta(batch, top_k=None)):
                probs = {o["label"].lower(): o["score"] for o in out}
                prob_mat[ki][ci] = probs.get("entailment", 0.0)
        except Exception as e:
            print(f"  [mDeBERTa] chunk {ci+1}/{nch} error: {e}")
    all_probs = [prob_mat[ki][ci] for ki in range(nc) for ci in range(nch)]
    if all_probs:
        sp = sorted(all_probs, reverse=True)
        print(f"  [mDeBERTa] {len(all_probs)} pairs | "
              f"max={sp[0]:.3f}  mean={sum(all_probs)/len(all_probs):.3f}  "
              f"min={sp[-1]:.3f}")
    return prob_mat

def _matrix_xlmr(chunks: List[str], claims: List[str]) -> List[List[float]]:
    xlmr = _get_xlmr()
    if not xlmr:
        print("  [xlmr] Not available — falling back to mDeBERTa")
        return _matrix_mdeberta(chunks, claims)
    nc, nch  = len(claims), len(chunks)
    prob_mat = [[0.0] * nch for _ in range(nc)]
    for ci, chunk in enumerate(chunks):
        try:
            result = xlmr(chunk[:3000], candidate_labels=claims,
                          hypothesis_template="{}")
            for ki, (lbl, score) in enumerate(zip(result["labels"], result["scores"])):
                try:
                    claim_idx = claims.index(lbl)
                    prob_mat[claim_idx][ci] = float(score)
                except ValueError:
                    pass
        except Exception as e:
            print(f"  [xlmr] chunk {ci+1}/{nch} error: {e}")
    return prob_mat

def _get_matrix_fn_and_threshold(backend: str) -> Tuple[Any, float]:
    if backend == "minicheck":
        if not MINICHECK_READY:
            print("⚠  MiniCheck not available — using mDeBERTa")
            return _matrix_mdeberta, MDEBERTA_THRESHOLD
        return _matrix_minicheck, MINICHECK_THRESHOLD
    if backend == "xlmr":
        return _matrix_xlmr, XLMR_THRESHOLD
    return _matrix_mdeberta, MDEBERTA_THRESHOLD

def _get_active_matrix_fn() -> Tuple[Any, float]:
    return _get_matrix_fn_and_threshold(ACTIVE_EVAL_BACKEND)


# RAGAS METRICS

_pipeline_cp_cache: Dict[str, float] = {}

def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()

def _cp_cache_key(question: str, docs: List[dict]) -> str:
    content_hashes = "|".join(_sha(d.get("content", "")) for d in docs)
    return _sha(question + "|" + content_hashes)

def eval_answer_relevance(question: str, answer: str,
                          q_vec: Optional[np.ndarray] = None,
                          topic_info: Optional[dict] = None) -> float:
    try:
        if q_vec is None: q_vec = embed_text(question)
        a_vec = embed_text(answer[:1500])
        av    = a_vec[0]
        qv    = q_vec[0]
        denom_full = np.linalg.norm(qv) * np.linalg.norm(av)
        ar_full = float(np.dot(qv, av) / denom_full) if denom_full > 0 else 0.0

        keyword = (topic_info or {}).get("keyword", "")
        if keyword:
            kv = embed_text(keyword)[0]
            denom_kw = np.linalg.norm(kv) * np.linalg.norm(av)
            ar_kw = float(np.dot(kv, av) / denom_kw) if denom_kw > 0 else 0.0
            ar = max(ar_full, ar_kw)
            print(f"  [eval] AR full={ar_full:.3f}  AR keyword={ar_kw:.3f}  → {ar:.3f}")
        else:
            ar = ar_full

        return round(ar, 4)
    except Exception as e:
        print(f"  [eval] AR error: {e}"); return 0.0

def _ragas_nli_metrics(docs: List[dict], answer: str,
                        backend: Optional[str] = None) -> Tuple[float, float, float]:
    use_backend = backend or ACTIVE_EVAL_BACKEND
    claims = [c.lower() for c in _extract_claims(answer)]
    chunks = [_clean_for_nli(d["content"][:3000]) for d in docs]
    if not claims or not chunks: return 0.0, 0.0, 0.0

    mat_fn, T = _get_matrix_fn_and_threshold(use_backend)
    prob_mat  = mat_fn(chunks, claims)
    nc, nch   = len(claims), len(chunks)

    per_claim_max = [max(prob_mat[ki]) for ki in range(nc)]
    faithfulness  = round(float(np.mean(per_claim_max)), 4)
    supported_f   = sum(1 for v in per_claim_max if v >= T)
    print(f"  [{use_backend}] Faithfulness: np.median={faithfulness:.3f} | "
          f"binary: {supported_f}/{nc} claims >= T={T}")

    chunk_rel = [1 if max(prob_mat[ki][ci] for ki in range(nc)) >= T else 0
                 for ci in range(nch)]
    total_rel = sum(chunk_rel)
    if total_rel == 0:
        cp = 0.0
    else:
        running, weighted = 0, 0.0
        for i, rel in enumerate(chunk_rel, 1):
            if rel: running += 1; weighted += running / i
        cp = round(weighted / total_rel, 4)
    print(f"  [{use_backend}] Context Precision: {total_rel}/{nch} chunks -> {cp:.3f}")

    grounded = sum(1 for ki in range(nc) if max(prob_mat[ki]) >= T)
    cr       = round(grounded / nc, 4)
    print(f"  [{use_backend}] Context Recall: {grounded}/{nc} claims -> {cr:.3f}")
    return faithfulness, cp, cr

def evaluate(question: str, answer: str, docs: List[dict],
             q_vec: Optional[np.ndarray] = None,
             pipeline_cp: Optional[float] = None,
             topic_info: Optional[dict] = None) -> Tuple[dict, float, float]:
    t0 = time.time()
    ar             = eval_answer_relevance(question, answer, q_vec, topic_info)
    fa, cp_raw, cr = _ragas_nli_metrics(docs, answer)
    cp = pipeline_cp if pipeline_cp is not None else cp_raw
    metrics = {
        "answer_relevance":  ar,
        "faithfulness":      fa,
        "context_precision": cp,
        "context_recall":    cr,
        "eval_backend":      ACTIVE_EVAL_BACKEND,
    }
    return metrics, round(time.time() - t0, 3), cp_raw


# TEXT CLEANING

def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$",         "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def format_answer(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{1,2}(.*?)_{1,2}",   r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _extract_claims(answer: str) -> List[str]:
    # Strip any injected count sentence (starts with known patterns)
    lines = answer.split('\n')
    lines = [l for l in lines if not re.match(
        r'^(Im gesamten Berufsarchiv|The Berufsarchiv contains)', l.strip()
    )]
    answer = ' '.join(lines)

    text   = re.sub(r"\s+", " ", answer).strip()
    sents  = re.split(r"(?<=[.!?])\s+", text)

    # Filter: min length 15 chars (raised from 8), skip pure connective sentences
    _SKIP_PATTERNS = re.compile(
        r'^(zusätzlich|darüber hinaus|außerdem|jedoch|allerdings|'
        r'furthermore|additionally|however|moreover)\b',
        re.IGNORECASE
    )
    claims = [
        s.strip() for s in sents
        if len(s.strip()) >= 15 and not _SKIP_PATTERNS.match(s.strip())
    ]

    if len(claims) > 10:  # lowered from 15 → keeps only the most substantive claims
        print(f"  [claims] {len(claims)} sentences found, using first 10")
    return claims[:10]


# GENERATION

SYSTEM_PROMPT = (
    "You are a research assistant for the Berufsarchiv — a historical German archive "
    "of occupational profile documents (Berufsbilder). Documents are OCR-scanned and "
    "may contain noise.\n\n"
    "Your task: summarise what the provided CONTEXT says about the queried "
    "profession. Write in plain prose. Cover training duration, main "
    "responsibilities, required skills, and any other relevant details found "
    "in the context. Mention the source filenames (e.g. berufearchiv_1153.pdf) "
    "where appropriate.\n\n"
    "STRICT RULES:\n"
    "- Answer ONLY from the CONTEXT. Never invent or add facts.If the documents do not contain the answer, respond exactly: "
    "'Dieses Thema liegt außerhalb des Zuständigkeitsbereichs dieses Chatbots.' \n"
    "- You are given the TOP 5 most relevant retrieved chunks only. "
    "Do NOT invent details about documents not shown.\n"
    "- Be concise. Do not repeat the same information twice.\n"
    "- Answer in the SAME LANGUAGE as the question (German → German).\n"
    "- Do NOT use any markdown: no **, no ##, no bullet points, no numbered lists, no dashes as list markers.\n"
    "- Write in plain prose paragraphs only.\n"
    "- Output ONLY the final answer — no thinking, no planning, no meta-commentary.\n"
    "- Never refer to documents as 'Doc 1' or 'Document 2'. Use the actual filename."
)

def _is_rate_limit_error(exc: Exception) -> Tuple[bool, int]:
    retry_after = 60
    try:
        import groq as _groq
        if isinstance(exc, _groq.RateLimitError):
            try:
                ra = exc.response.headers.get("retry-after")
                if ra: retry_after = int(float(ra)) + 5
            except Exception:
                pass
            return True, retry_after
        if isinstance(exc, _groq.APIStatusError) and exc.status_code == 429:
            return True, retry_after
    except ImportError:
        pass
    try:
        from openai import RateLimitError as _ORL, APIStatusError as _OAS
        if isinstance(exc, _ORL):
            try:
                ra = exc.response.headers.get("retry-after")
                if ra: retry_after = int(float(ra)) + 5
            except Exception:
                pass
            return True, retry_after
        if isinstance(exc, _OAS) and exc.status_code == 429:
            return True, retry_after
    except ImportError:
        pass
    try:
        from mistralai.models import SDKError as _MSE
        if isinstance(exc, _MSE) and "429" in str(exc):
            return True, retry_after
    except ImportError:
        pass
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "quota" in msg or "too many requests" in msg:
        return True, retry_after
    if getattr(exc, "status_code", None) == 429:
        return True, retry_after
    return False, 0

def _mark_rate_limited(model_key: str, retry_after: int) -> None:
    until = time.time() + retry_after
    with _rate_limit_lock:
        _rate_limited_until[model_key] = until
    print(f"  [fallback] {model_key} rate-limited for {retry_after}s")

def _is_available(model_key: str) -> bool:
    with _rate_limit_lock:
        until = _rate_limited_until.get(model_key, 0)
    return time.time() >= until

def _call_model(model_key: str, user_prompt: str) -> str:
    cfg      = MODEL_OPTIONS[model_key]
    provider = cfg["provider"]
    model_id = cfg["id"]

    if provider == "groq":
        resp = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": user_prompt}],
            max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P,
        )
        return resp.choices[0].message.content.strip()

    elif provider == "mistral":
        if not mistral_client:
            raise RuntimeError("Mistral client not initialised.")
        resp = mistral_client.chat.complete(
            model=model_id,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": user_prompt}],
            max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P,
        )
        return resp.choices[0].message.content.strip()

    elif provider == "cerebras":
        if not cerebras_client:
            raise RuntimeError("Cerebras client not configured.")
        resp = cerebras_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": user_prompt}],
            max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P,
        )
        return resp.choices[0].message.content.strip()

    raise ValueError(f"Unknown provider '{provider}'")

def generate_answer(question: str, context: str, model_key: str,
                    docs: Optional[List[dict]] = None,
                    topic_info: Optional[dict] = None) -> Tuple[str, str]:
    n_total = (topic_info or {}).get("total_unique_docs", 0)
    prompt  = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
        f"ANSWER (plain prose only, no markdown, no thinking text):"
    )

    candidates = [model_key] + FALLBACK_CHAIN.get(model_key, [])
    tried      = []
    last_error = None

    for candidate in candidates:
        if candidate not in MODEL_OPTIONS:
            continue
        if MODEL_OPTIONS[candidate]["provider"] == "mistral" and not mistral_client:
            continue
        if MODEL_OPTIONS[candidate]["provider"] == "cerebras" and not cerebras_client:
            continue
        if not _is_available(candidate):
            with _rate_limit_lock:
                until = _rate_limited_until.get(candidate, 0)
            tried.append(f"{candidate}(cooling)")
            continue

        model_id    = MODEL_OPTIONS[candidate]["id"]
        user_prompt = prompt + (" /no_think" if "qwen" in model_id.lower() else "")

        try:
            if candidate != model_key:
                print(f"  [fallback] Trying {candidate}")
            raw = _call_model(candidate, user_prompt)

            eval_answer    = format_answer(strip_thinking(raw))
        
            display_answer = eval_answer

            if candidate != model_key:
                notice = (
                    f"\n\n[Note: This response was generated by {candidate} — "
                    f"{model_key} has reached its free-tier rate limit.]"
                )
                display_answer = display_answer + notice
                print(f"  [fallback] ✅ {candidate} succeeded")

            if n_total > 0:
                is_de = any(w in question.lower() for w in
                            ["was", "wie", "welche", "welcher", "wer", "habt",
                             "gibt", "über", "zum", "zur", "kannst", "bitte",
                             "sagen", "haben", "sind", "beruf", "ausbildung"])
                keyword = (topic_info or {}).get("keyword", "")
                if is_de:
                    count_sentence = (
                        f"Im gesamten Berufsarchiv sind {n_total} "
                        f"Quelldokument(e) zum Beruf {keyword.title()} vorhanden."
                    )
                else:
                    count_sentence = (
                        f"The Berufsarchiv contains {n_total} source "
                        f"document(s) for the occupation {keyword.title()}."
                    )
                display_answer = count_sentence + "\n\n" + display_answer

            return eval_answer, display_answer

        except Exception as exc:
            is_rl, retry_after = _is_rate_limit_error(exc)
            if is_rl:
                _mark_rate_limited(candidate, retry_after)
                tried.append(f"{candidate}(429)")
                last_error = exc
                continue
            else:
                raise

    exhausted_msg = (
        "All available models have reached their free-tier rate limits. "
        f"Please wait a few minutes. Models tried: {', '.join(tried)}."
    )
    print(f"  [fallback] ❌ All models exhausted: {tried}")
    return exhausted_msg, exhausted_msg


# END-TO-END RAG PIPELINE

_eval_results: Dict[str, dict] = {}
_eval_lock = threading.Lock()

def _run_background_eval(run_id: str, question: str, answer: str,
                          docs: List[dict], q_vec: np.ndarray,
                          model_key: str, top_k: int,
                          lats_sync: dict, topic_info: dict,
                          eval_backend_snapshot: str) -> None:
    try:
        ckey      = _cp_cache_key(question, docs)
        cache_key = f"{ckey}|{eval_backend_snapshot}"

        with _eval_lock:
            existing_cp = _pipeline_cp_cache.get(cache_key)

        t_eval = time.time()
        metrics, lat_v, fresh_cp = _evaluate_with_backend(
            eval_backend_snapshot, question, answer, docs,
            q_vec=q_vec, pipeline_cp=existing_cp, topic_info=topic_info
        )

        with _eval_lock:
            if cache_key not in _pipeline_cp_cache and fresh_cp is not None:
                _pipeline_cp_cache[cache_key] = fresh_cp

        lats = {**lats_sync, "lat_eval": round(lat_v, 3),
                "lat_total": round(lats_sync["lat_total"] + lat_v, 3)}

        _save_run(run_id, question, answer, model_key, top_k,
                  docs, metrics, lats, topic_info)

        with _eval_lock:
            _eval_results[run_id] = {
                "ready":   True,
                "metrics": metrics,
                "latency": lats,
            }
        print(f"  [eval done] run={run_id[:8]} model={model_key} "
              f"AR={metrics['answer_relevance']:.3f} F={metrics['faithfulness']:.3f}")
    except Exception as err:
        print(f"  [eval error] run={run_id[:8]}: {err}")
        traceback.print_exc()
        with _eval_lock:
            _eval_results[run_id] = {"ready": True, "metrics": None, "latency": lats_sync}

def _evaluate_with_backend(backend: str, question: str, answer: str,
                            docs: List[dict], q_vec: Optional[np.ndarray] = None,
                            pipeline_cp: Optional[float] = None,
                            topic_info: Optional[dict] = None) -> Tuple[dict, float, float]:
    t0 = time.time()
    ar = eval_answer_relevance(question, answer, q_vec, topic_info)

    claims = [c.lower() for c in _extract_claims(answer)]
    chunks = [_clean_for_nli(d["content"][:3000]) for d in docs]
    if not claims or not chunks:
        fa, cp_raw, cr = 0.0, 0.0, 0.0
    else:
        fa, cp_raw, cr = _ragas_nli_metrics(docs, answer, backend=backend)

    cp = pipeline_cp if pipeline_cp is not None else cp_raw
    metrics = {
        "answer_relevance":  ar,
        "faithfulness":      fa,
        "context_precision": cp,
        "context_recall":    cr,
        "eval_backend":      backend,
    }
    return metrics, round(time.time() - t0, 3), cp_raw

def _single_model_answer(question: str, model_key: str, top_k: int,
                          docs: List[dict], context: str,
                          q_vec: np.ndarray, topic_info: dict,
                          t0: float, lat_e: float, lat_r: float) -> dict:
    run_id = str(uuid.uuid4())

    tG = time.time()
    eval_answer, display_answer = generate_answer(question, context, model_key,
                                                  docs=docs, topic_info=topic_info)
    lat_g = round(time.time() - tG, 3)
    lat_total_sync = round(time.time() - t0, 3)

    lats_sync = {
        "lat_embed":    round(lat_e, 3),
        "lat_retrieve": round(lat_r, 3),
        "lat_generate": lat_g,
        "lat_eval":     0.0,
        "lat_total":    lat_total_sync,
    }

    eval_backend_snapshot = ACTIVE_EVAL_BACKEND

    with _eval_lock:
        _eval_results[run_id] = {"ready": False, "metrics": None, "latency": lats_sync}

    threading.Thread(
        target=_run_background_eval,
        args=(run_id, question, eval_answer, docs, q_vec,
              model_key, top_k, lats_sync, topic_info,
              eval_backend_snapshot),
        daemon=True,
    ).start()

    return {
        "run_id":         run_id,
        "answer":         display_answer,
        "eval_answer":    eval_answer,
        "docs":           docs,
        "model":          model_key,
        "eval":           None,
        "eval_pending":   True,
        "latency":        lats_sync,
        "topic_coverage": topic_info,
        "eval_backend":   ACTIVE_EVAL_BACKEND,
    }

def rag_answer(question: str, model_key: str = DEFAULT_MODEL,
               top_k: int = TOP_K) -> dict:
    t0 = time.time()

    tE = time.time()
    q_vec = embed_text(question)
    lat_e = time.time() - tE

    tR = time.time()
    docs = retrieve(question, top_k)
    lat_r = time.time() - tR

    topic_info = count_topic_coverage(question, q_vec)
    context    = build_context(docs)

    if model_key == "All Models":
        model_list = [k for k in MODEL_OPTIONS
                      if k != "All Models"
                      and not MODEL_OPTIONS[k].get("fallback_only", False)]
        results    = [None] * len(model_list)

        def _gen(idx, mk):
            try:
                results[idx] = _single_model_answer(
                    question, mk, top_k, docs, context,
                    q_vec, topic_info, t0, lat_e, lat_r
                )
            except Exception as e:
                results[idx] = {"model": mk, "answer": f"Error: {e}",
                                "run_id": str(uuid.uuid4()), "docs": docs,
                                "eval": None, "eval_pending": False,
                                "latency": {}, "topic_coverage": topic_info,
                                "eval_backend": ACTIVE_EVAL_BACKEND}

        threads = [threading.Thread(target=_gen, args=(i, mk), daemon=True)
                   for i, mk in enumerate(model_list)]
        for t in threads: t.start()
        for t in threads: t.join()

        return {
            "all_models":     True,
            "responses":      results,
            "topic_coverage": topic_info,
            "eval_backend":   ACTIVE_EVAL_BACKEND,
        }

    return _single_model_answer(
        question, model_key, top_k, docs, context,
        q_vec, topic_info, t0, lat_e, lat_r
    )


# PERSISTENCE + AUTO CSV

CSV_FIELDS = [
    "run_id", "created_at", "question", "answer", "gen_model", "eval_backend",
    "answer_relevance", "faithfulness", "context_precision", "context_recall",
    "topic_doc_count",
    "lat_embed", "lat_retrieve", "lat_generate", "lat_eval", "lat_total",
]

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _save_run(run_id, question, answer, model_key, top_k,
              docs, metrics, lats, topic_info) -> None:
    topic_count  = topic_info.get("total_unique_docs", 0)
    eval_backend = metrics.get("eval_backend", ACTIVE_EVAL_BACKEND)
    ts           = _now_iso()
    conn = sqlite3.connect(SQLITE_PATH)
    cur  = conn.cursor()
    try:
        cur.execute(
            "INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, ts, question, answer, model_key, top_k, eval_backend,
             lats["lat_total"], lats["lat_embed"], lats["lat_retrieve"],
             lats["lat_generate"], lats["lat_eval"], topic_count)
        )
        for d in docs:
            cur.execute("INSERT INTO retrieved_docs VALUES (?,?,?,?,?,?,?,?,?)",
                        (run_id, d["rank"], d["score_raw"], d["score_sim"],
                         d["heading"], d["link"], d["content"],
                         d["chunk_id"], int(d["chunked"])))
        for s in topic_info.get("top_sources", []):
            cur.execute("INSERT INTO topic_coverage VALUES (?,?,?,?)",
                        (run_id, s["source"], s["heading"], s["best_sim"]))
        cur.execute(
            "INSERT OR IGNORE INTO evaluation VALUES (?,?,?,?,?,?)",
            (run_id, eval_backend,
             metrics["answer_relevance"], metrics["faithfulness"],
             metrics["context_precision"], metrics["context_recall"])
        )
        conn.commit()
    finally:
        conn.close()

    threading.Thread(target=_append_csv, args=(run_id, ts, question, answer,
                     model_key, eval_backend, metrics, lats, topic_count),
                     daemon=True).start()

def _append_csv(run_id, ts, question, answer, model_key, eval_backend,
                metrics, lats, topic_count):
    file_exists = os.path.exists(CSV_PATH)
    try:
        with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if not file_exists:
                w.writeheader()
            w.writerow({
                "run_id": run_id, "created_at": ts,
                "question": question, "answer": answer,
                "gen_model": model_key, "eval_backend": eval_backend,
                "answer_relevance":  metrics["answer_relevance"],
                "faithfulness":      metrics["faithfulness"],
                "context_precision": metrics["context_precision"],
                "context_recall":    metrics["context_recall"],
                "topic_doc_count":   topic_count,
                "lat_embed":    lats["lat_embed"],
                "lat_retrieve": lats["lat_retrieve"],
                "lat_generate": lats["lat_generate"],
                "lat_eval":     lats["lat_eval"],
                "lat_total":    lats["lat_total"],
            })
    except Exception as e:
        print(f"  [csv] append error: {e}")

# save_feedback — persist the annotator's binary rating and optional free-text note.
# Blind evaluation protocol scores are hidden until user provide feedback
def save_feedback(run_id: str, rating: int, feedback_text: str = "") -> None:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute(
        "INSERT INTO feedback(run_id,created_at,rating,feedback_text) VALUES (?,?,?,?)",
        (run_id, _now_iso(), rating, feedback_text))
    conn.commit(); conn.close()


# SECTION 22 — FLASK ROUTES
app = Flask(__name__)

@app.route("/")
def index():
    model_opts = ["All Models"] + [
        k for k in MODEL_OPTIONS
        if not MODEL_OPTIONS[k].get("fallback_only", False)
    ]
    return render_template(
        "rag_evaluation_berufearchiv.html",
        model_options=model_opts,
        default_model=DEFAULT_MODEL,
        top_k=TOP_K,
        active_backend=ACTIVE_EVAL_BACKEND,
    )

@app.route("/health")
def health():
    if faiss_index is None or len(faiss_meta) == 0:
        return jsonify({
            "status": "degraded",
            "reason": "FAISS index not loaded",
        }), 503
    return jsonify({
        "status":        "ok",
        "faiss_vectors": faiss_index.ntotal,
        "metadata_docs": len(faiss_meta),
        "eval_backend":  ACTIVE_EVAL_BACKEND,
        "models":        list(MODEL_OPTIONS.keys()),
    })

@app.route("/ask", methods=["POST"])
def ask():
    data      = request.get_json(force=True)
    question  = (data.get("question") or "").strip()
    model_key = data.get("model", DEFAULT_MODEL)
    top_k     = int(data.get("top_k", TOP_K))
    if not question:
        return jsonify({"error": "Empty question"}), 400
    if model_key != "All Models" and model_key not in MODEL_OPTIONS:
        return jsonify({"error": f"Unknown model '{model_key}'"}), 400
    try:
        return jsonify(rag_answer(question, model_key, top_k))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/set_backend", methods=["POST"])
def set_backend():
    global ACTIVE_EVAL_BACKEND
    data    = request.get_json(force=True)
    backend = data.get("backend", "").strip().lower()
    if backend not in ("minicheck", "mdeberta", "xlmr"):
        return jsonify({"error": f"Unknown backend '{backend}'"}), 400
    if backend == "minicheck" and not MINICHECK_READY:
        return jsonify({"error": "MiniCheck not available. Use mdeberta or xlmr."}), 400
    if backend == "xlmr":
        _get_xlmr()
        if not XLMR_READY:
            return jsonify({"error": "xlm-roberta failed to load. Use mdeberta."}), 400
    ACTIVE_EVAL_BACKEND = backend
    print(f"  [backend] switched to {backend}")
    return jsonify({"backend": backend, "ok": True})

@app.route("/eval_result/<run_id>")
def eval_result(run_id):
    with _eval_lock:
        result = _eval_results.get(run_id, {"ready": False})
    return jsonify(result)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True)
    try:
        save_feedback(data.get("run_id", ""),
                      int(data.get("rating", 0)),
                      str(data.get("feedback_text", "")))
        return jsonify({"status": "saved"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/models")
def list_models():
    return jsonify({k: {"provider": v["provider"], "model_id": v["id"]}
                    for k, v in MODEL_OPTIONS.items()})

@app.route("/rate_limit_status")
def rate_limit_status():
    now = time.time()
    status = {}
    with _rate_limit_lock:
        for model_key in MODEL_OPTIONS:
            until = _rate_limited_until.get(model_key, 0)
            if until > now:
                status[model_key] = {
                    "available": False,
                    "rate_limited": True,
                    "cooldown_remaining_seconds": round(until - now),
                    "available_at": datetime.fromtimestamp(until).strftime("%H:%M:%S"),
                }
            else:
                status[model_key] = {
                    "available": True,
                    "rate_limited": False,
                    "cooldown_remaining_seconds": 0,
                }
    return jsonify({
        "timestamp":       datetime.now().strftime("%H:%M:%S"),
        "models":          status,
        "fallback_chains": FALLBACK_CHAIN,
    })


if __name__ == "__main__":
    USE_NGROK = os.environ.get("USE_NGROK", "false").lower() == "true"

    if USE_NGROK:
        from pyngrok import ngrok
        ngrok_token = _get_secret("NGROK_TOKEN")
        if ngrok_token:
            ngrok.set_auth_token(ngrok_token)
            ngrok.kill()
            public_url = ngrok.connect(5000)
            print(f"\n{'='*60}")
            print(f"Public URL : {public_url}")
        else:
            print("NGROK_TOKEN not set — running local only.")
            USE_NGROK = False

    print(f"\n{'='*60}")
    if not USE_NGROK:
        print(f"URL        : http://localhost:5000")
    print(f"Backend    : {ACTIVE_EVAL_BACKEND}")
    print(f"CSV        : {CSV_PATH}")
    print(f"SQLite     : {SQLITE_PATH}")
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=5000, use_reloader=False)