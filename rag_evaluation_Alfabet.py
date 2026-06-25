# Core imports for Flask backend, FAISS retrieval, AWS Bedrock LLM integration, and LangChain-based RAG pipeline
from flask import Flask, render_template, request, jsonify, session
import os, sys, traceback, hashlib, random, time, csv, re, json
from datetime import datetime
from typing import Any, Dict, List, Optional
import boto3, threading, sqlite3
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# BM25 sparse retrieval — rank-bm25 package.
# Falls back gracefully to dense-only retrieval if not installed.
try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    print("[bm25] rank-bm25 not installed — dense-only retrieval active.")
    print("pip install rank-bm25")

# Cross-encoder reranker — sentence-transformers package.
# Reranks the top RRF candidates before passing chunks to the language model.
try:
    from sentence_transformers import CrossEncoder
    _RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _reranker = CrossEncoder(_RERANKER_MODEL_NAME)
    _RERANKER_AVAILABLE = True
    print(f"[reranker] Loaded: {_RERANKER_MODEL_NAME}")
except ImportError:
    _reranker = None
    _RERANKER_AVAILABLE = False
    print("[reranker] sentence-transformers not installed — reranking disabled.")
except Exception as _re:
    _reranker = None
    _RERANKER_AVAILABLE = False
    print(f"[reranker] Load failed: {_re}")


# RETRIEVAL AND EVALUATION PARAMETERS


RERANKER_SCORE_THRESHOLD = 0.0  # drop chunks the cross-encoder scores negatively
MINICHECK_THRESHOLD = 0.35     # binary support threshold for roberta-large
K_FINAL = 5                    # max chunks sent to LLM after filtering


module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww  # type: ignore

app = Flask(__name__)
app.secret_key = "supersecretkey"

feedback_lock          = threading.Lock()
_pending_feedback      = {}
_pending_feedback_lock = threading.Lock()

# BEDROCK CLIENT

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)

# FAISS INDEX LOADING

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# List of (relative_path, index_name) pairs. Vectors created from the knowledgebase
CANDIDATE_INDEXES = [
    ("Faiss/Alfabet_11_10_1_JSON_v8", "Alfabet_11_10_1_JSON_v8"),
]

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=boto3_bedrock,
    model_kwargs={"dimensions": 1024, "normalize": True},
)

vectorstore = None
last_error  = None
for _rel_dir, _index_name in CANDIDATE_INDEXES:
    _folder = os.path.join(BASE_DIR, _rel_dir)
    try:
        vectorstore = FAISS.load_local(
            folder_path=_folder,
            embeddings=embeddings,
            index_name=_index_name,
            allow_dangerous_deserialization=True,
        )
        print(f"[index] Loaded: {_folder}  ({_index_name})")
        # Diagnostic probe: verify metadata schema of the loaded index.
        try:
            _s = vectorstore.similarity_search("application", k=1)
            if _s:
                _m = _s[0].metadata or {}
                print(f"[index] page_content[:80]  : {_s[0].page_content[:80]!r}")
                print(f"[index] context_links      : {_m.get('context_links','(empty)')!r}")
                print(f"[index] context_heading    : {_m.get('context_heading','(empty)')!r}")
                if not _m.get("context_links"):
                    print("[index] WARNING: context_links empty — rebuild index.")
                else:
                    print("[index] Metadata schema OK.")
        except Exception as _e:
            print(f"[index] Diagnostic failed: {_e}")
        break
    except Exception as e:
        last_error = e
        print(f"[index] Could not load {_folder}: {e}")

if vectorstore is None:
    raise RuntimeError(
        "No FAISS index could be loaded.\n"
        "Run build_faiss_index_alfabet.py to build the index.\n"
        f"Last error: {last_error}"
    )


# =============================================================================
# BM25 SPARSE INDEX
# =============================================================================

_bm25_index   = None
_bm25_docs    = []
_bm25_doc_ids = []

# Constructing a BM25 index over the FAISS docstore corpus. If rank-bm25 is not installed, this function returns without action and the pipeline falls back to dense-only retrieval
def _build_bm25_index():
    global _bm25_index, _bm25_docs, _bm25_doc_ids
    if not _BM25_AVAILABLE:
        return
    try:
        print("[bm25] Building BM25 index from FAISS docstore ...")
        raw            = vectorstore.docstore._dict
        corpus_tokens  = []
        for doc_id, doc in raw.items():
            text = (doc.page_content or "").strip()
            if text:
                corpus_tokens.append(text.lower().split())
                _bm25_docs.append(doc)
                _bm25_doc_ids.append(doc_id)
        _bm25_index = BM25Okapi(corpus_tokens)
        print(f"[bm25] Index ready — {len(_bm25_docs)} documents.")
    except Exception as e:
        print(f"[bm25] Index build failed: {e}")
        _bm25_index = None


_build_bm25_index()

# HYBRID RETRIEVAL - Retrieves and rank documents using a hybrid dense-sparse pipeline.

def _rrf_hybrid_search(query: str, k: int = 8, rrf_k: int = 60) -> list:
    fetch_k = max(k * 5, 30)

    if not _BM25_AVAILABLE or _bm25_index is None:
        candidates = vectorstore.similarity_search(query, k=fetch_k)
    else:
        dense_docs  = vectorstore.similarity_search(query, k=fetch_k)
        tokens      = query.lower().split()
        bm25_scores = _bm25_index.get_scores(tokens)
        bm25_top    = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:fetch_k]
        bm25_docs = [_bm25_docs[i] for i in bm25_top]

        # RRF score accumulation: score(doc) = sum(1 / (rrf_k + rank))
        scores:  Dict[str, float]    = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            key          = doc.page_content[:200]
            scores[key]  = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs, start=1):
            key          = doc.page_content[:200]
            scores[key]  = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_map[key] = doc

        ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:fetch_k]
        candidates  = [doc_map[key] for key in ranked_keys]

    # Cross-encoder reranking
    if _RERANKER_AVAILABLE and _reranker is not None and candidates:
        try:
            pairs     = [(query, doc.page_content[:512]) for doc in candidates]
            re_scores = _reranker.predict(pairs)
            ranked    = sorted(zip(re_scores, candidates),
                               key=lambda x: x[0], reverse=True)
            filtered  = [(s, d) for s, d in ranked
                         if s >= RERANKER_SCORE_THRESHOLD]
            if not filtered:
                # All scores below threshold — retain top-k unfiltered.
                print(f"  [reranker] All scores below threshold "
                      f"{RERANKER_SCORE_THRESHOLD} — returning top-{k} unfiltered.")
                filtered = ranked
            result  = [d for _, d in filtered[:K_FINAL]]
            dropped = len(candidates) - len(result)
            if dropped:
                print(f"  [reranker] Kept {len(result)}/{len(candidates)} chunks "
                      f"({dropped} below threshold {RERANKER_SCORE_THRESHOLD})")
            return result
        except Exception as e:
            print(f"  [reranker] Reranking failed: {e} — using RRF order.")

    return candidates[:K_FINAL]

# LANGUAGE MODELS

CLAUDE_MODEL_ID     = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
CLAUDE_MODEL_CONFIG = {"max_tokens": 700, "temperature": 0.1}

llm_claude = ChatBedrock(
    model_id=CLAUDE_MODEL_ID,
    client=boto3_bedrock,
    model_kwargs=CLAUDE_MODEL_CONFIG,
)

NOVA_LITE_MODEL_ID = "amazon.nova-lite-v1:0"
NOVA_PRO_MODEL_ID  = "amazon.nova-pro-v1:0"

# NOVA SYSTEM PROMPT

_NOVA_SYSTEM_PROMPT = (
    "You are a helpful assistant for the Alfabet Enterprise Architecture "
    "Management tool. "
    "You will receive information from the Alfabet help documentation. "
    "Answer solely based on the text inside those documents.\n\n"
    "Instructions:\n"
    "- If the documents do not contain the answer, respond exactly: "
    "'This topic is outside the purview of this chatbot'\n"
    "- Do NOT add steps, UI elements, field names, or attributes unless they "
    "appear verbatim in the passages.\n"
    "- If the passages only mention a single action, do NOT expand it into a "
    "multi-step form workflow.\n"
    "- Answer concisely using short paragraphs or bullet points.\n"
    "- Do NOT include any URLs, links, or a Related Documentation section. "
    "The system appends verified links automatically.\n"
    "- If the passages contain no relevant information for a product question, "
    "respond exactly: 'This topic is outside the purview of this chatbot'\n"
    "- For personal or off-topic questions: briefly say you are an AI assistant "
    "focused on Alfabet help.\n"
    "- For greetings: respond with a greeting only.\n"
    "- Respond in English only.\n"
    "- Do not add follow-up questions.\n"
    "- Do not start your answer with 'Based on the context' or 'According to'."
)

# NOVA CONVERSE API

def _nova_call(model_id: str, prompt: str):
    start = time.time()
    resp  = bedrock_runtime.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system=[{"text": _NOVA_SYSTEM_PROMPT}],
        inferenceConfig={"maxTokens": 700, "temperature": 0.1},
    )
    return resp["output"]["message"]["content"][0]["text"], time.time() - start


def ask_nova_lite(prompt: str):
    return _nova_call(NOVA_LITE_MODEL_ID, prompt)


def ask_nova_pro(prompt: str):
    return _nova_call(NOVA_PRO_MODEL_ID, prompt)


# NOVA CONVERSATION HISTORY

_nova_hist: Dict[str, list] = {}


def _nk(sid: str, model: str) -> str:
    #Return a session-scoped key for the Nova history store.
    return f"{sid}__{model}"


def get_nova_history(sid: str, model: str) -> list:
    return _nova_hist.setdefault(_nk(sid, model), [])


def add_nova_history(sid: str, model: str, role: str, content: str):
    get_nova_history(sid, model).append({"role": role, "content": content})


def clear_conversation_history(sid: str = None):
    # Clear Nova conversation history for a session, or globally if sid is None.
    if sid:
        for k in [k for k in _nova_hist if k.startswith(sid)]:
            del _nova_hist[k]
    else:
        _nova_hist.clear()

# LINK APPENDER
#Append deduplicated source links to a generated answer. Only links with '.html' URLs are included, as these correspond to Alfabet documentation pages.
def append_links(answer: str, docs: List[Document], max_links: int = 4) -> str:
    if not answer:
        answer = ""
    answer = re.sub(r"Related Documentation.*$", "", answer,
                    flags=re.DOTALL | re.IGNORECASE).strip()
    seen_urls     = set()
    seen_headings = set()
    links         = []

    for doc in (docs or []):
        meta    = getattr(doc, "metadata", {}) or {}
        url     = (meta.get("context_links")   or "").strip()
        heading = (meta.get("context_heading") or "").strip()
        if not url or not url.endswith(".html"):
            continue
        if not heading:
            heading = url
        heading_norm = re.sub(r"\s+", " ", heading).lower().strip()
        if url in seen_urls or heading_norm in seen_headings:
            continue
        seen_urls.add(url)
        seen_headings.add(heading_norm)
        links.append((url, heading))
        if len(links) >= max_links:
            break

    if links:
        answer += "\n\nRelated Documentation:"
        for url, heading in links:
            answer += f'\n<a href="{url}" target="_blank">{heading}</a>'
    return answer

# NOVA CONTEXT BUILDER - Format retrieved documents as a numbered context block for Nova prompts.


def build_nova_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta    = getattr(doc, "metadata", {}) or {}
        content = (doc.page_content or "").strip()
        heading = (meta.get("context_heading") or "").strip()
        link    = (meta.get("context_links")   or "").strip()
        chunk   = f"[Context {i}]"
        if heading:
            chunk += f"\nHeading : {heading}"
        if link:
            chunk += f"\nLink    : {link}"
        chunk += f"\nContent : {content}"
        parts.append(chunk)
    return "\n\n".join(parts)

# EVIDENCE BUNDLE - Construct a fingerprinted evidence record for a retrieval event. The retrieval_id is a SHA-256 hash of the question and all retrieved chunk fingerprints.

def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()


def build_evidence_bundle(question: str, docs: List[Document]) -> Dict[str, Any]:
    fps = []
    for d in (docs or []):
        meta = getattr(d, "metadata", None) or {}
        fps.append({
            "content_sha256":  _sha(d.page_content or ""),
            "metadata_sha256": _sha(json.dumps(meta, sort_keys=True,
                                               ensure_ascii=False)),
            "context_heading": meta.get("context_heading"),
            "context_links":   meta.get("context_links"),
        })
    rid = _sha(question + "|" + "|".join(f["content_sha256"] for f in fps))
    return {"retrieval_id": rid, "n_docs": len(fps), "doc_fingerprints": fps}

# CLAUDE LCEL HISTORY-AWARE RAG CHAIN

_claude_hist_store: Dict[str, ChatMessageHistory] = {}


def _get_claude_hist(sid: str) -> ChatMessageHistory:
    if sid not in _claude_hist_store:
        _claude_hist_store[sid] = ChatMessageHistory()
    return _claude_hist_store[sid]


def clear_claude_history(sid: Optional[str] = None):
    """Clear Claude conversation history for a session, or globally."""
    if sid is None:
        _claude_hist_store.clear()
    else:
        _claude_hist_store.pop(sid, None)

# Constructing a history-aware LCEL retrieval chain for Claude.
def build_claude_chain(llm, vs, k: int = 8):
    
    from langchain.chains import create_retrieval_chain, create_history_aware_retriever
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import PromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun

    class HybridRetriever(BaseRetriever):
        k: int = K_FINAL

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            return _rrf_hybrid_search(query, k=self.k)

    retriever = HybridRetriever(k=K_FINAL)

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and the user's latest message, "
         "rewrite the message as a self-contained search query for a "
         "documentation retrieval system. "
         "Return ONLY the rewritten query, nothing else."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    hist_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=rewrite_prompt
    )

    doc_prompt = PromptTemplate.from_template(
        "[Context]\nHeading: {context_heading}\nContent: {page_content}\n"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant for the Alfabet Enterprise Architecture "
         "Management tool.\n\n"
         "You will receive information from the Alfabet help documentation. "
         "Answer solely based on the text inside those documents."
         "{context}\n\n"
         "Instructions:\n"
         "- If the documents do not contain the answer, respond exactly: "
         "'This topic is outside the purview of this chatbot'\n"
         "- Do NOT add steps, UI elements, field names, or attributes unless "
         "they appear verbatim in the passages.\n"
         "- If the passages only mention a single action, do NOT expand it into "
         "a multi-step workflow.\n"
         "- Answer concisely using short paragraphs or bullet points.\n"
         "- Separate paragraphs with a blank line.\n"
         "- Do NOT include any URLs, links, or a Related Documentation section. "
         "The system appends verified links automatically.\n"
         "- If the passages contain no relevant information for a product "
         "question, respond exactly: "
         "'This topic is outside the purview of this chatbot'\n"
         "- For personal or off-topic questions: briefly say you are an AI "
         "assistant focused on Alfabet help.\n"
         "- For greetings: respond with a greeting only.\n"
         "- Respond in English only.\n"
         "- Do not add follow-up questions.\n"
         "- Do not start your answer with 'Based on the context' or "
         "'According to'.\n"),
        MessagesPlaceholder("chat_history"),
        ("human", "Question: {input}\nAnswer:"),
    ])

    qa_chain = create_stuff_documents_chain(
        llm, qa_prompt, document_prompt=doc_prompt
    )
    retrieval_chain = create_retrieval_chain(hist_retriever, qa_chain)

    return RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history=_get_claude_hist,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


claude_rag_chain = build_claude_chain(llm_claude, vectorstore, k=K_FINAL)
print("[claude] History-aware RAG chain ready.")


def _warmup_minicheck():
    """Load MiniCheck in a background thread to avoid first-query latency."""
    print("[minicheck] Background warm-up starting ...")
    _get_minicheck()
    print("[minicheck] Warm-up complete.")


threading.Thread(target=_warmup_minicheck, daemon=True).start()

# Reinitialise the Claude chain and clear all conversation histories.
def reset_chains():
    global claude_rag_chain
    claude_rag_chain = build_claude_chain(llm_claude, vectorstore, k=K_FINAL)
    clear_claude_history()
    clear_conversation_history()

# PIPELINE-LEVEL CONTEXT PRECISION CACHE

_pipeline_cp_cache: Dict[str, float] = {}

# Metric 1: Answer Relevance - Compute cosine similarity between question and answer embedding vectors.
def answer_relevance_score(question: str, answer: str,       
                           question_vec: np.ndarray = None) -> float:
    try:
        q = question_vec if question_vec is not None else \
            np.array(embeddings.embed_query(question), dtype=np.float32)
        a = np.array(embeddings.embed_query(answer[:1500]), dtype=np.float32)
        d = np.linalg.norm(q) * np.linalg.norm(a)
        return round(float(np.dot(q, a) / d), 3) if d > 0 else 0.0
    except Exception as e:
        print(f"  [eval] answer_relevance error: {e}")
        return 0.0


# MiniCheck model instance — loaded lazily on first evaluation call.
_minicheck = None
_mc_lock   = threading.Lock()


def _get_minicheck():
    global _minicheck
    with _mc_lock:
        if _minicheck is None:
            try:
                from minicheck.minicheck import MiniCheck
                print("  [eval] Loading MiniCheck roberta-large ...")
                _minicheck = MiniCheck(
                    model_name="roberta-large",
                    cache_dir=os.path.join(BASE_DIR, "ckpts"),
                )
                print("  [eval] MiniCheck ready.")
            except ImportError:
                print("  [eval] MiniCheck not installed.")
                print("  pip install 'minicheck @ "
                      "git+https://github.com/Liyan06/MiniCheck.git@main'")
            except Exception as e:
                print(f"  [eval] MiniCheck load failed: {e}")
    return _minicheck

# Strip HTML and link sections before claim extraction.
def _clean_for_eval(answer: str) -> str:
    answer = re.sub(r"Related Documentation.*$", "", answer,
                    flags=re.DOTALL | re.IGNORECASE)
    answer = re.sub(r"<a\s[^>]*>([^<]*)</a>", r"\1", answer)
    answer = re.sub(r"<[^>]+>", "", answer)
    answer = re.sub(r"^\s*[-*]\s+", "", answer, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", answer).strip()

#Split an answer into sentence-level claims for NLI scoring.
def _extract_claims(text: str) -> List[str]:
    claims = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 15 or re.match(r"^\d+[.)]\s*$", line):
            continue
        if len(line) <= 200:
            claims.append(line)
        else:
            for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
                s = sent.strip()
                if len(s) >= 15:
                    claims.append(s)
    return claims

# Extract original text from docs for MiniCheck scoring.
def _get_original_chunks(docs: list, max_chunks: int = 8) -> List[str]:
    chunks = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        raw  = (meta.get("original_content") or doc.page_content or "").strip()
        if len(raw) > 20:
            chunks.append(raw)
    return chunks[:max_chunks]

# Core shared computation for Metrics Faithfulness, Precision, and Recall. All three metrics read from this matrix — no redundant inference calls.
def _compute_minicheck_matrix(docs: list, answer: str):
    scorer = _get_minicheck()
    if scorer is None or not docs or not answer:
        return None, None, None

    chunks = _get_original_chunks(docs)
    if not chunks:
        return None, None, None

    claims = _extract_claims(_clean_for_eval(answer))
    if not claims:
        print("  [eval] No verifiable claims extracted from answer.")
        return None, None, None

    nc  = len(claims)
    nch = len(chunks)
    prob_matrix = [[0.0] * nch for _ in range(nc)]

    for ci, chunk in enumerate(chunks):
        try:
            _, raw_probs, _, _ = scorer.score(docs=[chunk] * nc, claims=claims)
            for ki, p in enumerate(raw_probs):
                prob_matrix[ki][ci] = float(p)
        except Exception as e:
            print(f"  [eval] MiniCheck chunk {ci+1}/{nch} error: {e}")

    return claims, chunks, prob_matrix

# Metric 2: Faithfulness - Compute soft faithfulness as the mean per-claim maximum support probability.
def faithfulness_score(docs: list, answer: str,
                       _matrix_cache: dict = None) -> float:
    if _matrix_cache and _matrix_cache.get("ready"):
        claims, chunks, prob_matrix = (
            _matrix_cache["claims"],
            _matrix_cache["chunks"],
            _matrix_cache["prob_matrix"],
        )
        if claims is None:
            return 0.0
    else:
        claims, chunks, prob_matrix = _compute_minicheck_matrix(docs, answer)
        if claims is None:
            return 0.0

    per_claim_max = [max(row) for row in prob_matrix]
    mean_prob     = float(np.mean(per_claim_max))
    supported     = sum(1 for s in per_claim_max if s >= MINICHECK_THRESHOLD)
    print(f"    [Faithfulness]      {supported}/{len(claims)} claims supported "
          f"| mean_max={mean_prob:.3f}")
    return round(mean_prob, 3)

# Metric 3: Context Precision - Compute Context Precision@k as a rank-weighted signal-to-noise ratio.
def context_precision_score(docs: list, answer: str,
                             _matrix_cache: dict = None) -> float:
    if _matrix_cache and _matrix_cache.get("ready"):
        claims, chunks, prob_matrix = (
            _matrix_cache["claims"],
            _matrix_cache["chunks"],
            _matrix_cache["prob_matrix"],
        )
        if claims is None:
            return 0.0
    else:
        claims, chunks, prob_matrix = _compute_minicheck_matrix(docs, answer)
        if claims is None:
            return 0.0

    k = len(chunks)
    chunk_relevance = []
    for j in range(k):
        col_max = max(prob_matrix[i][j] for i in range(len(claims)))
        print(f"    [CP] chunk {j+1}: max_prob={col_max:.3f} "
              f"{'RELEVANT' if col_max >= MINICHECK_THRESHOLD else 'noise'}")
        chunk_relevance.append(1 if col_max >= MINICHECK_THRESHOLD else 0)

    total_relevant = sum(chunk_relevance)
    if total_relevant == 0:
        print(f"    [Context Precision] 0/{k} chunks relevant → 0.000")
        return 0.0

    running_relevant = 0
    weighted_sum     = 0.0
    for i, rel in enumerate(chunk_relevance, start=1):
        if rel:
            running_relevant += 1
            weighted_sum     += running_relevant / i

    cp = round(weighted_sum / total_relevant, 3)
    print(f"    [Context Precision] {total_relevant}/{k} relevant → {cp:.3f}")
    return cp


def context_recall_score(docs: list, answer: str,
                          _matrix_cache: dict = None) -> float:
    """
    Compute Context Recall as the fraction of answer claims supported by context.

    A claim is considered supported if its maximum MiniCheck probability across
    all retrieved chunks meets MINICHECK_THRESHOLD (binary decision). Recall is
    the count of supported claims divided by the total claim count.
    Reference: Es et al. (EACL 2024) — RAGAS.
    """
    if _matrix_cache and _matrix_cache.get("ready"):
        claims, chunks, prob_matrix = (
            _matrix_cache["claims"],
            _matrix_cache["chunks"],
            _matrix_cache["prob_matrix"],
        )
        if claims is None:
            return 0.0
    else:
        claims, chunks, prob_matrix = _compute_minicheck_matrix(docs, answer)
        if claims is None:
            return 0.0

    nc        = len(claims)
    supported = sum(
        1 for i in range(nc) if max(prob_matrix[i]) >= MINICHECK_THRESHOLD
    )
    recall = round(supported / nc, 3)
    print(f"    [Context Recall]    {supported}/{nc} claims retrievable → {recall:.3f}")
    return recall

# Compute all four RAGAS metrics for a generated response.
def evaluate_response(question: str, answer: str, docs: list,
                      question_vec: np.ndarray = None,
                      pipeline_cp: float = None) -> tuple:
    t0 = time.time()

    ar = answer_relevance_score(question, answer, question_vec)

    claims, chunks, prob_matrix = _compute_minicheck_matrix(docs, answer)
    _matrix_cache = {
        "ready":       claims is not None,
        "claims":      claims,
        "chunks":      chunks,
        "prob_matrix": prob_matrix,
    }

    fa = faithfulness_score(docs, answer, _matrix_cache=_matrix_cache)
    cr = context_recall_score(docs, answer, _matrix_cache=_matrix_cache)

    if pipeline_cp is None:
        pipeline_cp = context_precision_score(
            docs, answer, _matrix_cache=_matrix_cache
        )

    metrics = {
        "answer_relevance":  ar,
        "faithfulness":      fa,
        "context_precision": pipeline_cp,
        "context_recall":    cr,
    }
    return metrics, round(time.time() - t0, 3), pipeline_cp

# Two-tier evaluation printout.
def print_eval(question: str, model: str, metrics: dict, lats: dict,
               is_first_model: bool = False):
    def grade(s):
        return "GOOD" if s >= 0.6 else "MODERATE" if s >= 0.4 else "LOW"

    ar = metrics["answer_relevance"]
    f  = metrics["faithfulness"]
    cp = metrics.get("context_precision", 0.0)
    cr = metrics.get("context_recall",    0.0)
    sep = "-" * 70

    if is_first_model:
        print(f"\n{'='*70}")
        print(f"  TIER 1 — PIPELINE  |  Q: {question[:60]}")
        print(f"  (shared retrieval — identical for all models)")
        print(sep)
        print(f"  Context Precision  {cp:.3f}  {grade(cp)}")
        print(f"  Retrieval Latency  {lats.get('lat_retrieve', 0.0):.2f}s")

    print(f"\n{'='*70}")
    print(f"  TIER 2 — MODEL  |  {model}")
    print(sep)
    print(f"  Answer Relevance   {ar:.3f}  {grade(ar)}")
    print(f"  Faithfulness       {f:.3f}")
    print(f"  Context Recall     {cr:.3f}  {grade(cr)}")
    print(sep)
    print(f"  embed:{lats.get('lat_embed',0):.2f}s  "
          f"retrieve:{lats.get('lat_retrieve',0):.2f}s  "
          f"generate:{lats.get('lat_generate',0):.2f}s  "
          f"eval:{lats.get('lat_eval',0):.2f}s")
    print("=" * 70)

# DATABASE AND CSV LOGGING

FEEDBACK_DB = os.path.join(BASE_DIR, "rag_evaluation_Alfabet.sqlite")
UNIFIED_CSV = os.path.join(BASE_DIR, "rag_evaluation_Alfabet.csv")

CSV_FIELDS = [
    "record_type", "created_at", "session_id", "model_name", "question",
    "answer_or_response", "answer_relevance", "faithfulness",
    "context_precision", "context_recall",
    "lat_embed", "lat_retrieve", "lat_generate", "lat_eval",
    "lat_total", "retrieval_id", "rating", "feedback_text", "regenerate",
]
_NUMERIC_FIELDS = {
    "answer_relevance", "faithfulness", "context_precision", "context_recall",
    "lat_embed", "lat_retrieve", "lat_generate",
    "lat_eval", "lat_total",
}


def _get_conn():
    """Return a SQLite connection with WAL journal mode for concurrent writes."""
    c = sqlite3.connect(FEEDBACK_DB, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    return c


def _init_db():
    """Create the feedback and evaluation tables if they do not exist."""
    with feedback_lock:
        c = _get_conn()
        try:
            c.execute("""CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT, session_id TEXT, model_name TEXT,
                question TEXT, response TEXT, rating INTEGER,
                feedback_text TEXT, regenerate TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS evaluation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT, session_id TEXT, model_name TEXT,
                question TEXT, answer TEXT,
                answer_relevance REAL, faithfulness REAL,
                context_precision REAL, context_recall REAL,
                lat_embed REAL, lat_retrieve REAL, lat_generate REAL,
                lat_eval REAL, lat_total REAL,
                retrieval_id TEXT, evidence_json TEXT)""")
            c.commit()
            # Add columns that may be absent in older database versions.
            for col, typ in [
                ("retrieval_id", "TEXT"), ("evidence_json", "TEXT"),
                ("lat_embed", "REAL"), ("lat_retrieve", "REAL"),
                ("lat_generate", "REAL"), ("lat_eval", "REAL"),
                ("lat_total", "REAL"), ("context_precision", "REAL"),
                ("context_recall", "REAL"),
            ]:
                try:
                    c.execute(f"ALTER TABLE evaluation ADD COLUMN {col} {typ}")
                    c.commit()
                except sqlite3.OperationalError:
                    pass  # Column already exists.
        finally:
            c.close()

# Remove anchor tags and generic HTML from a string before CSV storage
def _strip_html(t: str) -> str:
    if not t:
        return t
    t = re.sub(r'<a\s+href=[\'"][^\'"]*[\'"][^>]*>([^<]*)</a>', r'\1', t)
    return re.sub(r'<[^>]+>', '', t).strip()

# Load the unified CSV log into a list of row dicts
def _load_csv() -> list:
    if not os.path.exists(UNIFIED_CSV):
        return []
    try:
        with open(UNIFIED_CSV, "r", encoding="utf-8-sig") as f:
            first = f.readline().strip()
        delim = ";" if ";" in first else ","
        if not first.startswith("record_type"):
            os.rename(UNIFIED_CSV, UNIFIED_CSV.replace(".csv", "_backup.csv"))
            return []
        with open(UNIFIED_CSV, "r", encoding="utf-8-sig", newline="") as f:
            return [dict(r) for r in csv.DictReader(f, delimiter=delim)]
    except Exception as e:
        print(f"  [csv] read error: {e}")
        return []

# rite all rows to the unified CSV log
def _save_csv(rows: list):
    with open(UNIFIED_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS,
                           extrasaction="ignore", delimiter=";")
        w.writeheader()
        for row in rows:
            out = {}
            for k, v in row.items():
                if k in _NUMERIC_FIELDS and v not in ("", None):
                    try:
                        out[k] = str(float(v)).replace(".", ",")
                    except Exception:
                        out[k] = v
                else:
                    out[k] = v
            w.writerow(out)


def _append_csv(row: dict):
    row["answer_or_response"] = _strip_html(
        row.get("answer_or_response", "")
    )
    rows = _load_csv()
    rows.append(row)
    _save_csv(rows)

# Update the most recent matching row in the CSV log with feedback values
def _apply_feedback(rows, model, question, rating=None, fb=None, regen=None,
                    feedback_text=None, **kwargs) -> bool:
    if feedback_text is None:
        feedback_text = fb if fb is not None else (
            kwargs.get("feedback") or kwargs.get("text")
        )
    if regen is None:
        regen = kwargs.get("regenerate")
    for row in reversed(rows):
        if (row.get("model_name") == model and
                row.get("question", "").strip() == question.strip()):
            row["rating"]        = "" if rating is None else rating
            row["feedback_text"] = "" if feedback_text is None else feedback_text
            row["regenerate"]    = "" if regen is None else regen
            return True
    return False

# Write feedback to the CSV log, queuing it as pending if the row is absent
def _update_csv_feedback(sid, model, question, rating, fb, regen):
    rows = _load_csv()
    if _apply_feedback(rows, model, question, rating, fb, regen):
        _save_csv(rows)
    else:
        key = (model, question.strip())
        with _pending_feedback_lock:
            _pending_feedback[key] = {
                "rating": "" if rating is None else rating,
                "feedback_text": fb,
                "regenerate": regen,
            }


def _flush_pending(sid, model, question):
    """Apply any pending feedback for a question once its evaluation row exists."""
    key = (model, question.strip())
    with _pending_feedback_lock:
        pending = _pending_feedback.pop(key, None)
    if not pending:
        return
    rows = _load_csv()
    if _apply_feedback(rows, model, question, **pending):
        _save_csv(rows)

# Persist a completed evaluation record to the SQLite database and CSV log
def save_evaluation(sid, model, question, answer, metrics, lats, evidence=None):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ar  = metrics["answer_relevance"]
    f   = metrics["faithfulness"]
    cp  = metrics.get("context_precision", 0.0)
    cr  = metrics.get("context_recall",    0.0)
    rid = (evidence or {}).get("retrieval_id", "")
    evj = json.dumps(evidence or {}, ensure_ascii=False)

    with feedback_lock:
        c = _get_conn()
        try:
            c.execute(
                """INSERT INTO evaluation
                   (created_at, session_id, model_name, question, answer,
                    answer_relevance, faithfulness,
                    context_precision, context_recall,
                    lat_embed, lat_retrieve, lat_generate, lat_eval, lat_total,
                    retrieval_id, evidence_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ts, sid, model, question, answer,
                 ar, f, cp, cr,
                 lats.get("lat_embed"), lats.get("lat_retrieve"),
                 lats.get("lat_generate"), lats.get("lat_eval"),
                 lats.get("lat_total"), rid, evj))
            c.commit()
        finally:
            c.close()

        _append_csv({
            "record_type": "evaluation", "created_at": ts,
            "session_id": sid, "model_name": model,
            "question": question, "answer_or_response": answer,
            "answer_relevance":  ar, 
            "faithfulness": f,
            "context_precision": cp, 
            "context_recall": cr,
            "lat_embed":    lats.get("lat_embed", ""),
            "lat_retrieve": lats.get("lat_retrieve", ""),
            "lat_generate": lats.get("lat_generate", ""),
            "lat_eval":     lats.get("lat_eval", ""),
            "lat_total":    lats.get("lat_total", ""),
            "retrieval_id": rid,
            "rating": "", "feedback_text": "", "regenerate": "",
        })
        _flush_pending(sid, model, question)

# Persist a batch of user feedback entries to the database and CSV log
def save_feedback_entries(sid: str, entries: list) -> bool:
    try:
        ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db_rows = []
        for e in entries:
            mn    = str(e.get("model_name") or "Unknown").replace("\x00", "")
            q     = str(e.get("question")   or "").replace("\x00", "")
            resp  = str(e.get("response")   or "").replace("\x00", "")
            rat   = e.get("rating")
            rat   = int(rat) if rat is not None and str(rat).strip() != "" else None
            fb    = str(e.get("feedback")   or "").replace("\x00", "")
            regen = str(e.get("regenerate") or "No").replace("\x00", "")
            db_rows.append((ts, sid, mn, q, resp, rat, fb, regen))

        with feedback_lock:
            c = _get_conn()
            try:
                c.executemany(
                    """INSERT INTO feedback
                       (created_at, session_id, model_name, question,
                        response, rating, feedback_text, regenerate)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    db_rows)
                c.commit()
            finally:
                c.close()
            for _, s, mn, q, _, rat, fb, regen in db_rows:
                _update_csv_feedback(s, mn, q, rat, fb, regen)
        return True
    except Exception as e:
        print(f"Error saving feedback: {e}")
        traceback.print_exc()
        return False


try:
    _init_db()
    print(f"[db]  {FEEDBACK_DB}")
    print(f"[csv] {UNIFIED_CSV}")
except Exception as e:
    print(f"[db] WARNING: {e}")

# CORE CHATBOT RUNNER
# Route a user question to the appropriate model and trigger evaluation

def run_chatbot(question: str, model_name: str, session_id: str) -> str:

    t_start      = time.time()
    docs         = []
    lats         = {}
    evidence     = {}
    question_vec: np.ndarray = None

    if model_name == "Claude Model":
        t0 = time.time()
        question_vec      = np.array(embeddings.embed_query(question),
                                     dtype=np.float32)
        lats["lat_embed"]    = round(time.time() - t0, 3)
        lats["lat_retrieve"] = 0.0

        t0     = time.time()
        result = claude_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        lats["lat_generate"] = round(time.time() - t0, 3)

        answer   = (result.get("answer") or "").strip()
        docs     = result.get("context") or []
        answer   = append_links(answer, docs)
        evidence = build_evidence_bundle(question, docs)

    elif model_name in ("Nova Lite Model", "Nova Pro Model"):
        history         = get_nova_history(session_id, model_name)
        retrieval_query = question

        # Query rewriting: reformulate the latest message as a self-contained
        # retrieval query when conversation history is present.
        if history:
            rewrite_prompt = (
                "Given the conversation history below and the user's latest "
                "message, rewrite the latest message as a fully self-contained "
                "search query for a documentation retrieval system. "
                "Return ONLY the rewritten query, nothing else.\n\n"
                "Conversation history:\n"
            )
            for msg in history[-4:]:
                label = "User" if msg["role"] == "user" else "Assistant"
                rewrite_prompt += f"{label}: {msg['content']}\n"
            rewrite_prompt += f"\nLatest message: {question}\nRewritten query:"

            try:
                rewritten, _ = _nova_call(NOVA_LITE_MODEL_ID, rewrite_prompt)
                retrieval_query = rewritten.strip().strip('"').strip("'")
                if not retrieval_query:
                    retrieval_query = question
                print(f"  [nova rewrite] {question!r} → {retrieval_query!r}")
            except Exception as e:
                print(f"  [nova rewrite] Failed ({e}), using original query.")
                retrieval_query = question

        t0 = time.time()
        question_vec      = np.array(embeddings.embed_query(retrieval_query),
                                     dtype=np.float32)
        lats["lat_embed"] = round(time.time() - t0, 3)

        t0   = time.time()
        docs = _rrf_hybrid_search(retrieval_query, k=K_FINAL)
        lats["lat_retrieve"] = round(time.time() - t0, 3)

        context = build_nova_context(docs)
        history = get_nova_history(session_id, model_name)

        prompt = f"{context}\n\n"
        if history:
            prompt += "Conversation so far:\n"
            for msg in history[-4:]:
                label = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{label}: {msg['content']}\n"
            prompt += "\n"
        prompt += f"Question: {question}\nAnswer:"

        t0 = time.time()
        if model_name == "Nova Lite Model":
            answer, _ = ask_nova_lite(prompt)
        else:
            answer, _ = ask_nova_pro(prompt)
        lats["lat_generate"] = round(time.time() - t0, 3)

        answer   = append_links(answer.strip(), docs)
        evidence = build_evidence_bundle(question, docs)
        add_nova_history(session_id, model_name, "user",      question)
        add_nova_history(session_id, model_name, "assistant", answer)

    else:
        return "Invalid model selected."

    lats["lat_total"] = round(time.time() - t_start, 3)

    # Asynchronous evaluation: uses the pipeline CP cache to compute CP once
    # per question and reuse for all models.
    _d, _ev, _ans  = list(docs), dict(evidence), answer
    _m, _s, _q, _l = model_name, session_id, question, dict(lats)
    _qv             = question_vec
    _rid            = (_ev or {}).get("retrieval_id", "")

    def _run_eval():
        try:
            existing_cp = _pipeline_cp_cache.get(_rid)
            is_first    = existing_cp is None
            metrics, lat_eval, computed_cp = evaluate_response(
                _q, _ans, _d, question_vec=_qv, pipeline_cp=existing_cp,
            )
            if is_first and computed_cp is not None:
                _pipeline_cp_cache[_rid] = computed_cp
            _l["lat_eval"] = lat_eval
            print_eval(_q, _m, metrics, _l, is_first_model=is_first)
            save_evaluation(_s, _m, _q, _ans, metrics, _l, evidence=_ev)
        except Exception as err:
            print(f"  [eval] {_m}: {err}")
            traceback.print_exc()

    threading.Thread(target=_run_eval, daemon=True).start()
    return answer

# SESSION HELPERS

def ensure_session():
    """Assign a unique session ID to the request if one does not exist."""
    if "session_id" not in session:
        base = f"{request.remote_addr}{datetime.now()}{random.randint(1000,9999)}"
        session["session_id"] = hashlib.md5(base.encode()).hexdigest()

# FLASK ROUTES

@app.route("/")
def index():
    ensure_session()
    return render_template("rag_evaluation_Alfabet.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        ensure_session()
        data    = request.json or {}
        message = (data.get("user_message") or "").strip()
        model   = data.get("model_name") or "Claude Model"

        if not message:
            return jsonify({"responses": [{"model": model,
                                           "response": "Please enter a question."}],
                            "limitReached": False})

        if model == "All Models":
            out = []
            for m in ["Claude Model", "Nova Lite Model", "Nova Pro Model"]:
                try:
                    ans = run_chatbot(message, m, session["session_id"])
                except Exception as e:
                    ans = f"Error from {m}: {e}"
                out.append({"model": m, "response": ans})
            return jsonify({"responses": out, "limitReached": False})

        ans = run_chatbot(message, model, session["session_id"])
        return jsonify({"responses": [{"model": model, "response": ans}],
                        "limitReached": False})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"responses": [{"model": "Error",
                                       "response": f"An error occurred: {e}"}],
                        "limitReached": False})


@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        ensure_session()
        data    = request.json or {}
        entries = data.get("entries")

        if isinstance(entries, list):
            cleaned = []
            for e in entries:
                if not isinstance(e, dict):
                    continue
                if not e.get("model_name"):
                    return jsonify({"status": "Error: Missing model_name"})
                if e.get("rating") is None:
                    return jsonify({"status": f"Error: Missing rating for "
                                              f"{e.get('model_name')}"})
                cleaned.append(e)
            ok = save_feedback_entries(session.get("session_id"), cleaned)
            return jsonify({"status": "Feedback saved" if ok
                            else "Error saving feedback"})

        model  = data.get("model_name", "Unknown")
        q      = data.get("question", "")
        resp   = data.get("response", "")
        rating = data.get("rating")
        fb     = data.get("feedback", "")
        regen  = data.get("regenerate", "No")

        if not model:
            return jsonify({"status": "Error: Missing model name"})
        if rating is None:
            return jsonify({"status": "Error: Missing rating"})

        ok = save_feedback_entries(session.get("session_id"), [{
            "model_name": model, "question": q, "response": resp,
            "rating": rating, "feedback": fb, "regenerate": regen,
        }])
        return jsonify({"status": "Feedback saved" if ok
                        else "Error saving feedback"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": f"Error: {e}"})


@app.route("/clear_session", methods=["POST"])
def clear_session_route():
    try:
        ensure_session()
        reset_chains()
        clear_claude_history(session["session_id"])
        clear_conversation_history(session["session_id"])
        session.clear()
        return jsonify({"status": "Session cleared"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "Error clearing session"})


@app.route("/regenerate", methods=["POST"])
def regenerate():
    try:
        ensure_session()
        data       = request.json or {}
        question   = (data.get("user_message") or data.get("question") or "").strip()
        model_name = (data.get("model_name") or "Claude Model").strip()

        if not question:
            return jsonify({"responses": [{"model": model_name,
                                           "response": "No question provided."}]})

        sid = session.get("session_id", "unknown")
        if model_name == "Claude Model":
            clear_claude_history(sid)
        else:
            key = _nk(sid, model_name)
            _nova_hist.pop(key, None)

        answer = run_chatbot(question, model_name, sid)

        def _mark_regen():
            try:
                rows   = _load_csv()
                marked = 0
                for row in reversed(rows):
                    if (row.get("model_name") == model_name and
                            row.get("question", "").strip() == question):
                        row["regenerate"] = "Yes"
                        marked += 1
                        if marked >= 2:
                            break
                if marked:
                    _save_csv(rows)
            except Exception as e:
                print(f"  [regenerate] CSV mark error: {e}")

        threading.Thread(target=_mark_regen, daemon=True).start()

        return jsonify({"responses": [{"model": model_name, "response": answer}],
                        "limitReached": False})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"responses": [{"model": "Error",
                                       "response": f"Regeneration failed: {e}"}],
                        "limitReached": False})
    
# DEBUG ROUTES

@app.route("/debug/feedback")
def debug_feedback():
    try:
        c = _get_conn()
        n = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        c.close()
        return jsonify({"status": "OK", "db": FEEDBACK_DB,
                        "csv": UNIFIED_CSV, "feedback_rows": n})
    except Exception as e:
        return jsonify({"status": f"Error: {e}"})

# Return a per-model accuracy summary from the evaluation database
@app.route("/debug/evaluation")
def debug_evaluation():

    try:
        c = _get_conn()
        eval_rows = c.execute("""
            SELECT model_name,
                COUNT(*) AS n,
                ROUND(AVG(answer_relevance),  3) AS ar,
                ROUND(AVG(faithfulness),      3) AS fa,
                ROUND(AVG(context_precision), 3) AS cp,
                ROUND(AVG(context_recall),    3) AS cre,
                ROUND(AVG(lat_embed),    3) AS le,
                ROUND(AVG(lat_retrieve), 3) AS lr,
                ROUND(AVG(lat_generate), 3) AS lg,
                ROUND(AVG(lat_eval),     3) AS lv,
                ROUND(AVG(lat_total),    3) AS lt
            FROM evaluation
            GROUP BY model_name
            ORDER BY model_name
        """).fetchall()

        fb_rows = c.execute("""
            SELECT model_name,
                SUM(CASE WHEN rating =  1 THEN 1 ELSE 0 END) AS up,
                SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS down,
                ROUND(100.0 * SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END)
                      / NULLIF(COUNT(*), 0), 2) AS h_acc
            FROM feedback
            GROUP BY model_name
        """).fetchall()
        c.close()

        fb_map = {r[0]: {"thumbs_up": r[1], "thumbs_down": r[2],
                          "human_accuracy_pct": r[3]} for r in fb_rows}
        result = []
        for r in eval_rows:
            m = r[0]
            result.append({
                "model":                 m,
                "total_questions":       r[1],
                "avg_answer_relevance":  r[2],
                "avg_faithfulness":      r[3],
                "avg_context_precision": r[4],
                "avg_context_recall":    r[5],
                "avg_lat_embed":    r[6],
                "avg_lat_retrieve": r[7],
                "avg_lat_generate": r[8],
                "avg_lat_eval":     r[9],
                "avg_lat_total":    r[10],
                **fb_map.get(m, {"thumbs_up": 0, "thumbs_down": 0,
                                  "human_accuracy_pct": None}),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": f"Error: {e}"})


@app.route("/debug/search")
def debug_search():
    """Run a similarity search probe and return ranked results for inspection."""
    query = request.args.get("q", "which servers are at risk of failure")
    k     = int(request.args.get("k", 20))
    try:
        results = vectorstore.similarity_search(query, k=k)
        out = []
        for i, doc in enumerate(results, 1):
            meta = doc.metadata or {}
            out.append({
                "rank":    i,
                "heading": meta.get("context_heading", "no heading"),
                "link":    meta.get("context_links", ""),
                "preview": doc.page_content[:150],
            })
        return jsonify({"query": query, "k": k, "results": out})
    except Exception as e:
        return jsonify({"error": str(e)})

# ENTRY POINT
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
