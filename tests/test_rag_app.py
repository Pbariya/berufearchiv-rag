# tests/test_rag_app.py
# CI-safe tests — mocks all heavy ML dependencies
# so the pipeline runs in under 2 minutes without downloading 6GB of models

import sys
import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

# ── Set fake env vars FIRST ───────────────────────────────────────────────────
os.environ["GROQ_API_KEY"]     = "ci-test-key"
os.environ["MISTRAL_API_KEY"]  = "ci-test-key"
os.environ["CEREBRAS_API_KEY"] = "ci-test-key"

# ── Build fake objects to replace real ML models ──────────────────────────────
_mock_index = MagicMock()
_mock_index.ntotal = 10
_mock_index.metric_type = 0
_mock_index.search.return_value = (
    np.array([[0.95, 0.88, 0.81, 0.74, 0.67]], dtype="float32"),
    np.array([[0, 1, 2, 3, 4]])
)

_mock_embed = MagicMock()
_mock_embed.encode.return_value = np.random.rand(1, 1024).astype("float32")

_mock_reranker = MagicMock()
_mock_reranker.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

_mock_meta = [
    {
        "source":           f"berufearchiv_{i:04d}.pdf",
        "Context_Heading":  f"Berufsbild {i}",
        "Context_Content":  f"Inhalt des Berufsbildes Nummer {i}. " * 10,
        "Context_Links":    f"berufearchiv_{i:04d}.pdf",
        "text":             f"Eintrag {i}",
        "chunk_id":         0,
        "chunked":          False,
        "char_count":       200,
        "page_count":       2,
    }
    for i in range(10)
]

# ── Stub heavy packages that CI does not install ─────────────────────────────
# The CI job installs only flask, pytest, numpy, rank-bm25, groq and python-dotenv.
# patch("faiss.read_index") and the app's own `import torch` / `import faiss`
# need importable modules, so register MagicMock stand-ins when the real
# package is missing. Where the real packages exist (a developer machine,
# the Docker image) nothing changes.
for _pkg in ("faiss", "torch", "sentence_transformers", "transformers"):
    try:
        __import__(_pkg)
    except ImportError:
        _stub = MagicMock()
        if _pkg == "torch":
            _stub.cuda.is_available.return_value = False   # app takes the CPU path
        sys.modules[_pkg] = _stub

# The app opens Faiss_Metadata/metadata.json at import. That file is not in the
# repository (the knowledge base is not published), so serve an empty stand-in for
# that one path and leave every other file open untouched. json.load is patched below.
_real_open = open
def _open_stub(path, *args, **kwargs):
    if str(path).endswith("metadata.json"):
        return mock_open(read_data="[]")()
    return _real_open(path, *args, **kwargs)

# ── Patch BEFORE importing the app ───────────────────────────────────────────
# The app loads models at module level — patches must be active during import
with patch("faiss.read_index", return_value=_mock_index), \
     patch("json.load", return_value=_mock_meta), \
     patch("builtins.open", side_effect=_open_stub), \
     patch("sentence_transformers.SentenceTransformer", return_value=_mock_embed), \
     patch("sentence_transformers.CrossEncoder", return_value=_mock_reranker), \
     patch("transformers.pipeline", return_value=MagicMock()), \
     patch("groq.Groq", return_value=MagicMock()), \
     patch("os.makedirs"):

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import rag_evaluation_berufearchiv as rag


@pytest.fixture
def client():
    rag.app.config["TESTING"] = True
    with rag.app.test_client() as c:
        yield c


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_returns_200(client):
    """Docker and CI pipeline require /health → 200."""
    resp = client.get("/health")
    assert resp.status_code == 200

def test_health_status_ok(client):
    assert client.get("/health").get_json()["status"] == "ok"

def test_health_has_faiss_info(client):
    data = client.get("/health").get_json()
    assert "faiss_vectors" in data
    assert data["faiss_vectors"] == 10


# ── /ask input validation ─────────────────────────────────────────────────────

def test_empty_question_returns_400(client):
    resp = client.post("/ask",
        data=json.dumps({"question": ""}),
        content_type="application/json")
    assert resp.status_code == 400

def test_unknown_model_returns_400(client):
    resp = client.post("/ask",
        data=json.dumps({"question": "Was ist ein Schreiner?",
                         "model": "GPT-99-Fake"}),
        content_type="application/json")
    assert resp.status_code == 400


# ── Config sanity ─────────────────────────────────────────────────────────────

def test_default_model_in_registry():
    assert rag.DEFAULT_MODEL in rag.MODEL_OPTIONS

def test_all_models_have_provider():
    for name, cfg in rag.MODEL_OPTIONS.items():
        assert "provider" in cfg, f"'{name}' missing provider"
        assert "id" in cfg, f"'{name}' missing id"

def test_fallback_chain_valid():
    for primary, fallbacks in rag.FALLBACK_CHAIN.items():
        for fb in fallbacks:
            assert fb in rag.MODEL_OPTIONS, \
                f"Fallback '{fb}' for '{primary}' not in MODEL_OPTIONS"


# ── Pure logic functions ──────────────────────────────────────────────────────

def test_extract_claims_max_10():
    long = ". ".join([f"Aussage {i} über den Beruf Schreiner" for i in range(20)])
    assert len(rag._extract_claims(long)) <= 10

def test_raw_to_sim_range():
    rag.FAISS_IS_IP = True
    for raw in [-1.0, 0.0, 0.5, 1.0]:
        sim = rag.raw_to_sim(raw)
        assert 0.0 <= sim <= 1.0


# ── /models endpoint ──────────────────────────────────────────────────────────

def test_models_returns_200(client):
    assert client.get("/models").status_code == 200

def test_all_models_in_response(client):
    data = client.get("/models").get_json()
    for name in rag.MODEL_OPTIONS:
        assert name in data
