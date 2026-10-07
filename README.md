# Domain-Specific AI Assistant for Technical Documentation
### Master Thesis — University of Koblenz, March 2026
**Author:** Pravina Bariya
**Supervisors:** Prof. Dr. Jan Jürjens & M.Sc. Thomas Reiser

This repository accompanies the master's thesis *Developing a Domain-Specific AI
Assistant for Technical Documentation Using AWS Bedrock Models*. It contains two
retrieval-augmented generation (RAG) pipelines that share one retrieval design
(FAISS dense search + BM25 + reciprocal rank fusion + cross-encoder re-ranking)
and one evaluation approach (NLI-based metrics grounded in RAGAS):

- **Alfabet pipeline** — enterprise product documentation, English, three AWS Bedrock models.
- **Berufearchiv pipeline** — OCR-processed historical German occupational archives
  (about 5,000 scanned files), six LLMs via Groq and Mistral AI.

---

## What This Project Does

This project builds and evaluates two RAG (Retrieval-Augmented Generation)
chatbot pipelines that answer questions from technical documentation.
Instead of relying on general AI knowledge, the system retrieves
the relevant section from real documents first, then generates a
grounded answer based only on that retrieved content.

| Pipeline | Documents | Models Used |
|---|---|---|
| **Alfabet** | Enterprise software docs (Bizzdesign) | Claude 4.5 Sonnet, Amazon Nova Pro, Nova Lite — via AWS Bedrock |
| **Berufearchiv** | Historical German occupational archives (OCR) | Llama-4, Llama-3.3-70B, Llama-3.1-8B, Qwen3-32B via Groq; Mistral-Large, Mistral-Small via Mistral AI |

---

## Project Structure

```
├── rag_evaluation_Alfabet.py          # Flask chatbot — Alfabet pipeline
├── rag_evaluation_berufearchiv.py     # Flask chatbot — Berufearchiv pipeline
├── build_faiss_index_alfabet.py       # Builds FAISS vector index (Alfabet)
├── build_faiss_index_berufearchiv.py  # Builds FAISS vector index (Archive)
├── ocr_berufearchiv.py                # OCR pipeline for scanned PDFs
├── requirements.txt                   # Python dependencies
├── .env.example                       # Required API key names (no real values)
├── rag_evaluation_Alfabet.csv         ## Evaluation results (CSV exports for Alfabet)
├── templates/                         # HTML templates for the chat UI
│   ├── rag_evaluation_Alfabet.html
│   └── rag_evaluation_berufearchiv.html
├── static/                            # CSS and JavaScript for web interface
├── utils/                             # AWS Bedrock helper functions
│   ├── bedrock.py
│   └── __init__.py
└── exports_v6/                        # Evaluation results (CSV exports for Berufearchiv)
```

---

## Before Installing Python Packages

Two OS-level tools must be installed first (required for Berufearchiv pipeline):

**Tesseract OCR:**
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux:   sudo apt install tesseract-ocr tesseract-ocr-deu
- Mac:     brew install tesseract tesseract-lang

**Poppler** (PDF to image conversion):
- Windows: https://github.com/oschwartz10612/poppler-windows/releases
- Linux:   sudo apt install poppler-utils
- Mac:     brew install poppler

---

## Setup

### 1. Clone this repository
```bash
git clone https://github.com/Pbariya/berufearchiv-rag.git
cd berufearchiv-rag
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Install MiniCheck (NLI evaluator — requires separate git install)
```bash
pip install git+https://github.com/Liyan06/MiniCheck.git@main
```

### 5. Set up API keys
Copy .env.example to .env and fill in your real keys:
```bash
cp .env.example .env       # Windows: copy .env.example .env
```
AWS credentials are configured separately via AWS CLI:
```bash
aws configure
```

---

## Running the Alfabet Pipeline (AWS Bedrock)

**Step 1 — Build the FAISS search index:**
```bash
python build_faiss_index_alfabet.py
```
> Requires Alfabet JSON documentation files in ./DataSets/Alfabet_11_10_1_JSON/
> The proprietary Alfabet documentation is not included in this repository.

**Step 2 — Start the chatbot:**
```bash
python rag_evaluation_Alfabet.py
```
Open browser: http://localhost:5000

---

## Running the Berufearchiv Pipeline (Groq and Mistral AI Models)

**Step 1 — Extract text from scanned PDFs using OCR:**
```bash
python ocr_berufearchiv.py
```
> Reads PDFs from ./ DATA/archivdaten/
> Outputs JSON files to ./Archivdaten_jsons/

**Step 2 — Build the FAISS search index:**
```bash
python build_faiss_index_berufearchiv.py
```

**Step 3 — Start the chatbot:**
```bash
python rag_evaluation_berufearchiv.py
```
Open browser: http://localhost:5000

### Rate limits and failover

The six Berufearchiv models run on free-tier APIs (Groq, Mistral AI). A circuit
breaker fails over to Cerebras (Qwen3-32B) when a primary endpoint is
rate-limited. The provider used for each record is logged in the `provider`
field, and failover events are logged with a `circuit_open` flag.

---

## How the Pipeline Works

```
User types a question
        ↓
[1] Question converted to a vector (Titan V2 for Alfabet / BGE-M3 for Archive)
        ↓
[2] Hybrid retrieval:
    FAISS dense search  +  BM25 keyword search
    → Combined with Reciprocal Rank Fusion (RRF)
    → Cross-encoder reranker selects best 5 chunks
        ↓
[3] Top 5 document chunks + question sent to language model
        ↓
[4] Model generates answer grounded in retrieved context
    Links validated from metadata (Alfabet pipeline only)
        ↓
[5] Background evaluation (does not slow down the user):
    Answer Relevance  |  Faithfulness  |  Context Precision  |  Context Recall
    (computed using NLI — MiniCheck for Alfabet, mDeBERTa for Berufearchiv)
        ↓
Answer shown to user — all results saved to SQLite + CSV
```

**Relation to RAGAS.** The four metrics follow the RAGAS metric suite. They differ
from the reference RAGAS implementation in two ways: factual consistency is scored
with an NLI model (MiniCheck for English, mDeBERTa-v3 for German) instead of an LLM
judge, and one shared NLI matrix serves three of the metrics. Evaluation runs
asynchronously in a background thread (mean 134–225 s per query), so it does not
delay the user-facing response.

---

## Evaluation Results

### Evaluation overview

| | Alfabet | Berufearchiv |
|---|---|---|
| Models | 3 (Claude 4.5 Sonnet, Amazon Nova Pro, Amazon Nova Lite via AWS Bedrock) | 6 (Llama-4 Maverick, Llama-3.3-70B, Llama-3.1-8B, Qwen3-32B via Groq; Mistral-Large, Mistral-Small via Mistral AI) |
| Records | 189 (63 per model: 61 unique questions plus 2 regeneration records; 50 in scope, 11 out of scope) | 264 (44 questions per model) |
| Total | **453 evaluation records across 9 models and two corpora** | |

Human ratings (acceptable / not acceptable) were given by a single annotator, the
thesis author, blind to the automated scores. Treat the approval rates as
indicative, not definitive (see Limitations).

### Alfabet Pipeline (AWS Bedrock) — 63 records per model

In-scope responses only (abstentions excluded; n = 55 / 49 / 44 answered in-scope responses).

| Model | Human Approval (in-scope) | Faithfulness | Context Precision |
|---|---|---|---|
| Claude 4.5 Sonnet | **80.0%** | 0.688 | **0.897** |
| Amazon Nova Pro   | 79.6%     | 0.701 | 0.691     |
| Amazon Nova Lite  | 77.3%     | **0.746** | 0.684 |

### Berufearchiv Pipeline — 44 questions per model

Mean over all 44 questions per model, abstentions included.

| Model | Faithfulness | Context Recall | Abstention Rate |
|---|---|---|---|
| Mistral-Small  | **0.798** | **0.833** | 36.4%    |
| Llama-3.1-8B   | 0.703     | 0.784     | **6.8%** |
| Mistral-Large  | 0.660     | 0.731     | 43.2%    |

Across all 44 questions per model, Llama-3.1-8B had the lowest abstention rate
(6.8%), Mistral-Large the highest (43.2%) and the highest generation latency
(4.59 s). Context precision ranged from 0.712 to 0.896.

Cost and latency (Alfabet pipeline): Amazon Nova Lite is about 50× cheaper than
Claude 4.5 Sonnet per token at list price (Claude input $3.00 per million tokens);
this is a price ratio per token, not a measured cost per answer. Mean total
response time was 10.6 s for Claude 4.5 Sonnet, 4.2 s for Nova Pro and 4.4 s for
Nova Lite (Claude's retrieval time is absorbed inside the LangChain chain and is
not timed separately).

Full evaluation results including all models and per-question metrics
are available in the exports_v6/ folder as CSV files.

## CI/CD

A GitHub Actions workflow (`.github/workflows/deploy.yml`) runs the tests on every
push and pull request. The build and deploy jobs (Docker image to Artifact Registry,
then Google Cloud Run in europe-west3) are skipped until the repository variable
`DEPLOY_ENABLED` is set to `true`. Before enabling it, configure a GCP project,
replace the placeholder project ID `YOUR-GCP-PROJECT-ID` in the workflow, and add
the service-account key and secrets described in the workflow header. No deployment
is configured in this repository.

## Limitations

- Human ratings come from one annotator (the author); inter-annotator agreement could not be computed.
- Model-specific query rewriting means retrieval was not identical across the three Alfabet models (only about 5 of 61 questions produced identical retrieval), so Context Precision partly reflects the rewriting step.
- Evaluation is reference-free; there is no golden dataset of verified answers. A golden dataset of frequently asked questions is scoped as future work.
- Berufearchiv runs were spread across several sessions because of free-tier rate limits, and each evaluation was run once.
- Llama-4 Maverick was later decommissioned on Groq (March 2026); its results reflect its operating period.
- Of the Alfabet refusals, 37 of 41 were refusals of questions the annotator judged answerable (87.5–92.9% per model), pointing to retrieval coverage as the main cause.

---

## Key Notes

- AWS credentials are configured via AWS CLI (`aws configure`), not in .env
- The Alfabet knowledge base is proprietary and not included in this repository
- The Berufearchiv knowledge base is too large and not included in this repository
- Tesseract and Poppler must be installed at OS level before running the OCR script
- MiniCheck requires a separate git-based install command (see Setup step 4)
- CUDA GPU is used automatically for BGE-M3 embeddings if available

---

## Reference

This repository accompanies the Master's thesis:
*Developing a Domain-Specific AI Assistant for Technical Documentation
Using AWS Bedrock Models*
Faculty 4: Computer Science, University of Koblenz, March 2026.
