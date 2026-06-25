"""
build_faiss_index_alfabet.py
-----------------------------
Constructs a FAISS vector index from the Alfabet Enterprise Architecture
Management documentation corpus (JSON format).

Each document chunk is optionally enriched with a one-sentence context
summary prepended to its content before embedding. This contextual
enrichment technique, introduced by Anthropic (2024), improves retrieval
recall for short or generic chunks whose raw text lacks standalone meaning.

Embedding model : Amazon Titan Text Embeddings V2 (via Amazon Bedrock)
Index backend   : LangChain FAISS wrapper
Output          : FAISS index files in OUTPUT_PATH

References
----------
Anthropic. (2024). Introducing contextual retrieval.
    https://www.anthropic.com/news/contextual-retrieval

Usage
-----
    python build_faiss_index_alfabet.py             # full build
    python build_faiss_index_alfabet.py --inspect   # preview document structure
    python build_faiss_index_alfabet.py --dry-run   # enrich first 5 docs only
    python build_faiss_index_alfabet.py --no-enrich # skip contextual enrichment
"""

import os
import json
import sys
import time
from typing import Any, Dict, List


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_PATH  = "./DataSets/Alfabet_11_10_1_JSON"
OUTPUT_PATH = "./Final_Evaluation/Faiss/Alfabet_11_10_1_JSON_v8"
INDEX_NAME  = "Alfabet_11_10_1_JSON_v8"

# Titan Text Embeddings V2 supports dimensions 256, 512, or 1024.
# 1024 yields the highest retrieval quality; 512 halves index size
# with a moderate quality trade-off.
EMBED_DIMENSIONS = 1024

# Model used for contextual summary generation (enrichment step).
# Claude Haiku is selected for its low inference latency and cost.
ENRICH_MODEL_ID   = "anthropic.claude-haiku-4-5-20251001"
ENRICH_MAX_TOKENS = 80    # one-sentence summaries require at most ~80 tokens
ENRICH_BATCH_PAUSE = 0.1  # inter-call pause (seconds) to respect API rate limits

# Documents shorter than this threshold are skipped during loading.
MIN_CONTENT_LENGTH = 20


# =============================================================================
# BEDROCK CLIENT
# =============================================================================

def _get_clients():
    """
    Initialise and return the Bedrock management and runtime clients.

    Credentials are read from environment variables:
        AWS_DEFAULT_REGION   : deployment region (default: us-east-1)
        BEDROCK_ASSUME_ROLE  : optional IAM role ARN for cross-account access
    """
    import boto3
    sys.path.append(os.path.abspath(".."))
    from utils import bedrock as bedrock_utils  # type: ignore

    boto3_bedrock = bedrock_utils.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    return boto3_bedrock, bedrock_runtime


# =============================================================================
# CONTEXTUAL ENRICHMENT
# =============================================================================

def _generate_context_summary(bedrock_runtime, heading: str,
                               content: str) -> str:
    """
    Generate a one-sentence summary contextualising a documentation chunk.

    For longer, self-contained chunks (>400 characters with a descriptive
    heading), the heading combined with the first sentence is sufficient
    context, so no API call is made. For shorter or ambiguously-headed
    chunks, a prompt is sent to the enrichment model to produce a summary
    grounded in Alfabet-specific terminology.

    Falls back to the heading string if the API call fails.

    Parameters
    ----------
    bedrock_runtime : boto3 client — Bedrock runtime client.
    heading         : str — section heading from Context_Heading field.
    content         : str — section body from Context_Content field.

    Returns
    -------
    str — one-sentence summary, or heading on failure.
    """
    if len(content) > 400 and heading and len(heading) > 20:
        first_sentence = content.split(".")[0].strip()
        return f"{heading}. {first_sentence}." if first_sentence else heading

    prompt = (
        f"You are indexing Alfabet Enterprise Architecture Management documentation.\n\n"
        f"Section heading: {heading or '(no heading)'}\n"
        f"Section content: {content[:600]}\n\n"
        f"Write ONE concise sentence (max 30 words) describing what this section "
        f"is about. Be specific — mention the Alfabet feature or concept. "
        f"Do not start with 'This section'."
    )

    try:
        resp = bedrock_runtime.converse(
            modelId=ENRICH_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": ENRICH_MAX_TOKENS, "temperature": 0.0},
        )
        summary = resp["output"]["message"]["content"][0]["text"].strip()
        summary = summary.strip('"').strip("'")
        if summary.lower().startswith("summary:"):
            summary = summary[8:].strip()
        return summary
    except Exception as e:
        print(f"  [enrich] API error: {e} — falling back to heading")
        return heading or "(no context)"


def enrich_content(bedrock_runtime, heading: str, content: str,
                   use_enrichment: bool = True) -> tuple:
    """
    Prepend a context summary to a document chunk.

    The enriched format stored in the FAISS index is:
        "[Context: <heading>. <summary>]\\n<original content>"

    The original (unenriched) content is stored separately in metadata
    under the key 'original_content'. This separation is necessary because
    faithfulness evaluation (MiniCheck) scores against the source prose,
    not the enriched prefix, to avoid penalising grounded responses for
    text the model did not generate (Anthropic, 2024).

    Parameters
    ----------
    use_enrichment : bool — if False, content is returned unchanged.

    Returns
    -------
    tuple[str, str] — (enriched_text, summary). Summary is "" if skipped.
    """
    if not use_enrichment:
        return content, ""

    summary        = _generate_context_summary(bedrock_runtime, heading, content)
    context_prefix = f"[Context: {heading or 'Alfabet documentation'}. {summary}]"
    return f"{context_prefix}\n{content}", summary


# =============================================================================
# INSPECTION MODE
# =============================================================================

def inspect(folder: str, max_files: int = 3):
    """
    Print the structure of up to max_files JSON files without making API calls.

    Intended for verifying dataset schema before committing to a full index
    build. Displays raw field values and a simulated enriched prefix.
    """
    sep = "=" * 70
    print(sep)
    print("  JSON STRUCTURE INSPECTOR  (no API calls)")
    print(f"  Folder: {folder}")
    print(sep)

    if not os.path.isdir(folder):
        print(f"\n  ERROR: folder not found: {folder}")
        return

    files = [f for f in sorted(os.listdir(folder)) if f.endswith(".json")]
    print(f"\n  JSON files found: {len(files)}")

    for filename in files[:max_files]:
        path = os.path.join(folder, filename)
        print(f"\n  ---- {filename} ----")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            entries = raw if isinstance(raw, list) else [raw]
            print(f"  Entries: {len(entries)}")
            for i, e in enumerate(entries[:2]):
                heading = (e.get("Context_Heading") or "").strip()
                link    = (e.get("Context_Links")   or "").strip()
                content = (e.get("Context_Content") or "").strip()
                print(f"\n  Entry {i+1}:")
                print(f"    Context_Heading : {heading[:80]!r}")
                print(f"    Context_Links   : {link!r}")
                print(f"    Context_Content : {content[:120]!r}")
                sim_prefix = (f"[Context: {heading or 'Alfabet documentation'}. "
                              f"Describes {heading or 'this feature'}.]")
                print(f"    Enriched prefix : {sim_prefix!r}  (simulated)")
        except Exception as ex:
            print(f"  ERROR reading {filename}: {ex}")

    print(f"\n{sep}")
    print("  Run without --inspect to build the index.")
    print(sep)


# =============================================================================
# DOCUMENT LOADING
# =============================================================================

def load_documents(folder: str, bedrock_runtime, use_enrichment: bool = True,
                   dry_run: bool = False) -> List:
    """
    Load JSON entries from disk, apply contextual enrichment, and return
    a list of LangChain Document objects.

    Each entry is expected to contain Context_Heading, Context_Links, and
    Context_Content fields. Entries missing content or falling below
    MIN_CONTENT_LENGTH are skipped. The 'original_content' metadata field
    stores the pre-enrichment text for downstream faithfulness evaluation.

    Parameters
    ----------
    folder         : str  — input directory of JSON files.
    bedrock_runtime        — Bedrock runtime client for enrichment API calls.
    use_enrichment : bool — whether to apply contextual enrichment.
    dry_run        : bool — if True, stops after 5 documents.

    Returns
    -------
    list[Document] — LangChain Document objects with enriched page_content.
    """
    from langchain_core.documents import Document

    documents  = []
    skipped    = 0
    no_link    = 0
    files_read = 0

    all_files = [f for f in sorted(os.listdir(folder)) if f.endswith(".json")]

    for filename in all_files:
        path = os.path.join(folder, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"  [skip] Cannot read {filename}: {e}")
            skipped += 1
            continue

        entries    = raw if isinstance(raw, list) else [raw]
        files_read += 1

        for entry in entries:
            if not isinstance(entry, dict):
                skipped += 1
                continue

            content = (
                entry.get("Context_Content") or
                entry.get("content") or
                entry.get("text") or ""
            ).strip()

            if len(content) < MIN_CONTENT_LENGTH:
                skipped += 1
                continue

            link    = (entry.get("Context_Links")   or "").strip()
            heading = (entry.get("Context_Heading") or "").strip()
            if not link:
                no_link += 1

            enriched_content, summary = enrich_content(
                bedrock_runtime, heading, content,
                use_enrichment=use_enrichment,
            )
            if use_enrichment and summary:
                print(f"  [enrich] {filename} | heading={heading[:40]!r} "
                      f"| summary={summary[:60]!r}")
            time.sleep(ENRICH_BATCH_PAUSE)

            meta: Dict[str, Any] = {
                "context_links":    link,
                "context_heading":  heading,
                "related_content":  (entry.get("Related_Content") or "").strip(),
                "source_file":      filename,
                "original_content": content,   # used by MiniCheck faithfulness evaluation
                "context_summary":  summary,
            }
            documents.append(Document(page_content=enriched_content, metadata=meta))

            if dry_run and len(documents) >= 5:
                print(f"\n  [dry-run] Stopped after {len(documents)} documents.")
                return documents

    print(f"  JSON files read : {files_read}")
    print(f"  Documents built : {len(documents)}")
    if skipped:
        print(f"  Entries skipped : {skipped}")
    if no_link:
        print(f"  Missing links   : {no_link}")
    return documents


# =============================================================================
# INDEX BUILD
# =============================================================================

def build(use_enrichment: bool = True, dry_run: bool = False):
    """
    Execute the full index build pipeline.

    Stages
    ------
    1. Load and optionally enrich all documents.
    2. Verify a sample of enriched outputs.
    3. Embed all documents via Bedrock Titan Text Embeddings V2 and
       persist the FAISS index to OUTPUT_PATH.

    Parameters
    ----------
    use_enrichment : bool — apply contextual enrichment (default True).
    dry_run        : bool — process 5 documents and skip index persistence.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_aws import BedrockEmbeddings

    boto3_bedrock, bedrock_runtime = _get_clients()

    embed_model = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        client=boto3_bedrock,
        model_kwargs={"dimensions": EMBED_DIMENSIONS, "normalize": True},
    )

    sep = "=" * 70
    print(sep)
    print("  ALFABET FAISS INDEX BUILDER")
    print(f"  Input        : {INPUT_PATH}")
    print(f"  Output       : {OUTPUT_PATH}")
    print(f"  Index name   : {INDEX_NAME}")
    print(f"  Embedding    : amazon.titan-embed-text-v2:0  (dim={EMBED_DIMENSIONS})")
    print(f"  Enrichment   : {ENRICH_MODEL_ID if use_enrichment else 'disabled'}")
    print(f"  Dry run      : {dry_run}")
    print(sep)

    if not os.path.isdir(INPUT_PATH):
        raise FileNotFoundError(
            f"Input folder not found: {INPUT_PATH}\n"
            "Update INPUT_PATH at the top of this script."
        )

    # Stage 1: document loading and enrichment
    print("\n[1/3] Loading documents ...")
    docs = load_documents(INPUT_PATH, bedrock_runtime,
                          use_enrichment=use_enrichment, dry_run=dry_run)
    if not docs:
        raise RuntimeError("No documents loaded. Run with --inspect to debug.")

    if dry_run:
        print("\n  [dry-run] Sample outputs (first 3 documents):")
        for i, doc in enumerate(docs[:3]):
            print(f"\n  Doc {i+1}:")
            print(f"    page_content[:200]   : {doc.page_content[:200]!r}")
            print(f"    original_content[:80]: "
                  f"{doc.metadata.get('original_content','')[:80]!r}")
            print(f"    context_summary      : {doc.metadata.get('context_summary','')!r}")
        print("\n  [dry-run] Index not written. Remove --dry-run for full build.")
        return

    # Stage 2: sample verification
    print("\n[2/3] Sample verification (first 3 documents):")
    for i, doc in enumerate(docs[:3]):
        m = doc.metadata
        print(f"\n  Doc {i+1}:")
        print(f"    page_content[:150]   : {doc.page_content[:150]!r}")
        print(f"    original_content[:80]: {m.get('original_content','')[:80]!r}")
        print(f"    context_summary      : {m.get('context_summary','')!r}")
        print(f"    context_links        : {m.get('context_links','')!r}")
        if not m.get("context_links"):
            print("    WARNING: context_links is empty in this entry.")

    # Stage 3: embedding and persistence
    print(f"\n[3/3] Embedding {len(docs)} documents ...")
    vs = FAISS.from_documents(docs, embed_model)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    vs.save_local(OUTPUT_PATH, index_name=INDEX_NAME)

    print()
    print(sep)
    print("  INDEX BUILD COMPLETE")
    print(f"  Saved to   : {OUTPUT_PATH}")
    print(f"  Vectors    : {len(docs)}")
    print(f"  Dimensions : {EMBED_DIMENSIONS}")
    print(f"  Enriched   : {use_enrichment}")
    print(sep)
    print(f"\n  Update CANDIDATE_INDEXES in rag_evaluation_pipeline.py:")
    print(f'  ("{OUTPUT_PATH}", "{INDEX_NAME}")')


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args       = sys.argv[1:]
    do_inspect = "--inspect"   in args
    do_dry_run = "--dry-run"   in args
    no_enrich  = "--no-enrich" in args

    if do_inspect:
        inspect(INPUT_PATH)
    else:
        build(use_enrichment=not no_enrich, dry_run=do_dry_run)
