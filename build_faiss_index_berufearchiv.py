import os
import json
import re

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# CONFIGURATION

BASE       = os.path.dirname(os.path.abspath(__file__))
JSON_DIR   = os.path.join(BASE, "Archivdaten_jsons")
OUTPUT_DIR = os.path.join(BASE, "Faiss_Metadata")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Device selection: use GPU if available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chunking thresholds: documents exceeding any of these limits are split.
CHAR_THRESHOLD  = 12000
WORD_THRESHOLD  = 1500
PAGE_THRESHOLD  = 6

# Chunking parameters for the sliding-window fallback.
CHUNK_CHARS     = 2200   # maximum characters per chunk
OVERLAP_CHARS   = 250    # overlap between consecutive chunks
MIN_CHUNK_CHARS = 300    # minimum chunk length; shorter chunks are discarded

# TEXT UTILITIES
# normalise raw OCR output for embedding

def clean_text(t: str) -> str:

    if not t:
        return ""
    t = t.replace("\u00ad", "")
    t = t.replace("\x00", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# Heuristically extract the document title from OCR output.
def extract_heading(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""
    for line in lines[:8]:
        if 5 < len(line) < 120 and (line.isupper() or line.istitle()):
            return line
    return lines[0][:120]

# CHUNKING
# determine whether a document should be split into chunks

def should_chunk(char_count, word_count, page_count) -> bool:
    if page_count  is not None and page_count  >= PAGE_THRESHOLD:
        return True
    if char_count  is not None and char_count  >  CHAR_THRESHOLD:
        return True
    if word_count  is not None and word_count  >  WORD_THRESHOLD:
        return True
    return False

#Split a document into overlapping text chunks.
def chunk_text(text: str, chunk_chars=CHUNK_CHARS,
               overlap=OVERLAP_CHARS) -> list:
    text = clean_text(text)
    if len(text) <= chunk_chars:
        return [text] if len(text) >= MIN_CHUNK_CHARS else []

    blocks = [b.strip() for b in re.split(r"(?:\n\s*\n)+", text)
              if len(b.strip()) >= MIN_CHUNK_CHARS]
    if blocks and max(len(b) for b in blocks) <= chunk_chars:
        return blocks

    chunks, start = [], 0
    L = len(text)
    while start < L:
        end = min(start + chunk_chars, L)
        ch  = text[start:end].strip()
        if len(ch) >= MIN_CHUNK_CHARS:
            chunks.append(ch)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

# RECORD LOADING
#Load JSON documents from disk and construct embedding-ready records.

def load_records(json_dir: str):
    
    texts, metadata = [], []
    skipped_empty = skipped_tiny = json_count = 0

    for fn in sorted(os.listdir(json_dir)):
        if not fn.endswith(".json"):
            continue

        json_count += 1
        with open(os.path.join(json_dir, fn), "r", encoding="utf-8") as f:
            obj = json.load(f)

        raw_text = clean_text(obj.get("text", "") or "")
        if not raw_text:
            skipped_empty += 1
            continue

        m          = obj.get("metadata", {}) or {}
        char_count = m.get("char_count", len(raw_text))
        word_count = m.get("word_count")
        page_count = m.get("page_count")
        source     = obj.get("source", fn)
        full_path  = obj.get("full_path", "")
        heading    = extract_heading(raw_text)

        base_meta = {
            "Context_Heading": heading,
            "Context_Links":   source,
            "source":          source,
            "full_path":       full_path,
            "char_count":      char_count,
            "word_count":      word_count,
            "page_count":      page_count,
            "file_name":       fn,
        }

        if not should_chunk(char_count, word_count, page_count):
            if len(raw_text) < MIN_CHUNK_CHARS:
                skipped_tiny += 1
                continue
            texts.append(raw_text)
            metadata.append({
                **base_meta,
                "Context_Content": raw_text,
                "chunked":         False,
                "chunk_id":        0,
                "text":            raw_text,
            })
        else:
            chunks = chunk_text(raw_text)
            if not chunks:
                skipped_tiny += 1
                continue
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadata.append({
                    **base_meta,
                    "Context_Content": chunk,
                    "chunked":         True,
                    "chunk_id":        i,
                    "text":            chunk,
                })

    print(f"JSON files scanned : {json_count}")
    print(f"Skipped (empty)    : {skipped_empty}")
    print(f"Skipped (too short): {skipped_tiny}")
    return texts, metadata

# MAIN

print(f"JSON_DIR : {JSON_DIR}")
print(f"Exists   : {os.path.exists(JSON_DIR)}")
if not os.path.exists(JSON_DIR):
    raise RuntimeError(
        "JSON_DIR not found. Verify the BASE path and Google Drive mount."
    )

print("\nLoading and chunking documents ...")
texts, meta = load_records(JSON_DIR)

print(f"\nEmbeddable records : {len(texts)}")
print(f"  Chunked          : {sum(1 for m in meta if m['chunked'])}")
print(f"  Whole documents  : {sum(1 for m in meta if not m['chunked'])}")

if not texts:
    raise RuntimeError("No embeddable text found. Check JSON contents.")

# Preview the first record to verify schema before committing to embedding.
print("\nSample metadata[0]:")
sample = {k: v for k, v in meta[0].items() if k != "Context_Content"}
sample["Context_Content_preview"] = meta[0]["Context_Content"][:200] + "…"
print(json.dumps(sample, ensure_ascii=False, indent=2))

# EMBEDDING

print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}  (device: {DEVICE})")
if DEVICE == "cuda":
    encoder = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )
else:
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

# Batch size: 64 is typical for 16 GB VRAM; reduce if out-of-memory errors occur.
batch_size = 64 if DEVICE == "cuda" else 16

embeddings = encoder.encode(
    texts,
    normalize_embeddings=True,   # L2 normalisation enables cosine sim via dot product
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=batch_size,
).astype("float32")

# INDEX CONSTRUCTION

dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # exact inner product; cosine sim on normalised vectors
index.add(embeddings)
print(f"FAISS index size: {index.ntotal} vectors  (dim={dim})")

# PERSISTENCE

index_path = os.path.join(OUTPUT_DIR, "index.faiss")
meta_path  = os.path.join(OUTPUT_DIR, "metadata.json")

faiss.write_index(index, index_path)
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\nIndex saved  : {index_path}")
print(f"Metadata saved : {meta_path}")
print(f"Total vectors  : {index.ntotal}  |  Metadata rows: {len(meta)}")
