# health_route.py
# ══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION: Copy the code below and paste it into
# rag_evaluation_berufearchiv.py
#
# WHERE: Find this line in your file:
#            @app.route("/ask", methods=["POST"])
#        Paste the block ABOVE it (leave a blank line between)
#
# WHY: Required by Docker HEALTHCHECK, GitHub Actions deploy step,
#      and Cloud Run uptime monitoring.
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    """
    Health check endpoint.
    Returns 200 if app and FAISS index loaded correctly.
    Returns 503 if FAISS index missing or empty.

    Called by:
      - Docker HEALTHCHECK (Dockerfile)
      - GitHub Actions: curl "$SERVICE_URL/health" after deploy
      - Cloud Run uptime checks
    """
    if faiss_index is None or len(faiss_meta) == 0:
        return jsonify({
            "status": "degraded",
            "reason": "FAISS index not loaded — check Faiss_Metadata/ volume mount",
        }), 503

    return jsonify({
        "status":        "ok",
        "faiss_vectors": faiss_index.ntotal,
        "metadata_docs": len(faiss_meta),
        "eval_backend":  ACTIVE_EVAL_BACKEND,
        "bm25_ready":    BM25_READY,
        "device":        DEVICE,
        "models":        list(MODEL_OPTIONS.keys()),
    })
