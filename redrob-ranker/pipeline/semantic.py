def run(candidates, embedder=None, config=None):
    for c in candidates:
        c["semantic_score"] = 0.5
    return candidates
