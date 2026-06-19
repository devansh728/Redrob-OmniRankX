def run(candidates, config=None):
    for c in candidates:
        c["behavioral_score"] = 0.5
    return candidates
