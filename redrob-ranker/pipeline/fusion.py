from pipeline.reasoning import build_reasoning

def run(candidates, config=None):
    results = []
    for i, c in enumerate(candidates):
        c["logistics_score"] = 0.5
        c["final_score"] = 0.5
        c["rank"] = i + 1
        c["reasoning"] = build_reasoning(c, i + 1)
        results.append(c)
    return results[:100]
