from pipeline.reasoning import build_reasoning

def logistics_score(candidate, config):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    
    loc = profile.get("location", "").lower()
    loc_match = any(city.lower() in loc for city in config.PREFERRED_CITIES)
    if loc_match:
        loc_score = 1.0
    elif signals.get("willing_to_relocate") is True:
        loc_score = 0.8
    else:
        loc_score = 0.0
        
    mode = signals.get("preferred_work_mode", "").lower()
    if mode in ["flexible", "hybrid"]:
        work_score = 1.0
    elif mode == "onsite":
        work_score = 0.8
    elif mode == "remote":
        work_score = 0.5
    else:
        work_score = 0.5
        
    return (loc_score + work_score) / 2.0

def run(candidates, config):
    for c in candidates:
        l_score = logistics_score(c, config)
        c["logistics_score"] = float(l_score)
        
        sem = c.get("semantic_score", 0.0)
        traj = c.get("trajectory_score", 0.0)
        beh = c.get("behavioral_score", 0.0)
        
        w = config.SCORE_WEIGHTS
        c["final_score"] = float(
            sem * w["semantic"] +
            traj * w["trajectory"] +
            beh * w["behavioral"] +
            l_score * w["logistics"]
        )

    sorted_candidates = sorted(
        candidates,
        key=lambda x: (-round(x["final_score"], 4), x["candidate_id"])
    )

    top_n = min(config.FINAL_TOP_N, len(sorted_candidates))
    results = sorted_candidates[:top_n]

    for idx, c in enumerate(results):
        rank = idx + 1
        c["rank"] = rank
        c["reasoning"] = build_reasoning(c, rank)

    return results
