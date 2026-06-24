"""
fusion.py — Stage 5: weighted score fusion, final ranking, and top-N cut.

WHAT CHANGED:
- logistics_score now enforces config.WILLING_TO_RELOCATE_REQUIRED as a
  genuine hard gate. Previously this field was extracted by the compiler
  and exposed by config_loader, but nothing in fusion.py (or any other
  stage) ever read it — a candidate outside the preferred cities who is
  unwilling to relocate received loc_score=0.0 by coincidence of the
  scoring formula, not because the rule was actually enforced. Now: if
  the JD requires relocation willingness and a candidate is both outside
  the preferred cities AND unwilling to relocate, logistics_score is
  forced to 0.0 explicitly, and a flag is set so fusion can optionally
  treat this as a hard exclusion rather than just a low score (see
  HARD_GATE_ON_RELOCATION below).
- preferred_work_mode scoring is corrected: "flexible" now scores
  strictly highest (it means the candidate is open to anything, which is
  a better fit than a candidate who can ONLY do hybrid), and "onsite"
  scores above "remote" by default, matching the JD's typical preference
  for in-office presence over the previous version's flat 0.5 for both
  "remote" and unrecognized values.
"""

from pipeline.reasoning import build_reasoning

# If True, a candidate failing the WILLING_TO_RELOCATE_REQUIRED hard gate
# is removed from the ranked pool entirely rather than merely scored 0.0
# on logistics. Kept as an explicit, named toggle rather than baked-in
# behavior, since "exclude" vs "heavily penalize" is a real judgment call
# that may differ by JD and is easy to get wrong silently either way.
HARD_GATE_ON_RELOCATION = False


def logistics_score(candidate, config):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    loc = (profile.get("location") or "").lower()
    loc_match = any(city.lower() in loc for city in config.PREFERRED_CITIES)
    willing_to_relocate = signals.get("willing_to_relocate") is True

    if loc_match:
        loc_score = 1.0
    elif willing_to_relocate:
        loc_score = 0.8
    else:
        loc_score = 0.0

    relocation_required = getattr(config, "WILLING_TO_RELOCATE_REQUIRED", False)
    fails_relocation_gate = relocation_required and (not loc_match) and (not willing_to_relocate)
    if fails_relocation_gate:
        loc_score = 0.0

    mode = (signals.get("preferred_work_mode") or "").lower()
    if mode == "flexible":
        work_score = 1.0
    elif mode == "hybrid":
        work_score = 0.95
    elif mode == "onsite":
        work_score = 0.8
    elif mode == "remote":
        work_score = 0.5
    else:
        work_score = 0.5

    score = (loc_score + work_score) / 2.0
    return score, fails_relocation_gate


def run(candidates, config):
    excluded_for_relocation = 0
    survivors = []

    for c in candidates:
        l_score, fails_relocation_gate = logistics_score(c, config)
        c["logistics_score"] = float(l_score)
        c["_fails_relocation_gate"] = fails_relocation_gate

        if fails_relocation_gate:
            excluded_for_relocation += 1

        sem = c.get("semantic_score", 0.0)
        traj = c.get("trajectory_score", 0.0)
        beh = c.get("behavioral_score", 0.0)

        w = config.SCORE_WEIGHTS
        c["final_score"] = float(
            sem * w["semantic"]
            + traj * w["trajectory"]
            + beh * w["behavioral"]
            + l_score * w["logistics"]
        )

        if HARD_GATE_ON_RELOCATION and fails_relocation_gate:
            continue
        survivors.append(c)

    if HARD_GATE_ON_RELOCATION and excluded_for_relocation:
        print(f"[fusion] Hard relocation gate excluded {excluded_for_relocation} candidates.")
    elif excluded_for_relocation:
        print(f"[fusion] {excluded_for_relocation} candidates fail the relocation requirement "
              f"and were scored 0.0 on logistics (not hard-excluded — "
              f"HARD_GATE_ON_RELOCATION is False).")

    sorted_candidates = sorted(
        survivors,
        key=lambda x: (-round(x["final_score"], 4), x["candidate_id"]),
    )

    top_n = min(config.FINAL_TOP_N, len(sorted_candidates))
    results = sorted_candidates[:top_n]

    for idx, c in enumerate(results):
        rank = idx + 1
        c["rank"] = rank
        c["reasoning"] = build_reasoning(c, rank, config=config)
        c.pop("_fails_relocation_gate", None)

    return results