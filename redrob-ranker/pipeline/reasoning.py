"""
reasoning.py — generates a 1-2 sentence human-readable justification for
each top-N candidate's rank.

WHAT CHANGED:
The previous version produced generic sentences ("Strong semantic match
for ML/IR role") with no connection to WHICH JD requirement was actually
satisfied. Now, when config carries TIER1_MANDATORY_EVIDENCE, the
reasoning generator checks the candidate's career text against each
evidence item's matching_keywords (or falls back to keywords derived from
requirement_name/evidence_proof_expectations) and names the SPECIFIC
requirement satisfied, e.g. "satisfies tier-1 requirement: production
experience with vector databases" — directly serving the explainability
goal, using data the compiler already produces rather than inventing new
extraction work.

The behavioral-priority-driven trajectory threshold adjustment from the
prior version is kept, since it's a genuine, correct use of
BEHAVIORAL_PRIORITIES.
"""

import re
from utils.date_utils import parse_date, REFERENCE_TODAY


def _evidence_keywords(evidence_item: dict) -> list:
    explicit = evidence_item.get("matching_keywords") or []
    if explicit:
        return explicit
    text = evidence_item.get("requirement_name", "") + " " + " ".join(
        evidence_item.get("evidence_proof_expectations") or []
    )
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9+\-]{2,}\b", text)
    return [w for w in words if w[0].isupper() or w.isupper()]


def _find_satisfied_evidence(candidate: dict, evidence_items: list, max_items: int = 2) -> list:
    text = (candidate.get("precomputed_career_text") or "").lower()
    satisfied = []
    for item in evidence_items:
        keywords = _evidence_keywords(item)
        if not keywords:
            continue
        if any(kw.lower() in text for kw in keywords):
            satisfied.append(item.get("requirement_name", ""))
        if len(satisfied) >= max_items:
            break
    return satisfied


def build_reasoning(candidate, rank, config=None):
    signals = candidate.get("redrob_signals", {})
    reasons = []

    tier1_evidence = getattr(config, "TIER1_MANDATORY_EVIDENCE", []) if config else []
    satisfied_requirements = _find_satisfied_evidence(candidate, tier1_evidence)
    if satisfied_requirements:
        joined = " and ".join(satisfied_requirements[:2])
        reasons.append(f"Satisfies tier-1 requirement(s): {joined}.")
    else:
        sem = candidate.get("semantic_score", 0.0)
        persona = getattr(config, "PRIMARY_PERSONA", "") if config else ""
        if sem > 0.75:
            if persona:
                reasons.append(f"Strong semantic match for {persona}.")
            else:
                reasons.append("Strong semantic match for the role's core requirements.")

    shipper = 5
    if config and hasattr(config, "BEHAVIORAL_PRIORITIES"):
        shipper = config.BEHAVIORAL_PRIORITIES.get("shipper_vs_researcher", 5)

    traj_threshold = 0.7
    if shipper > 5:
        traj_threshold = 0.65

    traj = candidate.get("trajectory_score", 0.0)
    if traj > traj_threshold:
        reasons.append("Stable, product-focused career trajectory.")

    last_act_str = signals.get("last_active_date")
    last_act = parse_date(last_act_str)
    is_active_14 = False
    if last_act:
        days = (REFERENCE_TODAY - last_act).days
        if days <= 14:
            is_active_14 = True

    if signals.get("open_to_work_flag") is True and is_active_14:
        reasons.append("Actively searching and recently engaged on the platform.")

    notice = signals.get("notice_period_days")
    if notice is not None and notice <= 30:
        reasons.append("Available within 30 days.")

    git_raw = signals.get("github_activity_score")
    if git_raw is not None and git_raw > 70.0:
        reasons.append("Strong GitHub activity.")

    if not reasons:
        return (f"Ranks #{rank} by composite scoring across semantic fit, "
                 f"career trajectory, and behavioral availability signals.")

    return " ".join(reasons[:2])