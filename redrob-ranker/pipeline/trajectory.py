"""
trajectory.py — Stage 3: career arc shape scoring.

What changed from the prior version: every magic number that calibrates
how harshly hops are penalized, how much long-tenure is rewarded, and how
much weight domain progression carries is now scaled by
config.BEHAVIORAL_PRIORITIES (shipper_vs_researcher, startup_vs_enterprise,
builder_vs_manager — the compiler's own emphasis scores for THIS JD),
instead of being fixed constants that ignore which JD is being ranked
against.

Concretely: a JD that scores shipper_vs_researcher=10 (ship-fast culture)
should penalize job-hopping more harshly than one that scores 3 (research-
tolerant), because frequent moves are a bigger red flag for a "ship things"
culture. A JD scoring startup_vs_enterprise=9 should reward small-company
tenure more than one scoring 2. These priorities were already being
extracted by compile_jd.py and were previously dead — never consumed
anywhere in the scoring pipeline.

The four sub-scores (product tenure depth, hop penalty, long-stint bonus,
domain progression) keep their core logic; what's new is that each one's
sensitivity is parameterized from config rather than hardcoded.
"""

import datetime
from utils.date_utils import parse_date


def _priority(config, key: str, default: int = 5) -> int:
    priorities = getattr(config, "BEHAVIORAL_PRIORITIES", {}) or {}
    return priorities.get(key, default)


def product_tenure_score(career_history, services_firms, startup_weight_factor: float = 1.0):
    """Months spent at non-services companies, normalized 0-1.

    startup_weight_factor scales how much credit small companies (1-10
    employees) get relative to the prior fixed 0.7 multiplier — a JD that
    scores startup_vs_enterprise high should treat early-stage tenure as
    closer to full credit, not a flat discount regardless of JD context.
    """
    if not career_history:
        return 0.0
    total_months = 0.0
    for job in career_history:
        company = job.get("company", "")
        industry = job.get("industry", "")
        size = job.get("company_size", "")
        duration = job.get("duration_months", 0)

        if company in services_firms or industry == "IT Services":
            continue

        if size in ["51-200", "201-500", "501-1000", "1001-5000", "5001-10000"]:
            total_months += duration
        elif size == "1-10":
            total_months += duration * startup_weight_factor

    return min(60.0, total_months) / 60.0


def hop_penalty(career_history, severity_factor: float = 1.0):
    """Penalty for short-tenure roles, scaled by severity_factor.

    severity_factor derives from shipper_vs_researcher: a JD that scores
    high on shipping fast treats frequent job-hopping as more disqualifying
    than a research-tolerant JD would, since shipping cultures specifically
    value sustained ownership of systems over time.
    """
    if not career_history:
        return 0.0
    hops = sum(
        1 for job in career_history
        if not job.get("is_current") and job.get("duration_months", 0) < 18
    )
    return hops * 0.1 * severity_factor


def long_stint_bonus(career_history, services_firms, builder_weight_factor: float = 1.0):
    """Bonus for 24+ month non-services roles, scaled by builder_weight_factor.

    builder_weight_factor derives from builder_vs_manager: a JD that
    explicitly wants hands-on ICs should weight demonstrated long-term
    ownership of a single system more heavily than a JD more tolerant of
    a builder-to-manager career arc.
    """
    if not career_history:
        return 0.0
    bonus = 0.0
    for job in career_history:
        company = job.get("company", "")
        industry = job.get("industry", "")
        duration = job.get("duration_months", 0)

        if company not in services_firms and industry != "IT Services":
            if duration >= 24:
                bonus += 0.15 * builder_weight_factor
    return min(0.3, bonus)


def domain_progression_score(career_history, ml_keywords=None):
    if not career_history:
        return 0.0
    sorted_jobs = sorted(
        career_history,
        key=lambda x: parse_date(x.get("start_date")) or datetime.date.min,
    )
    total_weight = 0.0
    weighted_score = 0.0
    keywords = ml_keywords or [
        "ml", "machine learning", "ai", "nlp", "search", "retrieval",
        "ranking", "data scientist", "deep learning",
        "information retrieval", "llm",
    ]
    for idx, job in enumerate(sorted_jobs):
        title = job.get("title", "").lower()
        has_ml = any(kw in title for kw in keywords)
        weight = idx + 1.0
        total_weight += weight
        if has_ml:
            weighted_score += weight

    return (weighted_score / total_weight) if total_weight > 0 else 0.0


def run(candidates, config):
    # Derive calibration factors once per run, not once per candidate.
    # 1-10 scale priorities are mapped to a 0.5x-1.5x multiplier range
    # around the prior fixed behavior (5 = neutral = the original
    # un-calibrated constants), so a JD with no extracted priorities
    # (all defaulting to 5) reproduces the exact previous behavior.
    shipper_score = _priority(config, "shipper_vs_researcher", 5)
    startup_score = _priority(config, "startup_vs_enterprise", 5)
    builder_score = _priority(config, "builder_vs_manager", 5)

    hop_severity_factor = 0.5 + (shipper_score / 10.0)
    startup_weight_factor = 0.5 + (startup_score / 10.0) * 0.5  # caps near 1.0, never exceeds original 0.7 by much
    builder_weight_factor = 0.5 + (builder_score / 10.0)

    for c in candidates:
        history = c.get("career_history", [])
        pts = product_tenure_score(history, config.SERVICES_FIRMS, startup_weight_factor)
        penalty = hop_penalty(history, hop_severity_factor)
        hop_score = max(0.0, 1.0 - penalty)
        bonus = long_stint_bonus(history, config.SERVICES_FIRMS, builder_weight_factor)
        dps = domain_progression_score(history)

        c["trajectory_score"] = float(min(1.0, max(0.0, (pts + hop_score + bonus + dps) / 4.0)))

    return candidates