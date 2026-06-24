"""
behavioral.py — Stage 4: redrob_signals composite scoring, plus generic
application of JD-compiled soft disqualifier penalties.

The six core behavioral sub-signals (recency, response rate, github,
interview completion, notice period, salary fit) are unchanged — they are
JD-agnostic by nature; every JD cares about whether a candidate is
reachable and available.

What changed: soft disqualifier penalties are now applied via
pipeline.rule_engine.evaluate_rule()/evaluate_escape_hatch(), dispatching
on each rule's rule_type, instead of matching on condition_name strings.
A rule whose condition is met gets its penalty_weight applied as a
multiplicative discount on the behavioral composite — UNLESS its escape
hatch condition is also met, in which case the penalty is waived
entirely, mirroring the JD's own stated nuance (e.g. "this isn't a fit,
UNLESS you have prior product experience").
"""

import math
from utils.date_utils import parse_date, REFERENCE_TODAY
from pipeline.rule_engine import evaluate_rule, evaluate_escape_hatch, count_unresolved_rules


def recency_score(last_active_date):
    d = parse_date(last_active_date)
    if not d:
        return 0.0
    days = (REFERENCE_TODAY - d).days
    return math.exp(-max(0, days) / 90.0)


def github_score(raw):
    if raw == -1 or raw is None:
        return 0.5
    return min(1.0, max(0.0, raw / 100.0))


def notice_score(notice_period_days):
    if notice_period_days is None:
        return 0.6
    if notice_period_days <= 30:
        return 1.0
    elif notice_period_days <= 60:
        return 0.8
    else:
        return 0.6


def salary_fit(expected_salary, budget_max):
    if not expected_salary:
        return 1.0
    max_val = expected_salary.get("max")
    if max_val is None:
        return 1.0
    if budget_max <= 0.0:
        # No budget figure was extracted from the JD at all — a missing
        # budget should never silently penalize every candidate via a
        # 0.0 * 1.25 comparison, so treat "no stated budget" as neutral.
        return 1.0
    if max_val > budget_max * 1.25:
        return 0.5
    return 1.0


def _neutral_if_sentinel(value, sentinel=-1):
    if value is None or value == sentinel:
        return 0.5
    return value


def score_behavioral_composite(signals: dict, config) -> float:
    """Computes the six-signal behavioral composite for one candidate.

    Input: a candidate's redrob_signals dict, the runtime config.
    Output: float 0.0-1.0.
    How it works: weighted sum of recency/response/github/interview/
            notice/salary, then an open_to_work multiplier, then clamp.
    """
    r_score = recency_score(signals.get("last_active_date"))
    resp_rate = _neutral_if_sentinel(signals.get("recruiter_response_rate"))
    git_score = github_score(signals.get("github_activity_score"))
    int_rate = _neutral_if_sentinel(signals.get("interview_completion_rate"))
    n_score = notice_score(signals.get("notice_period_days"))
    sal_score = salary_fit(signals.get("expected_salary_range_inr_lpa"), config.SALARY_BUDGET_MAX_LPA)

    comp = (
        r_score * 0.25
        + resp_rate * 0.25
        + git_score * 0.15
        + int_rate * 0.15
        + n_score * 0.10
        + sal_score * 0.10
    )

    if signals.get("open_to_work_flag") is True:
        comp = comp * 1.2

    return min(1.0, max(0.0, comp))


def apply_soft_disqualifier_penalties(candidate: dict, behavioral_score: float, config) -> float:
    """Applies every compiled soft disqualifier's penalty generically.

    Input: a candidate dict, the candidate's current behavioral_score,
           the runtime config (carrying SOFT_DISQUALIFIERS).
    Output: the behavioral_score after multiplicative penalty discounts.
    How it works: for each soft disqualifier rule, dispatches via
            rule_engine.evaluate_rule on rule_type (never condition_name);
            if the rule matches AND its escape hatch does NOT apply,
            multiplies the running score by (1 - penalty_weight). Rules
            with rule_type=unresolved are skipped entirely — see
            count_unresolved_rules usage in run() for the visibility this
            produces.
    """
    soft_disqualifiers = getattr(config, "SOFT_DISQUALIFIERS", [])
    score = behavioral_score
    for rule in soft_disqualifiers:
        if rule.get("rule_type", "unresolved") == "unresolved":
            continue
        penalty = float(rule.get("penalty_weight", 0.0))
        if penalty <= 0.0:
            continue

        matched = evaluate_rule(rule, candidate)
        if matched is not True:
            continue

        if evaluate_escape_hatch(rule, candidate):
            continue

        score = score * (1.0 - penalty)

    return score


def run(candidates, config):
    soft_disqualifiers = getattr(config, "SOFT_DISQUALIFIERS", [])
    unresolved_count = count_unresolved_rules(soft_disqualifiers)
    if unresolved_count:
        print(f"[behavioral] WARNING: {unresolved_count} soft disqualifier(s) have "
              f"an unresolved rule_type and will NOT be applied as a penalty.")

    for c in candidates:
        signals = c.get("redrob_signals", {})
        base_score = score_behavioral_composite(signals, config)
        final_score = apply_soft_disqualifier_penalties(c, base_score, config)
        c["behavioral_score"] = float(min(1.0, max(0.0, final_score)))

    return candidates