"""
rule_engine.py — generic, JD-agnostic executor for compiled disqualifier
rules.

WHY THIS FILE EXISTS:
Earlier versions of pruner.py and behavioral.py had Python branches that
matched on condition_name strings ("Title-chasers", "Researcher",
"Consulting firms", ...). That looks config-driven but isn't: condition_name
is free text the compiler invents fresh per JD run. A different JD, or even
a re-run of the SAME JD through the compiler, can produce different
spelling/casing/wording for conceptually identical rules, and every
existing elif branch would then silently never fire — disqualification
would go to zero with no error, no warning, nothing.

THE FIX: every HardDisqualifier/SoftDisqualifier now carries a `rule_type`
field constrained to a small fixed enum (utils.jd_schemas.RuleType). This
module has exactly one execution function per rule_type value. Pipeline
stages call evaluate_rule(rule, candidate) and never branch on
condition_name themselves. Adding support for a new JD's rules means the
compiler classifies them into the existing taxonomy — it does NOT mean
writing new Python.

If a rule's rule_type is UNRESOLVED (the compiler genuinely could not
classify it, or the config was produced by an older compiler version with
no rule_type at all), evaluate_rule returns None rather than False. None
means "could not evaluate" and is handled distinctly from False ("evaluated,
did not match") by every caller — an unresolved rule is logged and
surfaced for human review, never silently treated as "no match" (which
would silently let a real disqualifier through) or "match" (which would
silently reject candidates based on a rule nobody verified).
"""

from typing import Optional


def _get_career_history(candidate: dict) -> list:
    return candidate.get("career_history") or []


def _role_matches_any_all(roles: list, predicate, applies_to: str) -> Optional[bool]:
    """Applies a per-role boolean predicate across a role list per the
    rule's quantifier.

    Input: list of role dicts, a predicate function (role -> bool),
           the applies_to quantifier string.
    Output: True/False, or None if applies_to is not_applicable/unrecognized
            or roles is empty (nothing to quantify over).
    How it works: any_role -> True if predicate holds for at least one
            role; all_roles -> True only if predicate holds for every
            role; current_role_only -> evaluates predicate on the single
            role where is_current is True, if one exists.
    """
    if not roles:
        return None
    if applies_to == "any_role":
        return any(predicate(r) for r in roles)
    if applies_to == "all_roles":
        return all(predicate(r) for r in roles)
    if applies_to == "current_role_only":
        current_roles = [r for r in roles if r.get("is_current")]
        if not current_roles:
            return None
        return predicate(current_roles[0])
    return None


def _text_contains_any(text: str, keywords: list) -> bool:
    if not keywords:
        return False
    text_lower = (text or "").lower()
    return any(kw.lower() in text_lower for kw in keywords if kw)


def _eval_career_industry_match(rule: dict, candidate: dict) -> Optional[bool]:
    roles = _get_career_history(candidate)
    named_values = [v.lower() for v in (rule.get("named_values") or [])]
    if not named_values:
        return None
    return _role_matches_any_all(
        roles,
        lambda r: (r.get("industry") or "").lower() in named_values,
        rule.get("applies_to", "not_applicable"),
    )


def _eval_career_title_keyword(rule: dict, candidate: dict) -> Optional[bool]:
    roles = _get_career_history(candidate)
    keywords = rule.get("primary_keywords") or []
    if not keywords:
        return None
    return _role_matches_any_all(
        roles,
        lambda r: _text_contains_any(r.get("title", ""), keywords),
        rule.get("applies_to", "not_applicable"),
    )


def _eval_career_text_keyword(rule: dict, candidate: dict) -> Optional[bool]:
    keywords = rule.get("primary_keywords") or []
    if not keywords:
        return None
    text = candidate.get("precomputed_career_text", "")
    return _text_contains_any(text, keywords)


def _eval_company_name_match(rule: dict, candidate: dict) -> Optional[bool]:
    roles = _get_career_history(candidate)
    named_values = [v.lower() for v in (rule.get("named_values") or [])]
    if not named_values:
        return None
    return _role_matches_any_all(
        roles,
        lambda r: (r.get("company") or "").lower() in named_values,
        rule.get("applies_to", "not_applicable"),
    )


def _eval_tenure_pattern(rule: dict, candidate: dict) -> Optional[bool]:
    """Detects a 'frequent short stints' pattern generically.

    Uses numeric_threshold as the max acceptable average tenure in
    months (falls back to 18 if not provided), and flags True when there
    are 3+ roles and the average tenure is below that threshold — the
    same generic shape as the JD's title-chasing pattern, but without
    hardcoding "Title-chasers" anywhere.
    """
    roles = _get_career_history(candidate)
    if len(roles) < 3:
        return False
    durations = [r.get("duration_months", 0) for r in roles]
    avg_tenure = sum(durations) / len(durations) if durations else 0.0
    threshold = rule.get("numeric_threshold") or 18.0
    return avg_tenure < threshold


def _eval_current_title_keyword(rule: dict, candidate: dict) -> Optional[bool]:
    keywords = rule.get("primary_keywords") or []
    if not keywords:
        return None
    current_title = candidate.get("profile", {}).get("current_title", "")
    return _text_contains_any(current_title, keywords)


def _eval_platform_activity(rule: dict, candidate: dict) -> Optional[bool]:
    """Generic inactivity/non-engagement check.

    Flags True when the candidate shows no positive engagement signal:
    not open to work AND no applications submitted in the last 30 days.
    This generalizes the "is this candidate actually reachable" concept
    without hardcoding any specific JD's condition name.
    """
    signals = candidate.get("redrob_signals", {})
    open_to_work = signals.get("open_to_work_flag") is True
    apps = signals.get("applications_submitted_30d", 0) or 0
    return (not open_to_work) and apps == 0


def _eval_location_relocation(rule: dict, candidate: dict) -> Optional[bool]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    country = (profile.get("country") or "").lower()
    willing = signals.get("willing_to_relocate")
    is_outside_target = country and country != "india"
    is_unwilling = willing is False
    return bool(is_outside_target and is_unwilling)


def _eval_skill_or_domain_balance(rule: dict, candidate: dict) -> Optional[bool]:
    primary = rule.get("primary_keywords") or []
    corroborating = rule.get("corroborating_keywords") or []
    if not primary or not corroborating:
        return None
    text = candidate.get("precomputed_career_text", "")
    has_primary = _text_contains_any(text, primary)
    has_corroborating = _text_contains_any(text, corroborating)
    return has_primary and not has_corroborating


def _eval_title_description_consistency(rule: dict, candidate: dict) -> Optional[bool]:
    """Per-role check: does THIS role's title imply a function that this
    SAME role's own description fails to support?

    This is the generic version of the JD's explicitly-named honeypot
    pattern ("AI keywords in skills, but title is Marketing Manager").
    It checks consistency within a single role entry, not across the
    candidate's aggregate text — catching a planted mismatch that a
    flat keyword search over all career text combined would miss,
    because the aggregate text might still contain genuine AI keywords
    from OTHER roles even when one specific role's title/description
    pair is inconsistent.

    Uses primary_keywords as "function-implying" terms expected in a
    technical title (engineer, scientist, researcher, developer) and
    flags a role where the title contains one of these but the
    description shows no corroborating technical content at all.
    """
    roles = _get_career_history(candidate)
    if not roles:
        return None
    technical_title_terms = rule.get("primary_keywords") or [
        "engineer", "scientist", "developer", "researcher", "architect",
    ]
    technical_content_terms = rule.get("corroborating_keywords") or [
        "model", "system", "pipeline", "algorithm", "code", "production",
        "deployed", "built", "designed", "implemented", "data", "ml", "ai",
    ]
    for role in roles:
        title = (role.get("title") or "").lower()
        description = (role.get("description") or "").lower()
        title_is_technical = any(t in title for t in technical_title_terms)
        description_has_no_technical_content = not any(
            t in description for t in technical_content_terms
        )
        if title_is_technical and description_has_no_technical_content and description:
            return True
    return False


_EVALUATORS = {
    "career_industry_match": _eval_career_industry_match,
    "career_title_keyword": _eval_career_title_keyword,
    "career_text_keyword": _eval_career_text_keyword,
    "company_name_match": _eval_company_name_match,
    "tenure_pattern": _eval_tenure_pattern,
    "current_title_keyword": _eval_current_title_keyword,
    "platform_activity": _eval_platform_activity,
    "location_relocation": _eval_location_relocation,
    "skill_or_domain_balance": _eval_skill_or_domain_balance,
    "title_description_consistency": _eval_title_description_consistency,
}


def evaluate_rule(rule: dict, candidate: dict) -> Optional[bool]:
    """Evaluates one compiled disqualifier rule against one candidate.

    Input: a rule dict (HardDisqualifier or SoftDisqualifier, model_dump'd
           or loaded from JSON), a candidate dict.
    Output: True (rule condition is met), False (not met), or None
            (could not be evaluated — rule_type is unresolved/unrecognized,
            or the rule is missing data it needs).
    How it works: dispatches purely on rule["rule_type"] to one of the
            _eval_* functions above. Never inspects condition_name.
    """
    rule_type = rule.get("rule_type", "unresolved")
    evaluator = _EVALUATORS.get(rule_type)
    if evaluator is None:
        return None
    return evaluator(rule, candidate)


def evaluate_escape_hatch(rule: dict, candidate: dict) -> bool:
    """Checks whether a soft disqualifier's escape clause applies.

    Input: a SoftDisqualifier rule dict, a candidate dict.
    Output: True if the escape hatch condition is satisfied (penalty
            should be waived), False otherwise (including when there is
            no escape hatch, or it can't be evaluated).
    How it works: builds a synthetic rule dict from escape_rule_type and
            escape_keywords, then reuses evaluate_rule's dispatch — the
            escape clause is itself just another rule, evaluated the same
            generic way as the primary condition.
    """
    if not rule.get("has_escape_hatch"):
        return False
    escape_rule_type = rule.get("escape_rule_type", "unresolved")
    escape_keywords = rule.get("escape_keywords") or []
    if escape_rule_type == "unresolved" or not escape_keywords:
        return False
    synthetic_rule = {
        "rule_type": escape_rule_type,
        "primary_keywords": escape_keywords,
        "applies_to": rule.get("applies_to", "any_role"),
    }
    result = evaluate_rule(synthetic_rule, candidate)
    return result is True


def count_unresolved_rules(rules: list) -> int:
    """Counts how many rules in a list have rule_type unresolved.

    Input: a list of rule dicts.
    Output: integer count.
    How it works: used by the orchestrator to log a visible warning when
            the compiler produced rules it could not classify, so this
            never fails silently — an UNRESOLVED rule is excluded from
            both pruning and penalty application, which means it has NO
            effect on ranking until a human reviews and either fixes the
            compiler's classification or manually assigns a rule_type.
    """
    return sum(1 for r in rules if r.get("rule_type", "unresolved") == "unresolved")