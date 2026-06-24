"""
config_loader.py — builds a RuntimeConfig object that the 5-stage ranking
pipeline (pruner / semantic / trajectory / behavioral / fusion) reads from.

WHAT'S FIXED IN THIS VERSION, on top of the prior one:

1. JD_SKILL_WEIGHTS was a confirmed dead code path. Nothing populated it
   from the compiled config — it only ever came from a static settings.py
   default (empty dict). This meant semantic.py's skill_assessment_scores
   RRF input silently contributed 0.0 for every candidate, every run, with
   no error or warning. Fixed by deriving JD_SKILL_WEIGHTS from
   tier1/tier2 evidence's requirement_name + evidence_proof_expectations
   text, building a list of (keyword, weight) pairs rather than a fixed
   dict — because skill_assessment_scores keys are CANDIDATE DATA (e.g.
   "FAISS", "Fine-tuning LLMs", "Pinecone"), not something the compiler can
   know in advance. semantic.py now does substring matching against each
   candidate's own actual keys using these weighted keyword phrases.
   Tier-1 (mandatory) evidence is weighted higher than tier-2 (preferred),
   preserving the distinction the compiler computed and the prior loader
   threw away by flattening both tiers into one string.

2. rule_type pass-through: HARD_DISQUALIFIERS / SOFT_DISQUALIFIERS are
   now expected to carry the new rule_type/applies_to/primary_keywords/
   etc fields from the updated compile_jd.py. This loader does NOT
   validate or reshape them — it passes the dicts through as-is, and
   pipeline.rule_engine.evaluate_rule() is solely responsible for
   interpreting rule_type. This keeps the loader simple and means a
   schema change in jd_schemas.py only requires updating rule_engine.py,
   not this file.

3. Backward compatibility: an older compiled config with no rule_type
   field on its disqualifiers still loads without crashing — each rule
   dict simply won't have "rule_type", and rule_engine.evaluate_rule()
   treats a missing rule_type the same as "unresolved" (see
   rule_engine.py's .get("rule_type", "unresolved") default), so old
   configs degrade to "no disqualifiers actively applied" rather than
   raising — visible via the printed unresolved-rule warning in
   pruner.py/behavioral.py, not a silent failure.
"""

import os
import re
import json
from config import settings


# Generic English stopwords stripped when deriving skill keywords from
# evidence text — this list is domain-agnostic on purpose, since the
# evidence text itself varies completely per JD.
_GENERIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "for", "to", "of", "in", "on",
    "at", "by", "from", "is", "are", "be", "this", "that", "experience",
    "production", "real", "users", "deployed", "scale", "meaningful",
    "has", "have", "having", "such", "as", "similar", "or", "etc",
}


def _derive_skill_keywords_from_evidence(evidence_items: list, base_weight: float) -> list:
    """Turns a list of tier1/tier2 evidence dicts into weighted keyword
    phrases usable for substring matching against candidate skill names.

    Input: list of evidence dicts (each with requirement_name and
           evidence_proof_expectations), a base importance weight.
    Output: list of (keyword_phrase, weight) tuples.
    How it works: pulls out multi-word capitalized-or-technical-looking
            phrases and standalone technical tokens from both
            requirement_name and evidence_proof_expectations, filters
            generic stopwords, and pairs each surviving phrase with the
            given base_weight. Phrases are kept short (1-3 words) since
            that is what's most likely to substring-match a candidate's
            actual skill_assessment_scores key (e.g. "Fine-tuning LLMs",
            "Pinecone", "FAISS").
    """
    keywords = []
    for item in evidence_items:
        texts = [item.get("requirement_name", "")] + (item.get("evidence_proof_expectations") or [])
        for text in texts:
            if not text:
                continue
            # Extract candidate phrases: parenthesized lists (common JD
            # pattern: "Pinecone, Weaviate, Qdrant, ...") split on commas;
            # otherwise fall back to extracting individual capitalized or
            # technical-looking tokens.
            paren_groups = re.findall(r"\(([^)]+)\)", text)
            for group in paren_groups:
                for piece in group.split(","):
                    piece = piece.strip().rstrip(".")
                    if piece and piece.lower() not in _GENERIC_STOPWORDS and len(piece) > 1:
                        keywords.append(piece)

            # Also pull standalone technical-looking words (mixed case,
            # all-caps acronyms, or hyphenated terms) from the rest of the
            # text, which catches things like "Strong Python" ->
            # "Python", or "Fine-tuning LLMs" -> kept whole since it's a
            # short multi-word phrase already.
            words = re.findall(r"\b[A-Za-z][A-Za-z0-9+\-]{1,}\b", text)
            for w in words:
                if w.lower() in _GENERIC_STOPWORDS or len(w) <= 2:
                    continue
                if w[0].isupper() or w.isupper():
                    keywords.append(w)

    seen = set()
    out = []
    for kw in keywords:
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((kw, base_weight))
    return out


class RuntimeConfig:
    def __init__(self, config_dict: dict | None = None):
        self.config_dict = config_dict or {}

        # --- Static defaults ---
        self.STAGE2_TOP_K = getattr(settings, "STAGE2_TOP_K", 1000)
        self.FINAL_TOP_N = getattr(settings, "FINAL_TOP_N", 100)
        self.SALARY_BUDGET_MAX_LPA = getattr(settings, "SALARY_BUDGET_MAX_LPA", 0.0)
        self.PREFERRED_CITIES = list(getattr(settings, "PREFERRED_CITIES", []))
        self.SERVICES_FIRMS = set(getattr(settings, "SERVICES_FIRMS", []))
        self.SCORE_WEIGHTS = dict(getattr(settings, "SCORE_WEIGHTS", {
            "semantic": 0.35, "trajectory": 0.25, "behavioral": 0.25, "logistics": 0.15,
        }))
        self.JD_QUERY = getattr(settings, "JD_QUERY", "")
        # JD_SKILL_WEIGHTS is now a list of (keyword, weight) tuples, not a
        # flat dict keyed by exact skill name — see module docstring.
        self.JD_SKILL_WEIGHTS: list[tuple] = list(getattr(settings, "JD_SKILL_WEIGHTS", []))
        self.MIN_YEARS_EXPERIENCE = 0.0
        self.MAX_YEARS_EXPERIENCE = 20.0
        self.MAX_NOTICE_PERIOD_DAYS = 90
        self.WILLING_TO_RELOCATE_REQUIRED = False

        self.HARD_DISQUALIFIERS: list[dict] = []
        self.SOFT_DISQUALIFIERS: list[dict] = []
        self.TIER1_MANDATORY_EVIDENCE: list[dict] = []
        self.TIER2_PREFERRED_EVIDENCE: list[dict] = []
        self.ANTI_PERSONAS: list[str] = []
        self.BUSINESS_INTENT = ""
        self.PRIMARY_PERSONA = ""
        self.BEHAVIORAL_PRIORITIES: dict = {}

        if self.config_dict:
            self._apply_compiled_config(self.config_dict)

    def _apply_compiled_config(self, data: dict) -> None:
        constraints = data.get("constraints", {}) or {}
        semantic_targets = data.get("semantic_targets", {}) or {}
        fusion_weights = data.get("normalized_fusion_weights", {}) or {}

        if fusion_weights:
            self.SCORE_WEIGHTS.update(fusion_weights)

        if "min_years_experience" in constraints:
            self.MIN_YEARS_EXPERIENCE = float(constraints["min_years_experience"])
        if "max_years_experience" in constraints:
            self.MAX_YEARS_EXPERIENCE = float(constraints["max_years_experience"])
        if "max_notice_period_days" in constraints:
            self.MAX_NOTICE_PERIOD_DAYS = int(constraints["max_notice_period_days"])
        if "max_salary_budget_lpa" in constraints:
            raw_salary = float(constraints["max_salary_budget_lpa"])
            if raw_salary > 0.0:
                self.SALARY_BUDGET_MAX_LPA = raw_salary
        if "willing_to_relocate_required" in constraints:
            self.WILLING_TO_RELOCATE_REQUIRED = bool(constraints["willing_to_relocate_required"])

        preferred_cities = constraints.get("preferred_cities") or []
        if preferred_cities:
            self.PREFERRED_CITIES = list({*self.PREFERRED_CITIES, *preferred_cities})

        # Hard/soft disqualifiers are passed through as-is; rule_engine.py
        # is the sole interpreter of rule_type. See module docstring point 2.
        self.HARD_DISQUALIFIERS = constraints.get("hard_disqualifiers") or []
        self.SOFT_DISQUALIFIERS = constraints.get("soft_disqualifiers") or []

        # Services-firm set: prefer named_values from any company_name_match
        # rule (the structured, schema-grounded source), falling back to
        # the legacy free-text scan only if no structured rule provided one.
        derived_firms = set()
        for d in self.HARD_DISQUALIFIERS + self.SOFT_DISQUALIFIERS:
            if d.get("rule_type") == "company_name_match":
                derived_firms.update(d.get("named_values") or [])
        if derived_firms:
            self.SERVICES_FIRMS = self.SERVICES_FIRMS.union(derived_firms)
        else:
            legacy_blacklist = constraints.get("blacklist_firms") or []
            if legacy_blacklist:
                self.SERVICES_FIRMS = self.SERVICES_FIRMS.union(set(legacy_blacklist))

        bounded_expansions = constraints.get("bounded_concept_expansions") or []

        self.BUSINESS_INTENT = semantic_targets.get("business_intent", "")
        self.PRIMARY_PERSONA = semantic_targets.get("primary_persona", "")
        self.ANTI_PERSONAS = semantic_targets.get("anti_personas") or []

        tier1 = semantic_targets.get("tier1_mandatory_evidence") or []
        tier2 = semantic_targets.get("tier2_preferred_evidence") or []
        self.TIER1_MANDATORY_EVIDENCE = tier1
        self.TIER2_PREFERRED_EVIDENCE = tier2

        if tier1 or tier2:
            requirement_names = [e.get("requirement_name", "") for e in tier1 if e.get("requirement_name")]
            requirement_names += [e.get("requirement_name", "") for e in tier2 if e.get("requirement_name")]
            proof_terms = [
                p for e in (tier1 + tier2)
                for p in (e.get("evidence_proof_expectations") or [])
            ]
            mandatory_repeated = requirement_names[: len(tier1)] * 2
            query_parts = [self.PRIMARY_PERSONA] + mandatory_repeated + requirement_names + proof_terms + bounded_expansions
            self.JD_QUERY = " ".join(part for part in query_parts if part).strip()
        elif semantic_targets.get("jd_query"):
            self.JD_QUERY = semantic_targets["jd_query"]

        # JD_SKILL_WEIGHTS: derive weighted keyword phrases from tier1
        # (weight 1.0) and tier2 (weight 0.5) evidence, fixing the
        # previously-dead skill_assessment_scores RRF input. See module
        # docstring point 1.
        #
        # IMPORTANT LIMITATION, stated plainly rather than glossed over:
        # this keyword-substring approach reliably catches named tools
        # written verbatim in the JD (Pinecone, FAISS, Weaviate) because
        # those are extracted from parenthesized lists. It does NOT
        # reliably catch conceptual skill names that the JD describes in
        # prose rather than naming directly (e.g. the JD says
        # "fine-tuning" as a capability, but a candidate's skill is named
        # "Fine-tuning LLMs" — these only coincidentally overlap on the
        # word "LLMs"). bounded_concept_expansions is added as a second
        # source specifically because the compiler populates it with
        # broader conceptual terms beyond verbatim JD text, which closes
        # some of this gap, but this remains an approximate, not exact,
        # mechanism. semantic.py's substring matching is deliberately
        # permissive (partial overlap counts) rather than exact-key-match,
        # which is the right tradeoff for an approximate signal: a missed
        # match contributes 0 (safe default), never a false negative that
        # actively penalizes a candidate.
        tier1_keywords = _derive_skill_keywords_from_evidence(tier1, base_weight=1.0)
        tier2_keywords = _derive_skill_keywords_from_evidence(tier2, base_weight=0.5)
        expansion_keywords = [
            (term, 0.6) for term in bounded_expansions
            if term and term.lower() not in _GENERIC_STOPWORDS
        ]
        self.JD_SKILL_WEIGHTS = tier1_keywords + tier2_keywords + expansion_keywords

        behavioral_priorities = data.get("behavioral_priorities") or {}
        if behavioral_priorities:
            self.BEHAVIORAL_PRIORITIES = behavioral_priorities


def load_runtime_config(config_path: str | None = None) -> RuntimeConfig:
    """Loads the compiled JD config from disk and returns a RuntimeConfig.

    Input: optional explicit path to generated_config.json.
    Output: a populated RuntimeConfig instance.
    How it works: resolves the default path under config/ if none given,
            reads and parses the JSON, and falls back to a config with
            only static settings.py defaults if the file is missing or
            unreadable — never raises, since a missing compiled config
            should degrade gracefully rather than crash the pipeline.
    """
    if config_path is None:
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(proj_root, "config", "generated_config.json")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return RuntimeConfig(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[config_loader] Warning: failed to load {config_path} ({e}). "
                  "Falling back to static settings.py defaults.")
    return RuntimeConfig(None)