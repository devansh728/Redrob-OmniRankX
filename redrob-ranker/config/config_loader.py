"""
config_loader.py — builds a RuntimeConfig object that the 5-stage ranking
pipeline (pruner / semantic / trajectory / behavioral / fusion) reads from.

Changes from the previous version:
- Reads the new fields produced by the fixed compile_jd.py: hard_disqualifiers,
  soft_disqualifiers (with escape hatches), tier1/tier2 evidence, and the
  anti_personas list. The previous loader only read blacklist_firms and
  jd_query — everything else the compiler now extracts was being silently
  ignored downstream even when extraction itself worked correctly.
- Falls back gracefully field-by-field rather than all-or-nothing: if
  generated_config.json exists but is missing a key the compiler used to
  produce in an older format, RuntimeConfig still loads with sane defaults
  for that one key instead of discarding the whole compiled config.
- Builds JD_QUERY from tier1_mandatory_evidence if a compiled config is
  present, instead of relying on a separately-stored jd_query string that
  can fall out of sync with the structured evidence it was supposedly
  derived from.
"""

import os
import json
from config import settings


class RuntimeConfig:
    def __init__(self, config_dict: dict | None = None):
        self.config_dict = config_dict or {}

        # --- Static defaults (used when no compiled config is present, or
        # as a per-field fallback when the compiled config is missing a key) ---
        self.STAGE2_TOP_K = getattr(settings, "STAGE2_TOP_K", 1000)
        self.FINAL_TOP_N = getattr(settings, "FINAL_TOP_N", 100)
        self.SALARY_BUDGET_MAX_LPA = getattr(settings, "SALARY_BUDGET_MAX_LPA", 60.0)
        self.PREFERRED_CITIES = list(getattr(settings, "PREFERRED_CITIES", []))
        self.SERVICES_FIRMS = set(getattr(settings, "SERVICES_FIRMS", []))
        self.SCORE_WEIGHTS = dict(getattr(settings, "SCORE_WEIGHTS", {
            "semantic": 0.35, "trajectory": 0.25, "behavioral": 0.25, "logistics": 0.15,
        }))
        self.JD_QUERY = getattr(settings, "JD_QUERY", "")
        self.JD_SKILL_WEIGHTS = dict(getattr(settings, "JD_SKILL_WEIGHTS", {}))
        self.MIN_YEARS_EXPERIENCE = 0.0
        self.MAX_YEARS_EXPERIENCE = 20.0
        self.MAX_NOTICE_PERIOD_DAYS = 90
        self.WILLING_TO_RELOCATE_REQUIRED = False

        # --- New fields populated only from a compiled config ---
        self.HARD_DISQUALIFIERS: list[dict] = []
        self.SOFT_DISQUALIFIERS: list[dict] = []
        self.TIER1_MANDATORY_EVIDENCE: list[dict] = []
        self.TIER2_PREFERRED_EVIDENCE: list[dict] = []
        self.ANTI_PERSONAS: list[str] = []
        self.BUSINESS_INTENT = ""
        self.PRIMARY_PERSONA = ""

        if self.config_dict:
            self._apply_compiled_config(self.config_dict)

    def _apply_compiled_config(self, data: dict) -> None:
        constraints = data.get("constraints", {}) or {}
        semantic_targets = data.get("semantic_targets", {}) or {}
        fusion_weights = data.get("normalized_fusion_weights", {}) or {}

        if fusion_weights:
            # Only overwrite weights the compiled config actually provides;
            # keep static defaults for anything missing rather than zeroing
            # out a weight the compiler didn't return.
            self.SCORE_WEIGHTS.update(fusion_weights)

        # --- Constraints (Pass 2 output) ---
        if "min_years_experience" in constraints:
            self.MIN_YEARS_EXPERIENCE = float(constraints["min_years_experience"])
        if "max_years_experience" in constraints:
            self.MAX_YEARS_EXPERIENCE = float(constraints["max_years_experience"])
        if "max_notice_period_days" in constraints:
            self.MAX_NOTICE_PERIOD_DAYS = int(constraints["max_notice_period_days"])
        if "max_salary_budget_lpa" in constraints and constraints["max_salary_budget_lpa"]:
            self.SALARY_BUDGET_MAX_LPA = float(constraints["max_salary_budget_lpa"])
        if "willing_to_relocate_required" in constraints:
            self.WILLING_TO_RELOCATE_REQUIRED = bool(constraints["willing_to_relocate_required"])

        preferred_cities = constraints.get("preferred_cities") or []
        if preferred_cities:
            # Union rather than replace — a hand-curated static list in
            # settings.py (e.g. known ML hub cities) should not be erased
            # by a compiled config that only found a subset of cities.
            self.PREFERRED_CITIES = list({*self.PREFERRED_CITIES, *preferred_cities})

        hard_disqualifiers = constraints.get("hard_disqualifiers") or []
        self.HARD_DISQUALIFIERS = hard_disqualifiers

        soft_disqualifiers = constraints.get("soft_disqualifiers") or []
        self.SOFT_DISQUALIFIERS = soft_disqualifiers

        # Services-firm blacklist is derived from soft_disqualifiers whose
        # target_field_path mentions "company" or "career_history" and
        # whose rejection_value-equivalent names known services firms —
        # rather than relying on a separate, possibly-empty blacklist_firms
        # field the way the previous loader did.
        derived_firms = set()
        for d in soft_disqualifiers:
            condition_name = (d.get("condition_name") or "").lower()
            quote = (d.get("traceability", {}) or {}).get("verbatim_text_quote", "").lower()
            if "consulting" in condition_name or "services" in condition_name or "consulting" in quote:
                # Pull out capitalized company-like tokens from the quote
                # as a best-effort fallback if the compiler didn't already
                # break them into a structured list.
                import re
                derived_firms.update(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", quote.title()))
        if derived_firms:
            self.SERVICES_FIRMS = self.SERVICES_FIRMS.union(derived_firms)

        # Backward-compat: older compiled configs used "blacklist_firms"
        legacy_blacklist = constraints.get("blacklist_firms") or []
        if legacy_blacklist:
            self.SERVICES_FIRMS = self.SERVICES_FIRMS.union(set(legacy_blacklist))

        bounded_expansions = constraints.get("bounded_concept_expansions") or []

        # --- Semantic targets (Pass 1 output) ---
        self.BUSINESS_INTENT = semantic_targets.get("business_intent", "")
        self.PRIMARY_PERSONA = semantic_targets.get("primary_persona", "")
        self.ANTI_PERSONAS = semantic_targets.get("anti_personas") or []

        tier1 = semantic_targets.get("tier1_mandatory_evidence") or []
        tier2 = semantic_targets.get("tier2_preferred_evidence") or []
        self.TIER1_MANDATORY_EVIDENCE = tier1
        self.TIER2_PREFERRED_EVIDENCE = tier2

        # Build JD_QUERY from structured evidence rather than trusting a
        # separately stored string. Mandatory evidence is weighted into the
        # query more heavily by simple repetition-free inclusion; legacy
        # configs that only have a flat "jd_query" string still work via
        # the fallback below.
        if tier1 or tier2:
            requirement_names = [e.get("requirement_name", "") for e in tier1 if e.get("requirement_name")]
            requirement_names += [e.get("requirement_name", "") for e in tier2 if e.get("requirement_name")]
            proof_terms = [
                p for e in (tier1 + tier2)
                for p in (e.get("evidence_proof_expectations") or [])
            ]
            query_parts = [self.PRIMARY_PERSONA] + requirement_names + proof_terms + bounded_expansions
            self.JD_QUERY = " ".join(part for part in query_parts if part).strip()
        elif semantic_targets.get("jd_query"):
            self.JD_QUERY = semantic_targets["jd_query"]


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