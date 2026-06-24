"""
Pydantic schemas for the three-pass JD compiler.

Design notes (read before editing):
- Every extracted fact carries a TraceableQuote so a human can audit, in
  under a minute, whether the model actually read the JD or invented
  something. This is not optional decoration — it is the cheapest
  hallucination detector available, and it is what let us prove that the
  previous run had copied its few-shot example verbatim.
- bounded_concept_expansions no longer hard-fails on >15 items. A model
  that returns 18 good concepts should not crash the whole pass; we
  truncate to the top 15 in code after parsing instead of rejecting the
  response outright. Schema validation should be permissive on cardinality
  and strict on shape.
- Each schema is also chunk-mergeable: the compiler runs one pass per JD
  chunk and merges partial results with merge_pass1 / merge_pass2 /
  merge_pass3 below, rather than reassembling chunks into one giant prompt.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class TraceableQuote(BaseModel):
    extracted_fact: str
    verbatim_text_quote: str
    source_section: str


class HardDisqualifier(BaseModel):
    condition_name: str
    target_field_path: str
    check_operator: str
    rejection_value: str
    traceability: TraceableQuote


class SoftDisqualifier(BaseModel):
    condition_name: str
    target_field_path: str
    penalty_weight: float = Field(..., ge=0.0, le=1.0)
    escape_clause_condition: Optional[str] = None
    has_escape_hatch: bool
    traceability: TraceableQuote


class PositiveEvidenceRequirement(BaseModel):
    requirement_name: str
    evidence_proof_expectations: List[str]
    is_mandatory_tier1: bool
    traceability: TraceableQuote


class Pass1Schema(BaseModel):
    business_intent: str
    primary_persona: str
    anti_personas: List[str] = Field(default_factory=list)
    tier1_mandatory_evidence: List[PositiveEvidenceRequirement] = Field(default_factory=list)
    tier2_preferred_evidence: List[PositiveEvidenceRequirement] = Field(default_factory=list)


class Pass2Schema(BaseModel):
    min_years_experience: float = 0.0
    max_years_experience: float = 20.0
    preferred_cities: List[str] = Field(default_factory=list)
    willing_to_relocate_required: bool = False
    max_notice_period_days: int = 90
    max_salary_budget_lpa: float = 0.0
    hard_disqualifiers: List[HardDisqualifier] = Field(default_factory=list)
    soft_disqualifiers: List[SoftDisqualifier] = Field(default_factory=list)
    bounded_concept_expansions: List[str] = Field(default_factory=list)

    @field_validator("bounded_concept_expansions")
    @classmethod
    def cap_expansions(cls, v: List[str]) -> List[str]:
        # Truncate rather than reject. A model that found 20 good concepts
        # should not fail the whole pass over a cardinality limit.
        return v[:15]


class Pass3Schema(BaseModel):
    startup_vs_enterprise: int = Field(default=5, ge=1, le=10)
    shipper_vs_researcher: int = Field(default=5, ge=1, le=10)
    builder_vs_manager: int = Field(default=5, ge=1, le=10)
    generalist_vs_specialist: int = Field(default=5, ge=1, le=10)
    jd_ambiguity_score: int = Field(default=5, ge=1, le=10)
    raw_text_mention_counts: Dict[str, int] = Field(default_factory=dict)


class CompiledConfig(BaseModel):
    meta: Dict[str, Any]
    constraints: Dict[str, Any]
    semantic_targets: Dict[str, Any]
    behavioral_priorities: Dict[str, Any]
    normalized_fusion_weights: Dict[str, float]


# ---------------------------------------------------------------------------
# Chunk-merge helpers
#
# Each pass now runs once per JD chunk. These functions combine the partial
# Pass*Schema objects from each chunk into one final object per pass. The
# merge strategy is deliberately conservative: union lists and dedupe by a
# normalized key, take the tightest numeric bounds, and average integer
# scores rather than letting the last chunk silently overwrite the rest.
# ---------------------------------------------------------------------------

def _dedupe_by_key(items: list, key_fn) -> list:
    seen = set()
    out = []
    for item in items:
        k = key_fn(item)
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def merge_pass1(partials: List[Pass1Schema]) -> Pass1Schema:
    if not partials:
        return Pass1Schema(business_intent="", primary_persona="")

    # business_intent / primary_persona: take the longest non-empty value.
    # The chunk that contains the JD's opening summary almost always
    # produces the most complete answer for these two fields; shorter
    # chunks (e.g. a pure bullet-list chunk) tend to produce thin or
    # empty values here, so "longest wins" is a reasonable, cheap proxy.
    business_intent = max((p.business_intent for p in partials), key=len, default="")
    primary_persona = max((p.primary_persona for p in partials), key=len, default="")

    anti_personas = _dedupe_by_key(
        [a for p in partials for a in p.anti_personas],
        key_fn=lambda a: a.strip().lower(),
    )
    tier1 = _dedupe_by_key(
        [e for p in partials for e in p.tier1_mandatory_evidence],
        key_fn=lambda e: e.requirement_name.strip().lower(),
    )
    tier2 = _dedupe_by_key(
        [e for p in partials for e in p.tier2_preferred_evidence],
        key_fn=lambda e: e.requirement_name.strip().lower(),
    )

    return Pass1Schema(
        business_intent=business_intent,
        primary_persona=primary_persona,
        anti_personas=anti_personas,
        tier1_mandatory_evidence=tier1,
        tier2_preferred_evidence=tier2,
    )


def merge_pass2(partials: List[Pass2Schema]) -> Pass2Schema:
    if not partials:
        return Pass2Schema()

    # Numeric bounds: take the tightest (most informative) non-default
    # value seen across chunks rather than blindly averaging or taking the
    # last chunk. A chunk that never mentions years of experience will
    # return the schema default (0.0 / 20.0); we want a chunk that
    # actually states "5-9 years" to win over a chunk that said nothing.
    def pick_min_years(ps):
        vals = [p.min_years_experience for p in ps if p.min_years_experience > 0.0]
        return max(vals) if vals else 0.0

    def pick_max_years(ps):
        vals = [p.max_years_experience for p in ps if p.max_years_experience < 20.0]
        return min(vals) if vals else 20.0

    def pick_notice(ps):
        vals = [p.max_notice_period_days for p in ps if p.max_notice_period_days != 90]
        return min(vals) if vals else 90

    def pick_salary(ps):
        vals = [p.max_salary_budget_lpa for p in ps if p.max_salary_budget_lpa > 0.0]
        return max(vals) if vals else 0.0

    cities = _dedupe_by_key(
        [c for p in partials for c in p.preferred_cities],
        key_fn=lambda c: c.strip().lower(),
    )
    hard = _dedupe_by_key(
        [d for p in partials for d in p.hard_disqualifiers],
        key_fn=lambda d: d.condition_name.strip().lower(),
    )
    soft = _dedupe_by_key(
        [d for p in partials for d in p.soft_disqualifiers],
        key_fn=lambda d: d.condition_name.strip().lower(),
    )
    expansions = _dedupe_by_key(
        [e for p in partials for e in p.bounded_concept_expansions],
        key_fn=lambda e: e.strip().lower(),
    )[:15]

    return Pass2Schema(
        min_years_experience=pick_min_years(partials),
        max_years_experience=pick_max_years(partials),
        preferred_cities=cities,
        willing_to_relocate_required=any(p.willing_to_relocate_required for p in partials),
        max_notice_period_days=pick_notice(partials),
        max_salary_budget_lpa=pick_salary(partials),
        hard_disqualifiers=hard,
        soft_disqualifiers=soft,
        bounded_concept_expansions=expansions,
    )


def merge_pass3(partials: List[Pass3Schema]) -> Pass3Schema:
    if not partials:
        return Pass3Schema()

    def avg_int(field_name: str) -> int:
        vals = [getattr(p, field_name) for p in partials]
        return round(sum(vals) / len(vals))

    merged_counts: Dict[str, int] = {}
    for p in partials:
        for k, v in p.raw_text_mention_counts.items():
            merged_counts[k] = merged_counts.get(k, 0) + v

    return Pass3Schema(
        startup_vs_enterprise=avg_int("startup_vs_enterprise"),
        shipper_vs_researcher=avg_int("shipper_vs_researcher"),
        builder_vs_manager=avg_int("builder_vs_manager"),
        generalist_vs_specialist=avg_int("generalist_vs_specialist"),
        jd_ambiguity_score=avg_int("jd_ambiguity_score"),
        raw_text_mention_counts=merged_counts,
    )