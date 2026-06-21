from typing import List, Optional, Dict
from pydantic import BaseModel, Field

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
    anti_personas: List[str]
    tier1_mandatory_evidence: List[PositiveEvidenceRequirement]
    tier2_preferred_evidence: List[PositiveEvidenceRequirement]

class Pass2Schema(BaseModel):
    min_years_experience: float
    max_years_experience: float
    preferred_cities: List[str]
    willing_to_relocate_required: bool
    max_notice_period_days: int
    max_salary_budget_lpa: float
    hard_disqualifiers: List[HardDisqualifier]
    soft_disqualifiers: List[SoftDisqualifier]
    bounded_concept_expansions: List[str] = Field(..., max_items=15)

class Pass3Schema(BaseModel):
    startup_vs_enterprise: int = Field(..., ge=1, le=10)
    shipper_vs_researcher: int = Field(..., ge=1, le=10)
    builder_vs_manager: int = Field(..., ge=1, le=10)
    generalist_vs_specialist: int = Field(..., ge=1, le=10)
    jd_ambiguity_score: int = Field(..., ge=1, le=10)
    raw_text_mention_counts: Dict[str, int]