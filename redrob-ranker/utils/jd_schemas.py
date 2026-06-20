from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator

class Pass1Schema(BaseModel):
    business_intent: str
    primary_persona: str
    must_demonstrate: List[str]

    @field_validator("must_demonstrate")
    @classmethod
    def limit_must_demonstrate(cls, v):
        return v[:5]

class Pass2Schema(BaseModel):
    min_years_experience: float
    max_years_experience: float
    excluded_companies: List[str]
    forbidden_industries: List[str]
    semantic_expansions: List[str]

    @field_validator("semantic_expansions")
    @classmethod
    def limit_expansions(cls, v):
        return v[:15]

class Pass3Schema(BaseModel):
    startup_vs_enterprise: int
    shipper_vs_researcher: int
    builder_vs_manager: int
    technical_depth_weight: int
    behavioral_signals_weight: int
    logistics_importance_weight: int
    jd_ambiguity_score: int

    @field_validator(
        "startup_vs_enterprise",
        "shipper_vs_researcher",
        "builder_vs_manager",
        "technical_depth_weight",
        "behavioral_signals_weight",
        "logistics_importance_weight",
        "jd_ambiguity_score"
    )
    @classmethod
    def clamp_int(cls, v):
        return max(1, min(10, v))

class CompiledConfig(BaseModel):
    meta: Dict[str, Any]
    constraints: Dict[str, Any]
    semantic_targets: Dict[str, Any]
    behavioral_priorities: Dict[str, Any]
    normalized_fusion_weights: Dict[str, float]

    @model_validator(mode="after")
    def validate_weights(self):
        w = self.normalized_fusion_weights
        total = sum(w.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Fusion weights must sum to 1.0, got {total}")
        return self
