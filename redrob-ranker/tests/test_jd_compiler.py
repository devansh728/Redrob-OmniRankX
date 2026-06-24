import os
import json
import pytest
from pydantic import ValidationError
from utils.jd_schemas import Pass1Schema, Pass2Schema, Pass3Schema, CompiledConfig
from scripts.compile_jd import calculate_grounded_weights
from config.config_loader import load_runtime_config
from utils.text_utils import extract_raw_text_from_file

def test_pass1_schema_validation():
    data = {
        "business_intent": "Solve corporate problems.",
        "primary_persona": "Software Engineer",
        "anti_personas": ["Title-chaser"],
        "tier1_mandatory_evidence": [],
        "tier2_preferred_evidence": []
    }
    validated = Pass1Schema.model_validate(data)
    assert validated.business_intent == "Solve corporate problems."
    assert validated.primary_persona == "Software Engineer"
    assert validated.anti_personas == ["Title-chaser"]

def test_pass2_semantic_expansions_bounded():
    data = {
        "min_years_experience": 2.0,
        "max_years_experience": 10.0,
        "preferred_cities": [],
        "willing_to_relocate_required": False,
        "max_notice_period_days": 30,
        "max_salary_budget_lpa": 0.0,
        "hard_disqualifiers": [],
        "soft_disqualifiers": [],
        "bounded_concept_expansions": [str(i) for i in range(25)]
    }
    validated = Pass2Schema.model_validate(data)
    assert len(validated.bounded_concept_expansions) == 15
    assert validated.bounded_concept_expansions[-1] == "14"

def test_pass3_schema_clamping():
    data = {
        "startup_vs_enterprise": 15,
        "shipper_vs_researcher": 0,
        "builder_vs_manager": 5,
        "generalist_vs_specialist": 5,
        "jd_ambiguity_score": 11,
        "raw_text_mention_counts": {}
    }
    with pytest.raises(ValidationError):
        Pass3Schema.model_validate(data)

def test_normalize_weights_sums_to_one():
    pass1 = Pass1Schema(
        business_intent="intent",
        primary_persona="persona",
        tier1_mandatory_evidence=[],
        tier2_preferred_evidence=[]
    )
    pass2 = Pass2Schema(
        min_years_experience=0.0,
        max_years_experience=20.0,
        hard_disqualifiers=[{
            "condition_name": "researcher",
            "target_field_path": "career_history.career_role",
            "check_operator": "CONTAINS",
            "rejection_value": "researcher",
            "traceability": {
                "extracted_fact": "fact",
                "verbatim_text_quote": "quote",
                "source_section": "section"
            }
        }],
        soft_disqualifiers=[]
    )
    pass3 = Pass3Schema(
        startup_vs_enterprise=5,
        shipper_vs_researcher=5,
        builder_vs_manager=5,
        generalist_vs_specialist=5,
        jd_ambiguity_score=5,
        raw_text_mention_counts={"skills": 2, "architecture": 1, "location": 1, "notice_period": 1}
    )
    weights = calculate_grounded_weights(pass1, pass2, pass3)
    assert abs(sum(weights.values()) - 1.0) < 0.001
    assert weights["semantic"] > 0
    assert weights["trajectory"] > 0
    assert weights["behavioral"] > 0
    assert weights["logistics"] > 0

def test_config_loader_fallback():
    config = load_runtime_config("nonexistent_file.json")
    assert config.MIN_YEARS_EXPERIENCE == 0.0
    assert config.MAX_YEARS_EXPERIENCE == 20.0
    assert "semantic" in config.SCORE_WEIGHTS
    assert config.STAGE2_TOP_K == 1000

def test_text_extraction_docx():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    jd_path = os.path.join(project_root, "..", "India_runs_data_and_ai_challenge", "job_description.docx")
    assert os.path.exists(jd_path)
    text = extract_raw_text_from_file(jd_path)
    assert len(text) > 0
    assert "redrob" in text.lower() or "ai" in text.lower() or "engineer" in text.lower()
