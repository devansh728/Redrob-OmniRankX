import os
import json
import pytest
from utils.jd_schemas import Pass1Schema, Pass2Schema, Pass3Schema, CompiledConfig
from scripts.compile_jd import normalize_weights, build_output, apply_ambiguity_damping
from config.config_loader import load_runtime_config
from utils.text_utils import extract_raw_text_from_file

def test_pass1_schema_validation():
    data = {
        "business_intent": "Solve corporate problems.",
        "primary_persona": "Software Engineer",
        "must_demonstrate": ["a", "b", "c", "d", "e", "f", "g"]
    }
    validated = Pass1Schema.model_validate(data)
    assert len(validated.must_demonstrate) == 5
    assert validated.must_demonstrate == ["a", "b", "c", "d", "e"]

def test_pass2_semantic_expansions_bounded():
    data = {
        "min_years_experience": 2.0,
        "max_years_experience": 10.0,
        "excluded_companies": [],
        "forbidden_industries": [],
        "semantic_expansions": [str(i) for i in range(25)]
    }
    validated = Pass2Schema.model_validate(data)
    assert len(validated.semantic_expansions) == 15
    assert validated.semantic_expansions[-1] == "14"

def test_pass3_schema_clamping():
    data = {
        "startup_vs_enterprise": 15,
        "shipper_vs_researcher": 0,
        "builder_vs_manager": 5,
        "technical_depth_weight": 8,
        "behavioral_signals_weight": 4,
        "logistics_importance_weight": 2,
        "jd_ambiguity_score": 11
    }
    validated = Pass3Schema.model_validate(data)
    assert validated.startup_vs_enterprise == 10
    assert validated.shipper_vs_researcher == 1
    assert validated.builder_vs_manager == 5
    assert validated.jd_ambiguity_score == 10

def test_normalize_weights_sums_to_one():
    pass3 = Pass3Schema(
        startup_vs_enterprise=5,
        shipper_vs_researcher=5,
        builder_vs_manager=5,
        technical_depth_weight=8,
        behavioral_signals_weight=4,
        logistics_importance_weight=2,
        jd_ambiguity_score=5
    )
    weights = normalize_weights(pass3)
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
