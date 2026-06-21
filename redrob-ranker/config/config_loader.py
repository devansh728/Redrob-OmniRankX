import os
import json
from config import settings

class RuntimeConfig:
    def __init__(self, config_dict=None):
        self.config_dict = config_dict or {}
        
        # Core Platform Defaults
        self.STAGE2_TOP_K = getattr(settings, "STAGE2_TOP_K", 1000)
        self.FINAL_TOP_N = getattr(settings, "FINAL_TOP_N", 100)
        
        # Logistical & Structural Gating Values
        self.SALARY_BUDGET_MAX_LPA = getattr(settings, "SALARY_BUDGET_MAX_LPA", 60)
        self.PREFERRED_CITIES = getattr(settings, "PREFERRED_CITIES", [])
        self.MAX_NOTICE_PERIOD_DAYS = 180
        
        # Constraint Sets
        self.SERVICES_FIRMS = set(getattr(settings, "SERVICES_FIRMS", []))
        self.HARD_DISQUALIFIERS = []
        self.SOFT_DISQUALIFIERS = []
        
        # Weights & Query Formulations
        self.SCORE_WEIGHTS = {"semantic": 0.35, "trajectory": 0.25, "behavioral": 0.25, "logistics": 0.15}
        self.JD_QUERY = ""
        self.MIN_YEARS_EXPERIENCE = 0.0
        self.MAX_YEARS_EXPERIENCE = 20.0
        
        if self.config_dict:
            constraints = self.config_dict.get("constraints", {})
            semantic = self.config_dict.get("semantic_targets", {})
            weights = self.config_dict.get("normalized_fusion_weights", {})
            
            if weights:
                self.SCORE_WEIGHTS = weights
                
            # Parse Advanced Structural Fields
            self.MIN_YEARS_EXPERIENCE = float(constraints.get("min_years_experience", 0.0))
            self.MAX_YEARS_EXPERIENCE = float(constraints.get("max_years_experience", 20.0))
            self.PREFERRED_CITIES = constraints.get("preferred_cities", [])
            self.MAX_NOTICE_PERIOD_DAYS = int(constraints.get("max_notice_period_days", 180))
            
            self.HARD_DISQUALIFIERS = constraints.get("hard_disqualifiers", [])
            self.SOFT_DISQUALIFIERS = constraints.get("soft_disqualifiers", [])
            
            # Construct High-Fidelity Query String from Tiered Evidence Matrix
            mandatory_reqs = " ".join([r["requirement_name"] for r in semantic.get("tier1_mandatory_evidence", [])])
            self.JD_QUERY = f"{semantic.get('primary_persona', '')} {mandatory_reqs}".strip()

def load_runtime_config(config_path=None):
    if config_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(project_root, "config", "generated_config_new.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return RuntimeConfig(json.load(f))
        except Exception:
            pass
    return RuntimeConfig(None)