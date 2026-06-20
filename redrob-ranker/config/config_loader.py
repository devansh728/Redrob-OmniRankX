import os
import json
from config import settings

class RuntimeConfig:
    def __init__(self, config_dict=None):
        self.config_dict = config_dict or {}
        self.STAGE2_TOP_K = getattr(settings, "STAGE2_TOP_K", 1000)
        self.FINAL_TOP_N = getattr(settings, "FINAL_TOP_N", 100)
        self.SALARY_BUDGET_MAX_LPA = getattr(settings, "SALARY_BUDGET_MAX_LPA", 60)
        self.PREFERRED_CITIES = getattr(settings, "PREFERRED_CITIES", [])
        self.SERVICES_FIRMS = set(getattr(settings, "SERVICES_FIRMS", []))
        self.SCORE_WEIGHTS = dict(getattr(settings, "SCORE_WEIGHTS", {}))
        self.JD_QUERY = getattr(settings, "JD_QUERY", "")
        self.JD_SKILL_WEIGHTS = dict(getattr(settings, "JD_SKILL_WEIGHTS", {}))
        self.MIN_YEARS_EXPERIENCE = 0.0
        self.MAX_YEARS_EXPERIENCE = 20.0
        
        if self.config_dict:
            constraints = self.config_dict.get("constraints", {})
            sem_targets = self.config_dict.get("semantic_targets", {})
            fusion_weights = self.config_dict.get("normalized_fusion_weights", {})
            
            if fusion_weights:
                self.SCORE_WEIGHTS = fusion_weights
            
            if sem_targets.get("jd_query"):
                self.JD_QUERY = sem_targets["jd_query"]
            
            exp = constraints.get("experience", {})
            if "min" in exp:
                self.MIN_YEARS_EXPERIENCE = float(exp["min"])
            if "max" in exp:
                self.MAX_YEARS_EXPERIENCE = float(exp["max"])
                
            blacklist = constraints.get("blacklist_firms", [])
            if blacklist:
                self.SERVICES_FIRMS = self.SERVICES_FIRMS.union(set(blacklist))

def load_runtime_config(config_path=None):
    if config_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(project_root, "config", "generated_config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return RuntimeConfig(data)
        except Exception:
            pass
    return RuntimeConfig(None)
