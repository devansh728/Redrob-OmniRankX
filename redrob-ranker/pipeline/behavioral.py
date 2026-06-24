import math
from utils.date_utils import parse_date, REFERENCE_TODAY

def recency_score(last_active_date):
    d = parse_date(last_active_date)
    if not d:
        return 0.0
    days = (REFERENCE_TODAY - d).days
    return math.exp(-max(0, days) / 90.0)

def github_score(raw):
    if raw == -1 or raw is None:
        return 0.5
    return min(1.0, max(0.0, raw / 100.0))

def notice_score(notice_period_days):
    if notice_period_days is None:
        return 0.6
    if notice_period_days <= 30:
        return 1.0
    elif notice_period_days <= 60:
        return 0.8
    else:
        return 0.6

def salary_fit(expected_salary, budget_max):
    if not expected_salary:
        return 1.0
    max_val = expected_salary.get("max")
    if max_val is None:
        return 1.0
    if max_val > budget_max * 1.25:
        return 0.5
    return 1.0

def run(candidates, config):
    for c in candidates:
        signals = c.get("redrob_signals", {})
        
        r_score = recency_score(signals.get("last_active_date"))
        
        resp_rate = signals.get("recruiter_response_rate")
        if resp_rate is None or resp_rate == -1:
            resp_rate = 0.5
            
        git_raw = signals.get("github_activity_score")
        git_score = github_score(git_raw)
        
        int_rate = signals.get("interview_completion_rate")
        if int_rate is None or int_rate == -1:
            int_rate = 0.5
            
        n_score = notice_score(signals.get("notice_period_days"))
        
        sal_score = salary_fit(signals.get("expected_salary_range_inr_lpa"), config.SALARY_BUDGET_MAX_LPA)
        
        oar = signals.get("offer_acceptance_rate")
        if oar is None or oar == -1:
            oar = 0.5

        comp = (
            r_score * 0.25 +
            resp_rate * 0.25 +
            git_score * 0.15 +
            int_rate * 0.15 +
            n_score * 0.10 +
            sal_score * 0.10
        )
        
        if signals.get("open_to_work_flag") is True:
            comp = comp * 1.2
            
        behavioral_score = min(1.0, max(0.0, comp))
        
        text = c.get("precomputed_career_text", "").lower()
        for sd in getattr(config, "SOFT_DISQUALIFIERS", []):
            cond = sd.get("condition_name", "")
            penalty = float(sd.get("penalty_weight", 0.0))
            if penalty <= 0.0:
                continue
            
            trigger = False
            if cond == "Bounce Between Startups":
                history = c.get("career_history", [])
                has_startup = any(j.get("company_size") in ["1-10", "11-50"] for j in history)
                has_bigtech = any(any(kw in j.get("company", "").lower() for kw in ["google", "meta"]) for j in history)
                if has_startup and not has_bigtech:
                    trigger = True
            elif cond == "architecture_or_tech_lead_experience":
                has_hr_tech = any(kw in text for kw in ["hr-tech", "hr tech", "recruiting tech", "marketplace"])
                if has_hr_tech:
                    trigger = True
            elif cond == "Computer vision, speech, or robotics":
                cv_keywords = ["vision", "image", "video", "speech", "audio", "voice", "robot", "control", "autonomous", "lidar", "sensor"]
                nlp_keywords = ["nlp", "natural language", "text", "information retrieval", "search", "retrieval", "ranking", "embedding", "vector", "llm", "language model", "transformer", "bert", "rag"]
                has_cv = any(kw in text for kw in cv_keywords)
                has_nlp = any(kw in text for kw in nlp_keywords)
                if has_cv and not has_nlp:
                    trigger = True
            elif cond == "NoRedrobPlatform":
                is_active = (signals.get("open_to_work_flag") is True) or (signals.get("applications_submitted_30d", 0) > 0)
                if not is_active:
                    trigger = True
            elif cond == "Inactive_Candidate":
                d_active = parse_date(signals.get("last_active_date"))
                days = (REFERENCE_TODAY - d_active).days if d_active else 999
                r_rate = signals.get("recruiter_response_rate")
                if r_rate is None or r_rate == -1:
                    r_rate = 0.5
                if days > 180 and r_rate < 0.15:
                    trigger = True
            
            if trigger:
                behavioral_score = behavioral_score * (1.0 - penalty)
                
        c["behavioral_score"] = float(min(1.0, max(0.0, behavioral_score)))
        
    return candidates
