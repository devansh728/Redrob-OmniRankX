import datetime
from utils.date_utils import parse_date

def product_tenure_score(career_history, services_firms):
    if not career_history:
        return 0.0
    total_months = 0.0
    for job in career_history:
        company = job.get("company", "")
        industry = job.get("industry", "")
        size = job.get("company_size", "")
        duration = job.get("duration_months", 0)
        
        if company in services_firms or industry == "IT Services":
            continue
            
        if size in ["51-200", "201-500", "501-1000", "1001-5000", "5001-10000"]:
            total_months += duration
        elif size == "1-10":
            total_months += duration * 0.7
            
    return min(60.0, total_months) / 60.0

def hop_penalty(career_history):
    if not career_history:
        return 0.0
    hops = sum(1 for job in career_history if not job.get("is_current") and job.get("duration_months", 0) < 18)
    return hops * 0.1

def long_stint_bonus(career_history, services_firms):
    if not career_history:
        return 0.0
    bonus = 0.0
    for job in career_history:
        company = job.get("company", "")
        industry = job.get("industry", "")
        duration = job.get("duration_months", 0)
        
        if company not in services_firms and industry != "IT Services":
            if duration >= 24:
                bonus += 0.15
    return min(0.3, bonus)

def domain_progression_score(career_history):
    if not career_history:
        return 0.0
    sorted_jobs = sorted(
        career_history,
        key=lambda x: parse_date(x.get("start_date")) or datetime.date.min
    )
    total_weight = 0.0
    weighted_score = 0.0
    ml_keywords = ["ml", "machine learning", "ai", "nlp", "search", "retrieval", "ranking", "data scientist", "deep learning", "information retrieval", "llm"]
    for idx, job in enumerate(sorted_jobs):
        title = job.get("title", "").lower()
        has_ml = any(kw in title for kw in ml_keywords)
        weight = idx + 1.0
        total_weight += weight
        if has_ml:
            weighted_score += weight
            
    return (weighted_score / total_weight) if total_weight > 0 else 0.0

def run(candidates, config):
    for c in candidates:
        history = c.get("career_history", [])
        pts = product_tenure_score(history, config.SERVICES_FIRMS)
        penalty = hop_penalty(history)
        hop_score = max(0.0, 1.0 - penalty)
        bonus = long_stint_bonus(history, config.SERVICES_FIRMS)
        dps = domain_progression_score(history)
        
        c["trajectory_score"] = float(min(1.0, max(0.0, (pts + hop_score + bonus + dps) / 4.0)))
    return candidates
