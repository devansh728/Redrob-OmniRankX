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
            
        c["behavioral_score"] = float(min(1.0, max(0.0, comp)))
        
    return candidates
