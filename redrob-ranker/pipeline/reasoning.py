from utils.date_utils import parse_date, REFERENCE_TODAY

def build_reasoning(candidate, rank):
    signals = candidate.get("redrob_signals", {})
    
    reasons = []
    
    sem = candidate.get("semantic_score", 0.0)
    if sem > 0.75:
        reasons.append("Strong semantic match for ML/IR role.")
        
    traj = candidate.get("trajectory_score", 0.0)
    if traj > 0.7:
        reasons.append("Deep product-company career arc.")
        
    last_act_str = signals.get("last_active_date")
    last_act = parse_date(last_act_str)
    is_active_14 = False
    if last_act:
        days = (REFERENCE_TODAY - last_act).days
        if days <= 14:
            is_active_14 = True
            
    if signals.get("open_to_work_flag") is True and is_active_14:
        reasons.append("Actively searching.")
        
    notice = signals.get("notice_period_days")
    if notice is not None and notice <= 30:
        reasons.append("Immediately available.")
        
    git_raw = signals.get("github_activity_score")
    if git_raw is not None and git_raw > 70.0:
        reasons.append("High GitHub activity score.")
        
    if not reasons:
        return f"Ranks in top {rank} by composite scoring across semantic, career, and behavioral signals."
        
    return " ".join(reasons)
