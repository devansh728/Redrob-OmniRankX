import datetime
from utils.date_utils import parse_date, REFERENCE_TODAY
from utils.text_utils import concat_career_text

def run(candidates, settings):
    pruned_candidates = []
    
    for candidate in candidates:
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])
        education = candidate.get("education", [])
        signals = candidate.get("redrob_signals", {})
        
        honeypot_failures = 0
        
        duration_sum = sum(job.get("duration_months", 0) for job in career_history)
        exp_years = profile.get("years_of_experience", 0)
        if abs(duration_sum - exp_years * 12) > 24:
            honeypot_failures += 1
            
        job_dates = [parse_date(job.get("start_date")) for job in career_history]
        valid_job_years = [d.year for d in job_dates if d]
        if valid_job_years and education:
            earliest_job_year = min(valid_job_years)
            if any(edu.get("end_year", 0) > earliest_job_year for edu in education):
                honeypot_failures += 1
                
        github_score = signals.get("github_activity_score", -1)
        last_active = parse_date(signals.get("last_active_date"))
        if github_score >= 95 and last_active:
            days_inactive = (REFERENCE_TODAY - last_active).days
            # github_activity_score reflects activity in the last 12 months per schema definition. A score of 95+ requires sustained daily commits; platform inactivity of 18+ months is a physical contradiction.
            if days_inactive > 18 * 30:
                honeypot_failures += 1
                
        if signals.get("profile_completeness_score") == 100:
            has_empty = False
            for f in ["anonymized_name", "headline", "summary", "location", "country", "current_title", "current_company", "current_company_size", "current_industry"]:
                if not profile.get(f):
                    has_empty = True
                    break
            if profile.get("years_of_experience") is None:
                has_empty = True
            if has_empty:
                honeypot_failures += 1
                
        if signals.get("open_to_work_flag") is True and signals.get("applications_submitted_30d", 0) == 0:
            if last_active:
                days_inactive = (REFERENCE_TODAY - last_active).days
                if days_inactive > 6 * 30:
                    honeypot_failures += 1
                    
        if honeypot_failures >= 2:
            continue
            
        industries = [job.get("industry", "") for job in career_history]
        is_pure_research = False
        if industries and all(ind in {"Research", "Academia"} for ind in industries):
            all_text = concat_career_text(candidate).lower()
            deploy_langs = ["python", "java", "c++", "rust", "go", "scala", "c#", "javascript", "typescript"]
            if not any(lang in all_text for lang in deploy_langs):
                is_pure_research = True
        if is_pure_research:
            continue
            
        all_text = concat_career_text(candidate).lower()
        cv_keywords = ["vision", "image", "video", "speech", "audio", "voice", "robot", "control", "autonomous", "lidar", "sensor"]
        nlp_keywords = ["nlp", "natural language", "text", "information retrieval", "search", "retrieval", "ranking", "embedding", "vector", "llm", "language model", "transformer", "bert", "rag"]
        has_cv = any(kw in all_text for kw in cv_keywords)
        has_nlp = any(kw in all_text for kw in nlp_keywords)
        if has_cv and not has_nlp:
            continue
            
        country = profile.get("country", "").lower()
        willing_relocate = signals.get("willing_to_relocate")
        if country and country != "india" and willing_relocate is False:
            continue
            
        all_services = False
        if career_history:
            all_services = True
            for job in career_history:
                comp = job.get("company")
                if comp not in settings.SERVICES_FIRMS:
                    all_services = False
                    break
        if all_services:
            product_keywords = ["product", "startup", "scale", "saas", "platform"]
            has_product_lang = any(kw in all_text for kw in product_keywords)
            if not has_product_lang:
                continue
                
        pruned_candidates.append(candidate)
        
    return pruned_candidates
