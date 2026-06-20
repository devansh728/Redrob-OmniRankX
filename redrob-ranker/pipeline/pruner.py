import datetime
import json
import polars as pl
from utils.date_utils import REFERENCE_TODAY

def run(candidates, settings):
    if isinstance(candidates, list):
        from pipeline.loader import serialize_row
        flat_rows = [serialize_row(c) for c in candidates]
        lazy_df = pl.DataFrame(flat_rows).lazy()
    else:
        lazy_df = candidates

    duration_sum = pl.col("career_history").str.json_decode(
        pl.List(pl.Struct({"duration_months": pl.Int64}))
    ).list.eval(pl.element().struct.field("duration_months")).list.sum().fill_null(0)

    exp_years = pl.col("profile").str.json_path_match("$.years_of_experience").cast(pl.Float64).fill_null(0.0)
    failed_check_1 = ((duration_sum - exp_years * 12).abs() > 24).fill_null(False)

    job_years = pl.col("career_history").str.json_decode(
        pl.List(pl.Struct({"start_date": pl.String}))
    ).list.eval(pl.element().struct.field("start_date").str.slice(0, 4).cast(pl.Int64))
    earliest_job_year = job_years.list.min()

    edu_years = pl.col("education").str.json_decode(
        pl.List(pl.Struct({"end_year": pl.Int64}))
    ).list.eval(pl.element().struct.field("end_year"))
    latest_edu_year = edu_years.list.max()

    has_jobs_and_edu = earliest_job_year.is_not_null() & latest_edu_year.is_not_null()
    failed_check_2 = (has_jobs_and_edu & (latest_edu_year > earliest_job_year)).fill_null(False)

    github_score = pl.col("redrob_signals").str.json_path_match("$.github_activity_score").cast(pl.Float64).fill_null(-1.0)
    last_active = pl.col("redrob_signals").str.json_path_match("$.last_active_date").str.to_date()
    days_inactive = (pl.lit(REFERENCE_TODAY) - last_active).dt.total_days()
    # github_activity_score reflects activity in the last 12 months per schema definition. A score of 95+ requires sustained daily commits; platform inactivity of 18+ months is a physical contradiction.
    failed_check_3 = ((github_score >= 95.0) & (days_inactive > 18 * 30)).fill_null(False)

    profile_fields = [
        "anonymized_name", "headline", "summary", "location", 
        "country", "current_title", "current_company", 
        "current_company_size", "current_industry"
    ]
    has_empty_field = pl.col("profile").str.json_path_match("$.years_of_experience").is_null()
    for field in profile_fields:
        field_val = pl.col("profile").str.json_path_match(f"$.{field}")
        has_empty_field = has_empty_field | field_val.is_null() | (field_val.str.strip_chars() == "")

    failed_check_4 = ((pl.col("redrob_signals").str.json_path_match("$.profile_completeness_score").cast(pl.Float64) == 100.0) & has_empty_field).fill_null(False)

    open_to_work = pl.col("redrob_signals").str.json_path_match("$.open_to_work_flag") == "true"
    apps_submitted = pl.col("redrob_signals").str.json_path_match("$.applications_submitted_30d").cast(pl.Int64).fill_null(0)
    failed_check_5 = (open_to_work & (apps_submitted == 0) & (days_inactive > 6 * 30)).fill_null(False)

    honeypot_failures = (
        failed_check_1.cast(pl.Int32) +
        failed_check_2.cast(pl.Int32) +
        failed_check_3.cast(pl.Int32) +
        failed_check_4.cast(pl.Int32) +
        failed_check_5.cast(pl.Int32)
    )
    honeypot_dropped = (honeypot_failures >= 2).fill_null(False)

    is_research_or_academia = pl.col("career_history").str.json_decode(
        pl.List(pl.Struct({"industry": pl.String}))
    ).list.eval((pl.element().struct.field("industry") == "Research") | (pl.element().struct.field("industry") == "Academia"))
    all_research = (is_research_or_academia.list.len() > 0) & is_research_or_academia.list.all()

    deploy_langs = ["python", "java", "c\\+\\+", "rust", "\\bgo\\b", "scala", "c#", "javascript", "typescript"]
    regex_pattern = "(?i)" + "|".join(deploy_langs)
    has_deploy_lang = pl.col("precomputed_career_text").str.contains(regex_pattern)
    is_pure_researcher = (all_research & (~has_deploy_lang)).fill_null(False)

    cv_pattern = "(?i)" + "|".join(["vision", "image", "video", "speech", "audio", "voice", "robot", "control", "autonomous", "lidar", "sensor"])
    has_cv = pl.col("precomputed_career_text").str.contains(cv_pattern)

    nlp_pattern = "(?i)" + "|".join(["nlp", "natural language", "text", "information retrieval", "search", "retrieval", "ranking", "embedding", "vector", "llm", "language model", "transformer", "bert", "rag"])
    has_nlp = pl.col("precomputed_career_text").str.contains(nlp_pattern)
    is_wrong_domain = (has_cv & (~has_nlp)).fill_null(False)

    country = pl.col("profile").str.json_path_match("$.country").str.to_lowercase()
    is_outside_india = country.is_not_null() & (country != "india")
    is_unwilling_relocate = pl.col("redrob_signals").str.json_path_match("$.willing_to_relocate") == "false"
    outside_india_unwilling = (is_outside_india & is_unwilling_relocate).fill_null(False)

    companies = pl.col("career_history").str.json_decode(
        pl.List(pl.Struct({"company": pl.String, "industry": pl.String}))
    )
    is_service_entry = companies.list.eval(
        (pl.element().struct.field("company").is_in(list(settings.SERVICES_FIRMS))) | 
        (pl.element().struct.field("industry") == "IT Services")
    )
    all_services = (is_service_entry.list.len() > 0) & is_service_entry.list.all()

    product_pattern = "(?i)" + "|".join(["product", "startup", "scale", "saas", "platform"])
    has_product_lang = pl.col("precomputed_career_text").str.contains(product_pattern)
    is_services_only = (all_services & (~has_product_lang)).fill_null(False)

    min_exp = getattr(settings, "MIN_YEARS_EXPERIENCE", 0.0)
    max_exp = getattr(settings, "MAX_YEARS_EXPERIENCE", 20.0)
    failed_exp = (exp_years < min_exp) | (exp_years > max_exp)

    keep_mask = (
        (~honeypot_dropped) & 
        (~failed_exp) & 
        (~is_pure_researcher) & 
        (~is_wrong_domain) & 
        (~outside_india_unwilling) & 
        (~is_services_only)
    )


    pruned_df = lazy_df.filter(keep_mask).collect()

    pruned_candidates = []
    for row in pruned_df.iter_rows(named=True):
        candidate = {
            "candidate_id": row["candidate_id"],
            "precomputed_career_text": row["precomputed_career_text"]
        }
        for field in ["profile", "redrob_signals"]:
            val = row.get(field)
            candidate[field] = json.loads(val) if isinstance(val, str) else (val or {})
            
        for field in ["career_history", "education", "skills", "certifications", "languages"]:
            val = row.get(field)
            candidate[field] = json.loads(val) if isinstance(val, str) else (val or [])
            
        pruned_candidates.append(candidate)

    return pruned_candidates

