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

    career_history_expr = pl.col("career_history").str.json_decode(
        pl.List(pl.Struct({
            "company": pl.String,
            "title": pl.String,
            "industry": pl.String,
            "duration_months": pl.Int64,
            "start_date": pl.String
        }))
    )

    education_expr = pl.col("education").str.json_decode(
        pl.List(pl.Struct({
            "end_year": pl.Int64
        }))
    )

    duration_sum = career_history_expr.list.eval(pl.element().struct.field("duration_months")).list.sum().fill_null(0)

    exp_years = pl.col("profile").str.json_path_match("$.years_of_experience").cast(pl.Float64).fill_null(0.0)
    failed_check_1 = ((duration_sum - exp_years * 12).abs() > 24).fill_null(False)

    job_years = career_history_expr.list.eval(pl.element().struct.field("start_date").str.slice(0, 4).cast(pl.Int64))
    earliest_job_year = job_years.list.min()

    edu_years = education_expr.list.eval(pl.element().struct.field("end_year"))
    latest_edu_year = edu_years.list.max()

    has_jobs_and_edu = earliest_job_year.is_not_null() & latest_edu_year.is_not_null()
    failed_check_2 = (has_jobs_and_edu & (latest_edu_year > earliest_job_year)).fill_null(False)

    github_score = pl.col("redrob_signals").str.json_path_match("$.github_activity_score").cast(pl.Float64).fill_null(-1.0)
    last_active = pl.col("redrob_signals").str.json_path_match("$.last_active_date").str.to_date()
    days_inactive = (pl.lit(REFERENCE_TODAY) - last_active).dt.total_days()
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

    min_exp = getattr(settings, "MIN_YEARS_EXPERIENCE", 0.0)
    max_exp = getattr(settings, "MAX_YEARS_EXPERIENCE", 20.0)
    failed_exp = (exp_years < min_exp) | (exp_years > max_exp)

    disqualify_mask = pl.lit(False)
    for d in getattr(settings, "HARD_DISQUALIFIERS", []):
        cond = d.get("condition_name", "")
        val = (d.get("rejection_value", "") or "").lower()
        if not val:
            continue
        if cond == "Researcher":
            titles = career_history_expr.list.eval(pl.element().struct.field("title").str.to_lowercase().str.contains("researcher"))
            disqualify_mask = disqualify_mask | ((titles.list.len() > 0) & titles.list.all()).fill_null(False)
        elif cond == "pure_research":
            disqualify_mask = disqualify_mask | pl.col("precomputed_career_text").str.to_lowercase().str.contains("academic lab|research-only|pure research|academia").fill_null(False)
        elif cond == "recent_experience":
            disqualify_mask = disqualify_mask | pl.col("precomputed_career_text").str.to_lowercase().str.contains("langchain").fill_null(False)
        elif cond == "Title-chasers":
            durations = career_history_expr.list.eval(pl.element().struct.field("duration_months"))
            num_jobs = durations.list.len()
            avg_tenure = durations.list.sum() / num_jobs
            is_hopper = (num_jobs >= 3) & (avg_tenure < 18.0)
            has_senior_title = career_history_expr.list.eval(
                pl.element().struct.field("title").str.to_lowercase().str.contains("senior|staff|principal|lead")
            ).list.any()
            disqualify_mask = disqualify_mask | (is_hopper & has_senior_title).fill_null(False)
        elif cond == "Consulting firms":
            is_service_entry = career_history_expr.list.eval(
                (pl.element().struct.field("company").is_in(list(settings.SERVICES_FIRMS))) | 
                (pl.element().struct.field("industry") == "IT Services")
            )
            all_services = (is_service_entry.list.len() > 0) & is_service_entry.list.all()
            product_pattern = "(?i)" + "|".join(["product", "startup", "scale", "saas", "platform"])
            has_product_lang = pl.col("precomputed_career_text").str.contains(product_pattern)
            is_services_only = all_services & (~has_product_lang)
            disqualify_mask = disqualify_mask | is_services_only.fill_null(False)
        elif cond == "No_Relevant_Work_History":
            current_title = pl.col("profile").str.json_path_match("$.current_title").str.to_lowercase()
            is_marketing = current_title.str.contains("marketing manager|marketing")
            all_marketing = career_history_expr.list.eval(
                pl.element().struct.field("title").str.to_lowercase().str.contains("marketing manager|marketing")
            ).list.all()
            disqualify_mask = disqualify_mask | (is_marketing | all_marketing).fill_null(False)

    country = pl.col("profile").str.json_path_match("$.country").str.to_lowercase()
    is_outside_india = country.is_not_null() & (country != "india")
    is_unwilling_relocate = pl.col("redrob_signals").str.json_path_match("$.willing_to_relocate") == "false"
    
    if getattr(settings, "WILLING_TO_RELOCATE_REQUIRED", False):
        outside_india_unwilling = (is_outside_india & is_unwilling_relocate).fill_null(False)
    else:
        outside_india_unwilling = pl.lit(False)

    keep_mask = (
        (~honeypot_dropped) & 
        (~failed_exp) & 
        (~outside_india_unwilling) & 
        (~disqualify_mask)
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
