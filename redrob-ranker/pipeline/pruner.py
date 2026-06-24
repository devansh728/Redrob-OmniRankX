"""
pruner.py — Stage 1: deterministic hard filtering of the candidate pool.

Two layers of pruning happen here:
  1. Honeypot/consistency checks (date arithmetic, profile completeness
     paradoxes, etc) — unchanged in spirit from earlier versions, still
     vectorized in Polars for speed across 100k candidates.
  2. JD-compiled hard disqualifiers — now executed GENERICALLY via
     pipeline.rule_engine.evaluate_rule(), dispatching on each rule's
     rule_type enum rather than matching on condition_name strings. This
     is what makes pruning survive a re-compiled or entirely different JD
     without any code change here.

Because rule_engine evaluates Python dicts per-candidate (not vectorized
Polars expressions), hard disqualifier application happens in a second
pass over the already-honeypot-filtered set, not inside the big Polars
filter expression. This trades a small amount of speed for being able to
add JD-agnostic rules without writing new Polars expression code every
time the compiler invents a new disqualifier — given the disqualifier set
is per-JD-run and typically under ~15 rules, the per-candidate Python loop
over the post-honeypot-filtered pool is fast enough to stay well inside
the 10-second Stage 1 budget on a 100k pool, since the Polars pass already
removes most of the volume before the rule loop ever runs.
"""

import json
import polars as pl
from utils.date_utils import REFERENCE_TODAY
from pipeline.rule_engine import evaluate_rule, evaluate_escape_hatch, count_unresolved_rules


def run_honeypot_filter(lazy_df: "pl.LazyFrame") -> "pl.LazyFrame":
    """Vectorized honeypot/consistency checks, unchanged from prior design.

    Input: a Polars LazyFrame with the standard flattened candidate
           columns (profile/career_history/etc as JSON strings).
    Output: a LazyFrame filtered to drop honeypots and out-of-band
            experience-year candidates.
    How it works: five independent boolean expressions, each flagging one
            class of internal contradiction; a candidate is dropped if it
            fails two or more.
    """
    career_history_expr = pl.col("career_history").str.json_decode(
        pl.List(pl.Struct({
            "company": pl.String,
            "title": pl.String,
            "industry": pl.String,
            "duration_months": pl.Int64,
            "start_date": pl.String,
        }))
    )

    education_expr = pl.col("education").str.json_decode(
        pl.List(pl.Struct({"end_year": pl.Int64}))
    )

    duration_sum = career_history_expr.list.eval(
        pl.element().struct.field("duration_months")
    ).list.sum().fill_null(0)

    exp_years = pl.col("profile").str.json_path_match("$.years_of_experience").cast(pl.Float64).fill_null(0.0)
    failed_check_1 = ((duration_sum - exp_years * 12).abs() > 24).fill_null(False)

    job_years = career_history_expr.list.eval(
        pl.element().struct.field("start_date").str.slice(0, 4).cast(pl.Int64)
    )
    earliest_job_year = job_years.list.min()

    edu_years = education_expr.list.eval(pl.element().struct.field("end_year"))
    latest_edu_year = edu_years.list.max()

    has_jobs_and_edu = earliest_job_year.is_not_null() & latest_edu_year.is_not_null()
    failed_check_2 = (has_jobs_and_edu & (latest_edu_year > earliest_job_year)).fill_null(False)

    github_score = pl.col("redrob_signals").str.json_path_match("$.github_activity_score").cast(pl.Float64).fill_null(-1.0)
    last_active = pl.col("redrob_signals").str.json_path_match("$.last_active_date").str.to_date()
    days_inactive = (pl.lit(REFERENCE_TODAY) - last_active).dt.total_days()
    # github_activity_score reflects activity in the last 12 months per schema
    # definition. A score of 95+ requires sustained daily commits; platform
    # inactivity of 18+ months is a physical contradiction.
    failed_check_3 = ((github_score >= 95.0) & (days_inactive > 18 * 30)).fill_null(False)

    profile_fields = [
        "anonymized_name", "headline", "summary", "location",
        "country", "current_title", "current_company",
        "current_company_size", "current_industry",
    ]
    has_empty_field = pl.col("profile").str.json_path_match("$.years_of_experience").is_null()
    for field in profile_fields:
        field_val = pl.col("profile").str.json_path_match(f"$.{field}")
        has_empty_field = has_empty_field | field_val.is_null() | (field_val.str.strip_chars() == "")

    failed_check_4 = (
        (pl.col("redrob_signals").str.json_path_match("$.profile_completeness_score").cast(pl.Float64) == 100.0)
        & has_empty_field
    ).fill_null(False)

    open_to_work = pl.col("redrob_signals").str.json_path_match("$.open_to_work_flag") == "true"
    apps_submitted = pl.col("redrob_signals").str.json_path_match("$.applications_submitted_30d").cast(pl.Int64).fill_null(0)
    failed_check_5 = (open_to_work & (apps_submitted == 0) & (days_inactive > 6 * 30)).fill_null(False)

    honeypot_failures = (
        failed_check_1.cast(pl.Int32)
        + failed_check_2.cast(pl.Int32)
        + failed_check_3.cast(pl.Int32)
        + failed_check_4.cast(pl.Int32)
        + failed_check_5.cast(pl.Int32)
    )
    honeypot_dropped = (honeypot_failures >= 2).fill_null(False)

    return lazy_df.with_columns(_honeypot_dropped=honeypot_dropped)


def run(candidates, settings):
    if isinstance(candidates, list):
        from pipeline.loader import serialize_row
        flat_rows = [serialize_row(c) for c in candidates]
        lazy_df = pl.DataFrame(flat_rows).lazy()
    else:
        lazy_df = candidates

    lazy_df = run_honeypot_filter(lazy_df)

    min_exp = getattr(settings, "MIN_YEARS_EXPERIENCE", 0.0)
    max_exp = getattr(settings, "MAX_YEARS_EXPERIENCE", 20.0)
    exp_years_expr = pl.col("profile").str.json_path_match("$.years_of_experience").cast(pl.Float64).fill_null(0.0)
    failed_exp = (exp_years_expr < min_exp) | (exp_years_expr > max_exp)

    country = pl.col("profile").str.json_path_match("$.country").str.to_lowercase()
    is_outside_india = country.is_not_null() & (country != "india")
    is_unwilling_relocate = pl.col("redrob_signals").str.json_path_match("$.willing_to_relocate") == "false"

    if getattr(settings, "WILLING_TO_RELOCATE_REQUIRED", False):
        outside_india_unwilling = (is_outside_india & is_unwilling_relocate).fill_null(False)
    else:
        outside_india_unwilling = pl.lit(False)

    keep_mask = (
        (~pl.col("_honeypot_dropped"))
        & (~failed_exp)
        & (~outside_india_unwilling)
    )

    survivors_df = lazy_df.filter(keep_mask).collect()

    survivors = []
    for row in survivors_df.iter_rows(named=True):
        candidate = {
            "candidate_id": row["candidate_id"],
            "precomputed_career_text": row["precomputed_career_text"],
        }
        for field in ["profile", "redrob_signals"]:
            val = row.get(field)
            candidate[field] = json.loads(val) if isinstance(val, str) else (val or {})
        for field in ["career_history", "education", "skills", "certifications", "languages"]:
            val = row.get(field)
            candidate[field] = json.loads(val) if isinstance(val, str) else (val or [])
        survivors.append(candidate)

    # --- Hard disqualifier pass: generic, rule_type-driven ---
    hard_disqualifiers = getattr(settings, "HARD_DISQUALIFIERS", [])
    unresolved_count = count_unresolved_rules(hard_disqualifiers)
    if unresolved_count:
        print(f"[pruner] WARNING: {unresolved_count} hard disqualifier(s) have an "
              f"unresolved rule_type and will NOT be applied. Review the compiled "
              f"config and either fix the compiler's classification or assign a "
              f"rule_type manually before this run is trusted.")

    if not hard_disqualifiers:
        return survivors

    final_pool = []
    disqualified_count = 0
    for candidate in survivors:
        disqualified = False
        for rule in hard_disqualifiers:
            if rule.get("rule_type", "unresolved") == "unresolved":
                continue
            result = evaluate_rule(rule, candidate)
            if result is True:
                disqualified = True
                break
        if disqualified:
            disqualified_count += 1
        else:
            final_pool.append(candidate)

    print(f"[pruner] Hard disqualifiers removed {disqualified_count} of "
          f"{len(survivors)} honeypot/experience survivors "
          f"({len(hard_disqualifiers) - unresolved_count} active rules).")

    return final_pool