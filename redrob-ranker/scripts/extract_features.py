import polars as pl

lazy_df = pl.scan_parquet("data/candidates.parquet")

processed_df = lazy_df.select([
    pl.col("candidate_id"),
    pl.col("precomputed_career_text"),
    pl.col("profile").str.json_path_match("$.years_of_experience").cast(pl.Float64).alias("experience_years"),
    pl.col("profile").str.json_path_match("$.location").alias("city"),
    pl.col("profile").str.json_path_match("$.current_title").alias("current_title"),
    pl.col("profile").str.json_path_match("$.current_company_size").alias("company_size"),
    pl.col("redrob_signals").str.json_path_match("$.profile_completeness_score").cast(pl.Float64).alias("completeness"),
    pl.col("redrob_signals").str.json_path_match("$.github_activity_score").cast(pl.Float64).alias("github_score"),
    pl.col("redrob_signals").str.json_path_match("$.recruiter_response_rate").cast(pl.Float64).alias("response_rate"),
    pl.col("redrob_signals").str.json_path_match("$.last_active_date").alias("last_active_date"),
    pl.col("redrob_signals").str.json_path_match("$.notice_period_days").cast(pl.Int32).alias("notice_period"),
    (pl.col("redrob_signals").str.json_path_match("$.open_to_work_flag") == "true").alias("open_to_work"),
    pl.col("redrob_signals").str.json_path_match("$.expected_salary_range_inr_lpa.max").cast(pl.Float64).alias("salary_max"),
    pl.col("career_history"),
    pl.col("skills"),
    pl.col("education")
])

results = processed_df.head(5).collect()

with pl.Config(fmt_str_lengths=1000, tbl_cols=20):
    print(results)





