import gzip
import json
import os
import polars as pl

def stream_candidates(path):
    if not os.path.exists(path):
        return

    is_gz = path.endswith(".gz")
    open_fn = gzip.open if is_gz else open
    mode = "rt" if is_gz else "r"
    encoding = "utf-8"

    with open_fn(path, mode, encoding=encoding) as f:
        first_char = ""
        for line in f:
            stripped = line.strip()
            if stripped:
                first_char = stripped[0]
                break

    with open_fn(path, mode, encoding=encoding) as f:
        if first_char == "[":
            data = json.load(f)
            for item in data:
                yield item
        else:
            for line in f:
                line_str = line.strip()
                if line_str:
                    yield json.loads(line_str)

def serialize_row(row):
    flat_row = {"candidate_id": str(row.get("candidate_id", ""))}
    
    career_val = row.get("career_history")
    if isinstance(career_val, str):
        try:
            career_list = json.loads(career_val)
        except:
            career_list = []
    else:
        career_list = career_val or []

    career_text = " ".join([
        str(role.get("description", "")) for role in career_list 
        if isinstance(role, dict) and role.get("description")
    ])
    flat_row["precomputed_career_text"] = career_text
    
    object_fields = ["profile", "redrob_signals"]
    for field in object_fields:
        val = row.get(field)
        if isinstance(val, str):
            flat_row[field] = val
        else:
            flat_row[field] = json.dumps(val or {})
            
    array_fields = ["career_history", "education", "skills", "certifications", "languages"]
    for field in array_fields:
        val = row.get(field)
        if isinstance(val, str):
            flat_row[field] = val
        else:
            flat_row[field] = json.dumps(val or [])
            
    return flat_row

def load_all(path):
    if path.endswith(".parquet"):
        return pl.scan_parquet(path)
    
    if os.path.exists("data/candidates.parquet") and "sample_candidates" not in path:
        return pl.scan_parquet("data/candidates.parquet")
    elif os.path.exists("redrob-ranker/data/candidates.parquet") and "sample_candidates" not in path:
        return pl.scan_parquet("redrob-ranker/data/candidates.parquet")
        
    raw_rows = list(stream_candidates(path))
    flat_rows = [serialize_row(r) for r in raw_rows]
    df = pl.DataFrame(flat_rows)
    return df.lazy()

