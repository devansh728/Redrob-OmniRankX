import os
import json
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT = os.path.normpath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "India_runs_data_and_ai_challenge",
        "candidates.jsonl"
    )
)

OUTPUT = os.path.normpath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "data",
        "candidates.parquet"
    )
)

def serialize_row(row):
    flat_row = {"candidate_id": str(row.get("candidate_id", ""))}
    career_list = row.get("career_history") or []
    career_text = " ".join([
        str(role.get("description", "")) for role in career_list 
        if isinstance(role, dict) and role.get("description")
    ])
    flat_row["precomputed_career_text"] = career_text
    
    object_fields = ["profile", "redrob_signals"]
    for field in object_fields:
        flat_row[field] = json.dumps(row.get(field) or {})
        
    array_fields = ["career_history", "education", "skills", "certifications", "languages"]
    for field in array_fields:
        flat_row[field] = json.dumps(row.get(field) or [])
        
    return flat_row

def main():
    if not os.path.exists(INPUT):
        raise FileNotFoundError(INPUT)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    schema = pa.schema([
        ("candidate_id", pa.string()),
        ("precomputed_career_text", pa.string()),
        ("profile", pa.string()),
        ("career_history", pa.string()),
        ("education", pa.string()),
        ("skills", pa.string()),
        ("certifications", pa.string()),
        ("languages", pa.string()),
        ("redrob_signals", pa.string()),
    ])

    BATCH_SIZE = 5000
    current_batch = []
    writer = pq.ParquetWriter(OUTPUT, schema, compression="zstd")

    try:
        with open(INPUT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_row = json.loads(line)
                    current_batch.append(serialize_row(raw_row))
                
                if len(current_batch) >= BATCH_SIZE:
                    batch_table = pa.Table.from_pylist(current_batch, schema=schema)
                    writer.write_table(batch_table)
                    current_batch = []
            
            if current_batch:
                batch_table = pa.Table.from_pylist(current_batch, schema=schema)
                writer.write_table(batch_table)

        print(f"Created: {OUTPUT}")

    finally:
        writer.close()

if __name__ == "__main__":
    main()
