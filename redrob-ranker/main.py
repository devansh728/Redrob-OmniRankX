import os
import sys
import time
import csv
from pipeline import loader, pruner, semantic, trajectory, behavioral, fusion
from config.config_loader import load_runtime_config

_proj_root = os.path.dirname(os.path.abspath(__file__))
_debug_config = os.path.join(_proj_root, "config", "generated_config_new_2.json")
settings = load_runtime_config(_debug_config if os.path.exists(_debug_config) else None)

def get_next_submission_path(base_dir="outputs"):
    os.makedirs(base_dir, exist_ok=True)
    counter = 1
    while True:
        path = os.path.join(base_dir, f"submission{counter}.csv")
        if not os.path.exists(path):
            return path
        counter += 1

def main():
    start_time = time.time()
    
    candidates_path = sys.argv[1] if len(sys.argv) > 1 else "data/candidates.parquet"
    output_path = sys.argv[2] if len(sys.argv) > 2 else get_next_submission_path("outputs")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading candidates from {candidates_path}")
    candidates = loader.load_all(candidates_path)
    import polars as pl
    if isinstance(candidates, pl.LazyFrame):
        print(f"Loaded candidates LazyFrame in {time.time() - start_time:.2f}s")
    else:
        print(f"Loaded {len(candidates)} candidates in {time.time() - start_time:.2f}s")


    print("Running Stage 1: Pruning")
    t0 = time.time()
    pruned = pruner.run(candidates, settings)
    print(f"Pruning complete. {len(pruned)} candidates remaining in {time.time() - t0:.2f}s")

    print("Running Stage 2: Semantic scoring")
    t0 = time.time()
    semantic_scored = semantic.run(pruned, None, settings)
    print(f"Semantic scoring complete in {time.time() - t0:.2f}s")

    print("Running Stage 3: Trajectory scoring")
    t0 = time.time()
    trajectory_scored = trajectory.run(semantic_scored, settings)
    print(f"Trajectory scoring complete in {time.time() - t0:.2f}s")

    print("Running Stage 4: Behavioral scoring")
    t0 = time.time()
    behavioral_scored = behavioral.run(trajectory_scored, settings)
    print(f"Behavioral scoring complete in {time.time() - t0:.2f}s")

    print("Running Stage 5: Fusion and ranking")
    t0 = time.time()
    ranked = fusion.run(behavioral_scored, settings)
    print(f"Fusion and ranking complete in {time.time() - t0:.2f}s")

    print(f"Writing output to {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for item in ranked:
            formatted_score = f"{item['final_score']:.4f}"
            writer.writerow([
                item["candidate_id"],
                item["rank"],
                formatted_score,
                item["reasoning"]
            ])

    print(f"OmniRank-X completed successfully in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
