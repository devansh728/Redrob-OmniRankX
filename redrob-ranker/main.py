import os
import sys
import time
import csv
from pipeline import loader, pruner, semantic, trajectory, behavioral, fusion
from config import settings

def main():
    start_time = time.time()
    
    candidates_path = sys.argv[1] if len(sys.argv) > 1 else "../India_runs_data_and_ai_challenge/candidates.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/submission1.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading candidates from {candidates_path}")
    candidates = loader.load_all(candidates_path)
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
