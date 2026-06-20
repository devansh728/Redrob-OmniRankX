import os
import csv
import gradio as gr
import pandas as pd
from pipeline import loader, pruner, semantic, trajectory, behavioral, fusion
from config import settings

def run_ranking(file_obj):
    if not file_obj:
        return None, None, "Please upload a candidate file."
    
    file_path = file_obj.name
    try:
        candidates = loader.load_all(file_path)
        import polars as pl
        if isinstance(candidates, pl.LazyFrame):
            total_loaded = candidates.select(pl.len()).collect().item()
        else:
            total_loaded = len(candidates)
        
        pruned = pruner.run(candidates, settings)
        total_pruned = len(pruned)
        
        semantic_scored = semantic.run(pruned, None, settings)
        trajectory_scored = trajectory.run(semantic_scored, settings)
        behavioral_scored = behavioral.run(trajectory_scored, settings)
        ranked = fusion.run(behavioral_scored, settings)
        
        out_csv = os.path.join("outputs", "sandbox_submission.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
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
                
        display_data = []
        for item in ranked[:10]:
            display_data.append({
                "Rank": item["rank"],
                "ID": item["candidate_id"],
                "Score": f"{item['final_score']:.4f}",
                "Reasoning": item["reasoning"]
            })
        df = pd.DataFrame(display_data)
        
        log_msg = f"Successfully loaded {total_loaded} candidates. {total_pruned} passed pruning. Ranked top {len(ranked)}."
        return df, out_csv, log_msg
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# OmniRank-X Candidate Discovery & Ranking")
    gr.Markdown("Upload candidates JSON or JSONL file to discover and rank the top 100 candidates against the Senior AI Engineer role.")
    
    with gr.Row():
        file_input = gr.File(label="Upload Candidates (.json or .jsonl)", file_types=[".json", ".jsonl", ".gz"])
        
    run_btn = gr.Button("Run Ranking Pipeline", variant="primary")
    
    log_output = gr.Textbox(label="Logs & Status")
    
    with gr.Row():
        table_output = gr.DataFrame(label="Top 10 Ranked Candidates")
        file_output = gr.File(label="Download Ranked CSV (submission.csv)")
        
    run_btn.click(
        fn=run_ranking,
        inputs=file_input,
        outputs=[table_output, file_output, log_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
