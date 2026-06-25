"""
app.py — OmniRank-X Gradio Space entrypoint.

What changed from the original:
  - Config is now user-selectable: a dropdown for built-in configs
    (generated_config_2.json, generated_config.json) OR a custom JSON file
    upload.  The selected config is loaded before each pipeline run so
    different JDs can be evaluated without restarting the Space.
  - load_runtime_config() is called with the resolved path, so the
    JD-specific weights, thresholds, and disqualifier rules all come from
    whichever config the user chooses.
  - DataFrame output now shows the full top-100 table, not just top-10.
  - Log section shows per-stage timings copied from main.py's style.
  - Gradio theme upgraded to Monochrome for a cleaner look.
  - Error messages are surfaced in the log box rather than silently returning
    None.
"""

import os
import csv
import time
import gradio as gr
import pandas as pd
from pipeline import loader, pruner, semantic, trajectory, behavioral, fusion
from config.config_loader import load_runtime_config

# ------------------------------------------------------------------
# Built-in config paths (relative to this file's directory)
# ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILTIN_CONFIGS = {
    "Full Config (generated_config_2.json)": os.path.join(_HERE, "config", "generated_config_2.json"),
    "Minimal Config (generated_config.json)": os.path.join(_HERE, "config", "generated_config.json"),
}
_DEFAULT_CONFIG_LABEL = "Full Config (generated_config_2.json)"


# ------------------------------------------------------------------
# Core ranking function
# ------------------------------------------------------------------
def run_ranking(file_obj, config_choice: str, custom_config_file):
    """Runs the full 5-stage ranking pipeline and returns results.

    Inputs (from Gradio):
        file_obj          - uploaded candidates JSON/JSONL file object
        config_choice     - label from the built-in config dropdown
        custom_config_file - optional uploaded custom config JSON file object

    Outputs (to Gradio):
        DataFrame         - ranked top-N candidates (up to 100 rows)
        str (file path)   - path to the downloadable submission CSV
        str               - log / status text
    """
    if not file_obj:
        return None, None, "Please upload a candidate JSON or JSONL file."

    # --- Resolve config path ---
    if custom_config_file is not None:
        config_path = custom_config_file.name
        config_label = f"custom: {os.path.basename(config_path)}"
    else:
        config_path = _BUILTIN_CONFIGS.get(config_choice, _BUILTIN_CONFIGS[_DEFAULT_CONFIG_LABEL])
        config_label = config_choice

    try:
        settings = load_runtime_config(config_path)
    except Exception as e:
        return None, None, f"Failed to load config '{config_label}': {e}"

    file_path = file_obj.name
    logs = [f"Candidates: {os.path.basename(file_path)}",
            f"Config: {config_label}"]

    try:
        t_total = time.time()

        # Stage 0 - Load
        t0 = time.time()
        candidates = loader.load_all(file_path)
        import polars as pl
        if isinstance(candidates, pl.LazyFrame):
            total_loaded = candidates.select(pl.len()).collect().item()
        else:
            total_loaded = len(candidates)
        logs.append(f"Stage 0 - Loaded {total_loaded} candidates in {time.time()-t0:.2f}s")

        # Stage 1 - Prune
        t0 = time.time()
        pruned = pruner.run(candidates, settings)
        total_pruned = len(pruned)
        logs.append(f"Stage 1 - Pruned to {total_pruned} in {time.time()-t0:.2f}s")

        # Stage 2 - Semantic
        t0 = time.time()
        semantic_scored = semantic.run(pruned, None, settings)
        logs.append(f"Stage 2 - Semantic scored {len(semantic_scored)} in {time.time()-t0:.2f}s")

        # Stage 3 - Trajectory
        t0 = time.time()
        trajectory_scored = trajectory.run(semantic_scored, settings)
        logs.append(f"Stage 3 - Trajectory scored in {time.time()-t0:.2f}s")

        # Stage 4 - Behavioral
        t0 = time.time()
        behavioral_scored = behavioral.run(trajectory_scored, settings)
        logs.append(f"Stage 4 - Behavioral scored in {time.time()-t0:.2f}s")

        # Stage 5 - Fusion
        t0 = time.time()
        ranked = fusion.run(behavioral_scored, settings)
        logs.append(f"Stage 5 - Fused & ranked {len(ranked)} in {time.time()-t0:.2f}s")

        logs.append(f"Total wall time: {time.time()-t_total:.2f}s")

        # --- Write submission CSV ---
        out_dir = os.path.join(_HERE, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "sandbox_submission.csv")

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for item in ranked:
                writer.writerow([
                    item["candidate_id"],
                    item["rank"],
                    f"{item['final_score']:.4f}",
                    item["reasoning"],
                ])

        # --- Build DataFrame for display (all ranked rows) ---
        display_rows = []
        for item in ranked:
            display_rows.append({
                "Rank": item["rank"],
                "Candidate ID": item["candidate_id"],
                "Final Score": round(item["final_score"], 4),
                "Semantic": round(item.get("semantic_score", 0.0), 3),
                "Trajectory": round(item.get("trajectory_score", 0.0), 3),
                "Behavioral": round(item.get("behavioral_score", 0.0), 3),
                "Logistics": round(item.get("logistics_score", 0.0), 3),
                "Reasoning": item["reasoning"],
            })
        df = pd.DataFrame(display_rows)

        return df, out_csv, "\n".join(logs)

    except Exception as e:
        import traceback
        return None, None, f"Pipeline error:\n{traceback.format_exc()}"


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
custom_theme = gr.themes.Monochrome()

with gr.Blocks(
    theme=custom_theme,
    title="OmniRank-X | AI Candidate Ranking",
    css="""
    .container { max-width: 1200px; margin: auto; }
    .header-text { text-align: center; margin-bottom: 2rem; padding-top: 2rem; }
    .header-text h1 { color: var(--primary-600); font-weight: 800; font-size: 2.5rem; letter-spacing: -0.025em; }
    .header-text p { color: var(--body-text-color-subdued); font-size: 1.1rem; }
    .output-log { font-family: monospace; font-size: 0.9rem; }
    .file-upload { border: 2px dashed var(--border-color-primary); border-radius: var(--radius-lg); }
    """
) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            """
            <div class="header-text">
                <h1>OmniRank-X</h1>
                <p>Intelligent Candidate Discovery & Ranking Pipeline</p>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📂 Input Data")
                file_input = gr.File(
                    label="Upload Candidates",
                    file_types=[".parquet", ".json", ".jsonl", ".gz"],
                    elem_classes="file-upload",
                )
                gr.Markdown("*Supported formats: .parquet, .json, .jsonl, .jsonl.gz (up to ~100k candidates)*")
                
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Pipeline Configuration")
                config_dropdown = gr.Dropdown(
                    choices=list(_BUILTIN_CONFIGS.keys()),
                    value=_DEFAULT_CONFIG_LABEL,
                    label="Active Job Description (JD)",
                    info="Select a pre-compiled JD config.",
                )
                with gr.Accordion("Advanced: Custom JD Config", open=False):
                    gr.Markdown("*Overrides the dropdown selection above.*")
                    custom_config_input = gr.File(
                        label="Upload Custom Config (JSON)",
                        file_types=[".json"]
                    )
        
        with gr.Row():
            run_btn = gr.Button("🚀 Start Ranking Pipeline", variant="primary", size="lg")
            
        gr.Markdown("---")
            
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 🏆 Top Ranked Candidates")
                table_output = gr.DataFrame(
                    label="Candidate Rankings",
                    wrap=True,
                    height=500,
                )
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Pipeline Execution Log")
                log_output = gr.Textbox(
                    label="Status & Timings",
                    lines=15,
                    max_lines=20,
                    interactive=False,
                    elem_classes="output-log",
                    show_label=False
                )
                file_output = gr.File(label="📥 Download Submission CSV")

    run_btn.click(
        fn=run_ranking,
        inputs=[file_input, config_dropdown, custom_config_input],
        outputs=[table_output, file_output, log_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
