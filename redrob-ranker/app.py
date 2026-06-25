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
            candidates = candidates.collect().to_dicts()
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
with gr.Blocks(
    theme=gr.themes.Monochrome(primary_hue="indigo", neutral_hue="slate"),
    title="OmniRank-X - Candidate Ranking",
) as demo:

    gr.Markdown(
        """
        # OmniRank-X - Intelligent Candidate Discovery & Ranking
        Upload a candidate JSON / JSONL file (up to ~100 candidates for the sandbox demo).
        The full 5-stage pipeline runs entirely on CPU with no internet access during ranking.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="Upload Candidates (.json, .jsonl, or .jsonl.gz)",
                file_types=[".json", ".jsonl", ".gz"],
            )
        with gr.Column(scale=1):
            config_dropdown = gr.Dropdown(
                choices=list(_BUILTIN_CONFIGS.keys()),
                value=_DEFAULT_CONFIG_LABEL,
                label="Built-in Config",
                info="Select the pre-compiled JD config to use.",
            )
            custom_config_input = gr.File(
                label="Or upload a custom config JSON (overrides dropdown)",
                file_types=[".json"],
            )

    run_btn = gr.Button("Run Ranking Pipeline", variant="primary", size="lg")

    log_output = gr.Textbox(
        label="Pipeline Logs & Status",
        lines=8,
        max_lines=14,
        interactive=False,
    )

    with gr.Row():
        table_output = gr.DataFrame(
            label="Ranked Candidates (up to Top 100)",
            wrap=True,
        )

    file_output = gr.File(label="Download Submission CSV")

    run_btn.click(
        fn=run_ranking,
        inputs=[file_input, config_dropdown, custom_config_input],
        outputs=[table_output, file_output, log_output],
    )

    gr.Markdown(
        """
        ---
        **OmniRank-X** - Redrob Hackathon - 5-stage pipeline:
        Polars loader -> Deterministic pruner -> ONNX semantic RRF ->
        Trajectory scoring -> Behavioral composite -> Weighted fusion
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
