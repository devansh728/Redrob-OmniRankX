import os
import sys
import json
import argparse
from datetime import datetime
from multiprocessing import cpu_count
from typing import List
from llama_cpp import Llama
import json_repair

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text_utils import extract_raw_text_from_file
from utils.jd_schemas import Pass1Schema, Pass2Schema, Pass3Schema

def load_slm(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=cpu_count(),
        verbose=False
    )

def segment_jd_text(raw_text: str) -> List[str]:
    lines = raw_text.split("\n")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        current_chunk.append(line)
        current_length += len(line)
        if current_length > 3000:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0
            
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

def execute_inference_pass(llm, system_prompt, user_content, schema_class, execution_label):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    print(f"\n[ORCHESTRATOR] Initiating Pass: {execution_label}")
    
    if sys.stdin.isatty():
        confirm = input(f"Confirm inference execution for {execution_label}? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Execution halted by operator.")
            sys.exit(0)
            
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=1024
    )
    
    raw_response_output = response["choices"][0]["message"]["content"]
    repaired_json_dict = json_repair.loads(raw_response_output)
    
    if not isinstance(repaired_json_dict, dict):
        raise ValueError(f"SLM output failed JSON formatting constraint: {raw_response_output}")
        
    return schema_class.model_validate(repaired_json_dict)

def calculate_grounded_weights(pass1: Pass1Schema, pass2: Pass2Schema, pass3: Pass3Schema) -> dict:
    hard_disqualifiers_count = len(pass2.hard_disqualifiers)
    soft_disqualifiers_count = len(pass2.soft_disqualifiers)
    mandatory_evidence_count = len(pass1.tier1_mandatory_evidence)
    preferred_evidence_count = len(pass1.tier2_preferred_evidence)
    
    text_counts = pass3.raw_text_mention_counts
    
    logistics_mentions = text_counts.get("location", 0) + text_counts.get("salary", 0)
    behavioral_mentions = text_counts.get("notice_period", 0) + 1 
    technical_mentions = text_counts.get("skills", 0) + text_counts.get("architecture", 0)
    base_semantic_score = float((mandatory_evidence_count * 10) + (preferred_evidence_count * 4) + technical_mentions)
    base_trajectory_score = float((hard_disqualifiers_count * 12) + (soft_disqualifiers_count * 6))
    base_behavioral_score = float(behavioral_mentions * 8)
    base_logistics_score = float(logistics_mentions * 5)
    
    total_raw_mass = base_semantic_score + base_trajectory_score + base_behavioral_score + base_logistics_score
    if total_raw_mass == 0:
        return {"semantic": 0.35, "trajectory": 0.25, "behavioral": 0.25, "logistics": 0.15}
    return {
        "semantic": round(base_semantic_score / total_raw_mass, 4),
        "trajectory": round(base_trajectory_score / total_raw_mass, 4),
        "behavioral": round(base_behavioral_score / total_raw_mass, 4),
        "logistics": round(base_logistics_score / total_raw_mass, 4)
    }

def verify_named_entities_gate(raw_text: str, config_data: dict) -> List[str]:
    critical_check_tokens = ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini", "Pune", "Bangalore", "30-day", "5-9"]
    missing_tokens = []
    config_dump_string = json.dumps(config_data).lower()
    
    for token in critical_check_tokens:
        if token.lower() in raw_text.lower():
            if token.lower() not in config_dump_string:
                missing_tokens.append(token)
    return missing_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd-path", default="../India_runs_data_and_ai_challenge/job_description.docx")
    parser.add_argument("--output-path", default="config/generated_config_new.json")
    parser.add_argument("--model-path", default="models/qwen2.5-3b-instruct/Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    args = parser.parse_args()
    
    jd_abs_path = os.path.abspath(os.path.join(project_root, args.jd_path))
    output_abs_path = os.path.abspath(os.path.join(project_root, args.output_path))
    model_abs_path = os.path.abspath(os.path.join(project_root, args.model_path))
    
    full_untruncated_jd_text = extract_raw_text_from_file(jd_abs_path)
    text_partitions = segment_jd_text(full_untruncated_jd_text)
    
    llm = load_slm(model_abs_path)
    
    sys_prompt1 = "You are a recruitment structural matrix compiler. Analyze the raw Job Description text segments. Output ONLY valid JSON."
    user_prompt1 = (
        f"Analyze these full text sections from the Job Description:\n\n{' '.join(text_partitions)}\n\n"
        "Compile the target JSON structure EXACTLY following this shape configuration. Do not skip any keys:\n"
        "{\n"
        '  "business_intent": "Core corporate problem being solved",\n'
        '  "primary_persona": "Single target structural archetype",\n'
        '  "anti_personas": ["Archetypes to immediately avoid"],\n'
        '  "tier1_mandatory_evidence": [\n'
        '    {\n'
        '      "requirement_name": "Built retrieval systems",\n'
        '      "evidence_proof_expectations": ["Scaled vector DB", "Tuned BM25"],\n'
        '      "is_mandatory_tier1": true,\n'
        '      "traceability": {\n'
        '        "extracted_fact": "Must have search experience",\n'
        '        "verbatim_text_quote": "Experience with embeddings and ranking",\n'
        '        "source_section": "Requirements"\n'
        '      }\n'
        '    }\n'
        '  ],\n'
        '  "tier2_preferred_evidence": []\n'
        "}\n\n"
        "Output ONLY valid JSON. No markdown formatting blocks."
    )
    pass1 = execute_inference_pass(llm, sys_prompt1, user_prompt1, Pass1Schema, "Pass 1 - Targets & Evidence Matrix")
    
    sys_prompt2 = "You are an enterprise hiring guardrails analyst. Extract strict hard and soft constraints. Output ONLY valid JSON."
    user_prompt2 = (
        f"Analyze these full text sections from the Job Description:\n\n{' '.join(text_partitions)}\n\n"
        "Compile the constraint data EXACTLY following this JSON shape. Do not skip any keys:\n"
        "CRITICAL INSTRUCTION: If the JD lists specific company names (e.g. TCS, Infosys) or specific numbers (30-day, 5-9), YOU MUST EXTRACT THEM VERBATIM into your conditions. DO NOT SUMMARIZE.\n"
        "{\n"
        '  "min_years_experience": 5.0,\n'
        '  "max_years_experience": 9.0,\n'
        '  "preferred_cities": ["Pune", "Bangalore"],\n'
        '  "willing_to_relocate_required": false,\n'
        '  "max_notice_period_days": 30,\n'
        '  "max_salary_budget_lpa": 60.0,\n'
        '  "hard_disqualifiers": [\n'
        '    {\n'
        '      "condition_name": "No pure researchers",\n'
        '      "target_field_path": "career_history.industry",\n'
        '      "check_operator": "NOT_EQUALS",\n'
        '      "rejection_value": "Research",\n'
        '      "traceability": {\n'
        '        "extracted_fact": "Must have shipped to production",\n'
        '        "verbatim_text_quote": "We will not move forward with pure researchers",\n'
        '        "source_section": "Disqualifiers"\n'
        '      }\n'
        '    }\n'
        '  ],\n'
        '  "soft_disqualifiers": [],\n'
        '  "bounded_concept_expansions": ["Vector DBs", "Ranking algorithms", "System Design"]\n'
        "}\n\n"
        "Output ONLY valid JSON. No markdown formatting blocks."
    )
    pass2 = execute_inference_pass(llm, sys_prompt2, user_prompt2, Pass2Schema, "Pass 2 - Guardrails & Exclusions Graph")
    
    sys_prompt3 = "You are an organizational entropy analyst. Evaluate text counts and density parameters. Output ONLY valid JSON."
    user_prompt3 = (
        f"Analyze these text sections:\n\n{' '.join(text_partitions)}\n\n"
        "Output ONLY this specific structural JSON shape with integers from 1-10. Do not use decimals:\n"
        "{\n"
        '  "startup_vs_enterprise": 9,\n'
        '  "shipper_vs_researcher": 10,\n'
        '  "builder_vs_manager": 8,\n'
        '  "generalist_vs_specialist": 6,\n'
        '  "jd_ambiguity_score": 3,\n'
        '  "raw_text_mention_counts": {\n'
        '    "location": 2,\n'
        '    "notice_period": 1,\n'
        '    "salary": 0,\n'
        '    "skills": 14,\n'
        '    "architecture": 5\n'
        '  }\n'
        "}\n\n"
        "Output ONLY valid JSON integers. No markdown formatting blocks."
    )
    pass3 = execute_inference_pass(llm, sys_prompt3, user_prompt3, Pass3Schema, "Pass 3 - Philosophy & Quantitative Extraction")
    
    grounded_weights = calculate_grounded_weights(pass1, pass2, pass3)
    
    master_config_output = {
        "meta": {
            "compiled_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "jd_source_file": os.path.basename(jd_abs_path),
            "ambiguity_damping_factor": round(pass3.jd_ambiguity_score / 20.0, 4)
        },
        "constraints": pass2.model_dump(),
        "semantic_targets": pass1.model_dump(),
        "behavioral_priorities": pass3.model_dump(),
        "normalized_fusion_weights": grounded_weights
    }
    
    # Run the validation substring gate
    missing_entities = verify_named_entities_gate(full_untruncated_jd_text, master_config_output)
    if missing_entities:
        print(f"\n[🚨 VALIDATION FAILURE ALERT] The compiled configuration dropped critical named tokens: {missing_entities}")
        debug_path = output_abs_path.replace(".json", "_FAILED_DEBUG.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(master_config_output, f, indent=2)
        print(f"[🔍 DEBUG] I saved the broken LLM output to: {debug_path} so you can inspect what it missed.")
        print("[🚨 ERROR] execution terminating. Force structural prompt alignment.")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(output_abs_path), exist_ok=True)
    with open(output_abs_path, "w", encoding="utf-8") as f:
        json.dump(master_config_output, f, indent=2)
        
    print(f"\n[SUCCESS] Indestructible configuration compiled to: {output_abs_path}")

if __name__ == "__main__":
    main()