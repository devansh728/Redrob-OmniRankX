import os
import sys
import json
import argparse
from datetime import datetime
from multiprocessing import cpu_count
from llama_cpp import Llama
import json_repair

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text_utils import extract_raw_text_from_file
from utils.jd_schemas import Pass1Schema, Pass2Schema, Pass3Schema, CompiledConfig


def load_slm(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=cpu_count(),
        verbose=False
    )

def run_pass(llm, system_prompt, user_content, schema_class, pass_name):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    print("\n" + "="*60)
    print(f"PASS: {pass_name}")
    print(f"System: {system_prompt}")
    print(f"User: {user_content}")
    print(f"Expected Schema: {schema_class.__name__}")
    print("="*60)
    
    if sys.stdin.isatty():
        confirm = input("Proceed with inference for this pass? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted by user.")
            sys.exit(0)
    else:
        print("Non-interactive terminal detected. Auto-approving pass.")
        
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=512
    )
    
    response_text = response["choices"][0]["message"]["content"]
    parsed_dict = json_repair.loads(response_text)
    if not isinstance(parsed_dict, dict):
        raise ValueError(f"Failed to parse response as JSON dict: {response_text}")
        
    validated = schema_class.model_validate(parsed_dict)
    
    print("\nParsed Pydantic Model:")
    print(validated.model_dump_json(indent=2))
    print("="*60)
    
    return validated

def normalize_weights(pass3):
    tech = pass3.technical_depth_weight
    beh = pass3.behavioral_signals_weight
    log = pass3.logistics_importance_weight
    total = float(tech + beh + log)
    
    raw_semantic = (tech * 0.45) / (total * 0.70)
    raw_trajectory = (tech * 0.25) / (total * 0.70)
    raw_behavioral = beh / total
    raw_logistics = log / total
    
    raw_sum = raw_semantic + raw_trajectory + raw_behavioral + raw_logistics
    
    return {
        "semantic": round(raw_semantic / raw_sum, 4),
        "trajectory": round(raw_trajectory / raw_sum, 4),
        "behavioral": round(raw_behavioral / raw_sum, 4),
        "logistics": round(raw_logistics / raw_sum, 4)
    }

def apply_ambiguity_damping(config_dict, pass2, pass3):
    score = pass3.jd_ambiguity_score
    damping_factor = score / 50.0
    config_dict["meta"]["ambiguity_damping_factor"] = damping_factor
    
    query = config_dict["semantic_targets"]["jd_query"]
    if score >= 7:
        expansions = " ".join(pass2.semantic_expansions)
        query = f"{query} {expansions}"
    config_dict["semantic_targets"]["jd_query"] = query
    return config_dict

def build_output(pass1, pass2, pass3, weights, jd_source):
    meta = {
        "compiled_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "jd_source_file": os.path.basename(jd_source),
        "ambiguity_damping_factor": 0.0
    }
    
    constraints = {
        "experience": {
            "min": float(pass2.min_years_experience),
            "max": float(pass2.max_years_experience)
        },
        "blacklist_firms": pass2.excluded_companies,
        "forbidden_industries": pass2.forbidden_industries
    }
    
    base_query = f"{pass1.primary_persona} {' '.join(pass1.must_demonstrate)}"
    
    semantic_targets = {
        "business_intent": pass1.business_intent,
        "primary_persona": pass1.primary_persona,
        "evidence_expectations": pass1.must_demonstrate,
        "bounded_concept_expansions": pass2.semantic_expansions,
        "jd_query": base_query
    }
    
    behavioral_priorities = {
        "startup_velocity_coefficient": round(pass3.startup_vs_enterprise / 10.0, 2),
        "shipper_vs_researcher_coefficient": round(pass3.shipper_vs_researcher / 10.0, 2),
        "builder_vs_manager_coefficient": round(pass3.builder_vs_manager / 10.0, 2)
    }
    
    config_dict = {
        "meta": meta,
        "constraints": constraints,
        "semantic_targets": semantic_targets,
        "behavioral_priorities": behavioral_priorities,
        "normalized_fusion_weights": weights
    }
    
    return config_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd-path", default="../India_runs_data_and_ai_challenge/job_description.docx")
    parser.add_argument("--output-path", default="config/generated_config.json")
    parser.add_argument("--model-path", default="models/qwen2.5-3b-instruct/Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    args = parser.parse_args()
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    jd_abs_path = os.path.abspath(os.path.join(project_root, args.jd_path))
    output_abs_path = os.path.abspath(os.path.join(project_root, args.output_path))
    model_abs_path = os.path.abspath(os.path.join(project_root, args.model_path))
    
    raw_jd_text = extract_raw_text_from_file(jd_abs_path)
    if len(raw_jd_text) > 3000:
        raw_jd_text = raw_jd_text[:3000]
        
    llm = load_slm(model_abs_path)
    
    sys_prompt1 = "You are an expert technical recruiter analyzing a Job Description for a senior engineering role. Output ONLY valid JSON with no markdown, no explanation."
    user_prompt1 = (
        "Analyze this Job Description. Return a JSON object with:\n"
        '- "business_intent": one sentence describing the core corporate problem being solved (max 20 words)\n'
        '- "primary_persona": the single best archetype name for this candidate (max 5 words)\n'
        '- "must_demonstrate": a JSON array of exactly 3-5 strings, each being an observable proof of competence\n\n'
        f"Job Description:\n{raw_jd_text}\n\n"
        "Return only JSON. No markdown. No explanation."
    )
    pass1 = run_pass(llm, sys_prompt1, user_prompt1, Pass1Schema, "Pass 1 - Persona & Evidence")
    
    sys_prompt2 = "You are a hiring constraints analyst. Extract hard filters and exclusions from a Job Description. Output ONLY valid JSON with no markdown, no explanation."
    user_prompt2 = (
        "Extract constraints from this Job Description:\n"
        '- "min_years_experience": minimum years required as a float (use 0.0 if not stated)\n'
        '- "max_years_experience": maximum years preferred as a float (use 20.0 if not stated)\n'
        '- "excluded_companies": JSON array of company names explicitly avoided (empty array if none stated)\n'
        '- "forbidden_industries": JSON array of industries explicitly excluded (empty array if none stated)\n'
        '- "semantic_expansions": JSON array of EXACTLY 10-15 concept keywords that describe this role beyond the literal text. Focus on adjacent domains and transferable skills.\n\n'
        f"Job Description:\n{raw_jd_text}\n\n"
        "Return only JSON. No markdown. No explanation."
    )
    pass2 = run_pass(llm, sys_prompt2, user_prompt2, Pass2Schema, "Pass 2 - Guardrails & Exclusions")
    
    sys_prompt3 = "You are an organizational culture analyst. Score emphasis dimensions from a Job Description using integers only. Output ONLY valid JSON with no markdown, no explanation."
    user_prompt3 = (
        "Score this Job Description on the following dimensions using integers 1-10 only:\n"
        '- "startup_vs_enterprise": 1=pure enterprise, 10=pure startup\n'
        '- "shipper_vs_researcher": 1=deep research focus, 10=ship fast to production\n'
        '- "builder_vs_manager": 1=pure manager/leader, 10=hands-on builder/IC\n'
        '- "technical_depth_weight": how much technical expertise matters (1=low, 10=critical)\n'
        '- "behavioral_signals_weight": how much engagement/availability signals matter (1=low, 10=critical)\n'
        '- "logistics_importance_weight": how much location/relocation/mode matters (1=low, 10=critical)\n'
        '- "jd_ambiguity_score": how vague/underspecified is this JD (1=very precise, 10=very vague)\n\n'
        f"Job Description:\n{raw_jd_text}\n\n"
        "Return only JSON integers. No markdown. No explanation."
    )
    pass3 = run_pass(llm, sys_prompt3, user_prompt3, Pass3Schema, "Pass 3 - Philosophy & Priority Scoring")
    
    weights = normalize_weights(pass3)
    config_dict = build_output(pass1, pass2, pass3, weights, jd_abs_path)
    config_dict = apply_ambiguity_damping(config_dict, pass2, pass3)
    
    CompiledConfig.model_validate(config_dict)
    
    os.makedirs(os.path.dirname(output_abs_path), exist_ok=True)
    with open(output_abs_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
        
    print(f"\nSuccessfully compiled config to {output_abs_path}")
    print("\nSummary Table:")
    print(f"{'Metric':<30} | {'Value':<50}")
    print("-" * 85)
    print(f"{'Primary Persona':<30} | {config_dict['semantic_targets']['primary_persona']:<50}")
    print(f"{'Experience Bounds':<30} | {config_dict['constraints']['experience']['min']} - {config_dict['constraints']['experience']['max']} years")
    print(f"{'Normalized weights':<30} | {str(config_dict['normalized_fusion_weights']):<50}")
    print(f"{'JD Ambiguity Score':<30} | {pass3.jd_ambiguity_score:<50}")

if __name__ == "__main__":
    main()
