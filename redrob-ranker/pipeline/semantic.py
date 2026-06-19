import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from utils.text_utils import concat_career_text, tokenize_for_bm25

def load_embedder(model_dir="models/bge-small-en-v1.5-int8"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    session = ort.InferenceSession(os.path.join(model_dir, "model.onnx"))
    return tokenizer, session

def get_rrf_ranks(candidates, scores_dict):
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (-scores_dict[c["candidate_id"]], c["candidate_id"])
    )
    return {c["candidate_id"]: rank + 1 for rank, c in enumerate(sorted_candidates)}

def run(candidates, embedder=None, config=None):
    if not candidates:
        return []

    if embedder is None:
        try:
            tokenizer, session = load_embedder()
        except Exception:
            tokenizer, session = None, None
    else:
        tokenizer, session = embedder

    texts = [concat_career_text(c) for c in candidates]

    if tokenizer and session:
        input_names = [x.name for x in session.get_inputs()]
        batch_size = 64
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np"
            )
            ort_inputs = {name: inputs[name] for name in input_names if name in inputs}
            outputs = session.run(None, ort_inputs)
            batch_embeddings = outputs[0][:, 0, :]
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            batch_embeddings = batch_embeddings / (norms + 1e-9)
            embeddings_list.append(batch_embeddings)
        candidate_embeddings = np.concatenate(embeddings_list, axis=0)

        query_text = "Represent this sentence for searching relevant passages: " + config.JD_QUERY
        q_inputs = tokenizer(
            [query_text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        q_ort_inputs = {name: q_inputs[name] for name in input_names if name in q_inputs}
        q_outputs = session.run(None, q_ort_inputs)
        q_emb = q_outputs[0][0, 0, :]
        q_norm = np.linalg.norm(q_emb)
        q_emb = q_emb / (q_norm + 1e-9)

        sim_scores = np.dot(candidate_embeddings, q_emb)
        cos_scores = {c["candidate_id"]: float(sim_scores[idx]) for idx, c in enumerate(candidates)}
    else:
        cos_scores = {c["candidate_id"]: 0.0 for c in candidates}

    corpus = [tokenize_for_bm25(t) for t in texts]
    bm25 = BM25Okapi(corpus)
    q_tokens = tokenize_for_bm25(config.JD_QUERY)
    bm25_raw = bm25.get_scores(q_tokens)
    bm25_scores = {c["candidate_id"]: float(bm25_raw[idx]) for idx, c in enumerate(candidates)}

    skill_scores = {}
    for c in candidates:
        scores = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
        total_weight = 0.0
        weighted_sum = 0.0
        for skill, weight in config.JD_SKILL_WEIGHTS.items():
            score = scores.get(skill, 0.0) / 100.0
            weighted_sum += score * weight
            total_weight += weight
        skill_scores[c["candidate_id"]] = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    ranks_cos = get_rrf_ranks(candidates, cos_scores)
    ranks_bm25 = get_rrf_ranks(candidates, bm25_scores)
    ranks_skills = get_rrf_ranks(candidates, skill_scores)

    rrf_scores = {}
    for c in candidates:
        cid = c["candidate_id"]
        r_cos = ranks_cos[cid]
        r_bm = ranks_bm25[cid]
        r_sk = ranks_skills[cid]
        rrf_scores[cid] = 1.0 / (60.0 + r_cos) + 1.0 / (60.0 + r_bm) + 1.0 / (60.0 + r_sk)

    max_rrf = max(rrf_scores.values())
    min_rrf = min(rrf_scores.values())
    rrf_range = max_rrf - min_rrf

    for c in candidates:
        cid = c["candidate_id"]
        if rrf_range > 0:
            c["semantic_score"] = float((rrf_scores[cid] - min_rrf) / rrf_range)
        else:
            c["semantic_score"] = 1.0

    sorted_result = sorted(
        candidates,
        key=lambda c: (-c["semantic_score"], c["candidate_id"])
    )

    return sorted_result[:config.STAGE2_TOP_K]
