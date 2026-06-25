"""
semantic.py — Stage 2: dual-source semantic scoring via RRF fusion of
career-text embeddings, BM25 lexical matching, and skill-assessment
alignment.

WHAT CHANGED:
config.JD_SKILL_WEIGHTS is now a list of (keyword_phrase, weight) tuples
derived from the compiled JD's tier1/tier2 evidence (see
config_loader.py's _derive_skill_keywords_from_evidence), not a flat dict
keyed by exact skill name. The previous version did an exact dict lookup
(scores.get(skill, 0.0)) against config.JD_SKILL_WEIGHTS.items() — since
that dict was always empty (confirmed dead code path), skill_scores
silently computed 0.0 for every candidate, every run.

The fix does substring matching: for each candidate's actual
skill_assessment_scores keys (which are candidate data, not known to the
compiler in advance — e.g. "Fine-tuning LLMs", "FAISS", "Pinecone"), check
whether any derived JD keyword phrase appears in (or contains) that key,
case-insensitively. This is intentionally approximate — see the limitation
note in config_loader.py — but a missed match safely contributes nothing
rather than penalizing a candidate, which is the correct failure mode for
an approximate signal feeding into a multi-source RRF fusion.
"""

import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from utils.text_utils import concat_career_text, tokenize_for_bm25


_HF_ONNX_REPO = "optimum/bge-small-en-v1.5"
_TMP_MODEL_CACHE = "/tmp/models/bge-small-en-v1.5-int8"


def load_embedder(model_dir=None):
    """Loads the ONNX INT8 bge-small embedder.

    Resolution order (first working path wins):
      1. Explicit model_dir argument (if given and contains model.onnx).
      2. models/bge-small-en-v1.5-int8/ relative to the project root —
         the committed, Git-LFS-tracked copy baked into the Space repo.
         This is the expected path for the deployed HF Space and gives
         zero cold-start download time.
      3. /tmp/models/ cache — used when running in an ephemeral environment
         where the LFS files were not pulled (rare in normal Space operation).
      4. HuggingFace Hub snapshot_download — last resort if all local paths
         are missing.  Adds ~30-45 s on first cold start.

    Input:  optional explicit path to a local ONNX model directory.
    Output: (AutoTokenizer, onnxruntime.InferenceSession) tuple, or
            (None, None) if all resolution attempts fail.
    """
    # Resolve the project-root-relative default path (works regardless of cwd).
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    committed_dir = os.path.join(proj_root, "models", "bge-small-en-v1.5-int8")

    # Determine which directory to try loading from.
    if model_dir and os.path.isfile(os.path.join(model_dir, "model.onnx")):
        resolved_dir = model_dir
    elif os.path.isfile(os.path.join(committed_dir, "model.onnx")):
        # Happy path: model is committed to the Space repo via Git LFS.
        resolved_dir = committed_dir
    elif os.path.isfile(os.path.join(_TMP_MODEL_CACHE, "model.onnx")):
        # /tmp cache from a previous container run.
        resolved_dir = _TMP_MODEL_CACHE
    else:
        # Last resort: download from HF Hub.
        try:
            from huggingface_hub import snapshot_download
            print(f"[semantic] model.onnx not found locally. "
                  f"Downloading {_HF_ONNX_REPO} from HuggingFace Hub ...")
            snapshot_download(
                repo_id=_HF_ONNX_REPO,
                local_dir=_TMP_MODEL_CACHE,
                ignore_patterns=["*.msgpack", "*.h5", "flax_*", "*.pt", "*.bin"],
            )
            print(f"[semantic] Model downloaded to {_TMP_MODEL_CACHE}")
            resolved_dir = _TMP_MODEL_CACHE
        except Exception as e:
            print(f"[semantic] WARNING: could not download model: {e}. "
                  "Embedding will be skipped (BM25 + skills only).")
            return None, None

    tokenizer = AutoTokenizer.from_pretrained(resolved_dir)
    session = ort.InferenceSession(
        os.path.join(resolved_dir, "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    return tokenizer, session


def get_rrf_ranks(candidates, scores_dict):
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (-scores_dict[c["candidate_id"]], c["candidate_id"]),
    )
    return {c["candidate_id"]: rank + 1 for rank, c in enumerate(sorted_candidates)}


def score_skill_alignment(candidate: dict, jd_skill_weights: list) -> float:
    """Computes a weighted skill-alignment score for one candidate.

    Input: a candidate dict, a list of (keyword_phrase, weight) tuples.
    Output: float 0.0-1.0.
    How it works: for each of the candidate's actual
            skill_assessment_scores entries, finds every JD keyword phrase
            that substring-matches the skill name (in either direction —
            "LLMs" matches inside "Fine-tuning LLMs", and a JD keyword
            that happens to BE a full skill name matches directly), and
            accumulates weight * (score / 100) for each match. The final
            sum is normalized by the total weight actually matched, not
            by the full JD_SKILL_WEIGHTS list — a candidate should not be
            penalized for skills the JD never asked about; only the
            skills the JD DID ask about, weighted by how well they
            assessed, should determine this score.
    """
    if not jd_skill_weights:
        return 0.0

    scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    if not scores:
        return 0.0

    matched_weight_sum = 0.0
    total_matched_weight = 0.0
    for skill_name, raw_score in scores.items():
        skill_lower = skill_name.lower()
        for keyword, weight in jd_skill_weights:
            kw_lower = keyword.lower()
            if kw_lower in skill_lower or skill_lower in kw_lower:
                matched_weight_sum += weight * (raw_score / 100.0)
                total_matched_weight += weight
                break  # one keyword match per skill is enough signal

    if total_matched_weight <= 0.0:
        return 0.0
    return min(1.0, matched_weight_sum / total_matched_weight)


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
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            ort_inputs = {name: inputs[name].astype(np.int64) for name in input_names if name in inputs}
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
            return_tensors="np",
        )
        q_ort_inputs = {name: q_inputs[name].astype(np.int64) for name in input_names if name in q_inputs}
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

    skill_scores = {
        c["candidate_id"]: score_skill_alignment(c, config.JD_SKILL_WEIGHTS)
        for c in candidates
    }

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
        key=lambda c: (-c["semantic_score"], c["candidate_id"]),
    )

    return sorted_result[:config.STAGE2_TOP_K]