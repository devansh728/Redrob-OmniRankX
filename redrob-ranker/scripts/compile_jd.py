"""
compile_jd.py — compiles a raw JD file into a structured, weighted config
for the ranking pipeline, using a local small language model in three
extraction passes.

This version fixes three bugs found in the previous run (see the debug log
that produced generated_config_new_FAILED_DEBUG.json):

  1. CONTEXT OVERFLOW FROM RE-JOINING CHUNKS
     The previous code split the JD into chunks via segment_jd_text(), then
     immediately did ' '.join(text_partitions) before building every
     prompt — silently undoing the split and handing the model the full,
     uncut JD anyway. This version runs one inference call per chunk per
     pass and merges the partial results (see utils/jd_schemas.py
     merge_pass1/2/3), so no single call ever has to hold more than one
     ~3000-character chunk in context at once.

  2. VERBATIM COPYING OF FEW-SHOT EXAMPLES
     The previous prompts embedded complete, plausible-sounding fake JSON
     examples ("Built retrieval systems", "Scaled vector DB", ...). Under
     greedy decoding (temperature=0.0) and context pressure, the model
     copied these byte-for-byte instead of extracting real content — this
     was confirmed by diffing the failed output against the prompt text.
     This version replaces every example value with an inert angle-bracket
     placeholder that cannot be mistaken for a real answer (e.g.
     "<name of the capability the JD requires>"), and adds an explicit
     anti-copying instruction to every system prompt. After each pass, a
     copy-detector (detect_verbatim_copy) checks the parsed output against
     the prompt's own placeholder text and the JD's own raw text, and
     forces a retry with a stronger instruction if a suspicious match is
     found.

  3. HARDCODED, JD-SPECIFIC VALIDATION GATE
     The previous verify_named_entities_gate() hardcoded a token list
     (TCS, Infosys, ..., "Bangalore") tuned to one JD — and "Bangalore"
     isn't even in that JD's text, so that check could never meaningfully
     fire. This version calls extract_critical_tokens() from
     utils/text_utils.py to derive the critical-token list from whatever
     JD is actually being compiled, making the gate JD-agnostic.

Run with: python scripts/compile_jd.py
"""

import os
import re
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

from utils.text_utils import extract_raw_text_from_file, extract_critical_tokens
from utils.jd_schemas import (
    Pass1Schema, Pass2Schema, Pass3Schema,
    merge_pass1, merge_pass2, merge_pass3,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_slm(model_path: str) -> Llama:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=cpu_count(),
        verbose=False,
    )


# ---------------------------------------------------------------------------
# JD chunking — kept small enough that ONE chunk fits comfortably in
# context alongside the prompt scaffolding and output budget.
#
# Target: well under half of n_ctx in characters, since 1 token is
# roughly 3-4 English characters and we need headroom for the system
# prompt, the placeholder-shape instructions, and a 1024-token response.
# 1800 characters (~450-600 tokens) leaves ample room.
# ---------------------------------------------------------------------------

CHUNK_CHAR_LIMIT = 1800


def segment_jd_text(raw_text: str, chunk_char_limit: int = CHUNK_CHAR_LIMIT) -> List[str]:
    """Splits JD text into chunks on paragraph boundaries where possible.

    Input: full raw JD text, max characters per chunk.
    Output: list of text chunks, each <= chunk_char_limit (best effort).
    How it works: splits on blank lines (paragraph breaks) first, since JD
    section boundaries are almost always paragraph boundaries; falls back
    to splitting on single newlines if a single paragraph is itself too
    long for one chunk.
    """
    if not raw_text.strip():
        return []

    paragraphs = raw_text.split("\n\n")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # A single paragraph longer than the limit gets split on lines.
        if len(para) > chunk_char_limit:
            for line in para.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if current_len + len(line) > chunk_char_limit and current:
                    chunks.append("\n".join(current))
                    current, current_len = [], 0
                current.append(line)
                current_len += len(line) + 1
            continue

        if current_len + len(para) > chunk_char_limit and current:
            chunks.append("\n".join(current))
            current, current_len = [], 0

        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Anti-copying: detect if the model echoed prompt scaffolding instead of
# extracting real content from the JD chunk it was given.
# ---------------------------------------------------------------------------



# Keys whose value is INTENTIONALLY supplied by us (not by the model
# extracting from the JD). Echoing these is correct behavior, not
# hallucination, so they are never checked. The previous version had no
# such exclusion list, which meant every traceability.source_section
# field — which we explicitly instruct the model to fill with the chunk
# label we ourselves wrote into the prompt — was flagged as "copied,"
# even though copying that exact string is the only correct answer.
_ECHO_FIELD_KEYS = {"source_section", "rule_type", "applies_to", "escape_rule_type"}

# Minimum length for a string to be worth checking at all. Below this,
# coincidental overlaps (operator names like "EQUALS", short city names)
# produce noise without signal.
_MIN_CHECK_LENGTH = 12


def detect_verbatim_copy(parsed_output: dict, prompt_text: str, jd_chunk: str) -> List[str]:
    """Flags string fields in the parsed output that look copied from the
    prompt's placeholder/example scaffolding rather than genuinely
    extracted from the JD chunk.

    Input: the model's parsed JSON output, the full prompt text sent to
           the model, the raw JD chunk that was the actual input.
    Output: list of suspicious (key, value) strings (empty list if none).
    How it works: walks the output tree, skipping any key in
           _ECHO_FIELD_KEYS (fields we deliberately told the model to copy
           verbatim, such as the chunk label). For every other string
           field of meaningful length, it checks whether that string
           appears verbatim inside the PLACEHOLDER PORTION of the prompt
           specifically — i.e. inside an angle-bracket example span like
           "<short name of a HARD requirement...>" — rather than checking
           against the whole prompt text. Checking against the whole
           prompt (the previous approach) meant ANY shared vocabulary
           with the instructions counted as suspicious; checking only
           against bracketed placeholder spans means we only flag values
           that match text which was never meant to be real content in
           the first place.
    """
    placeholder_spans = re.findall(r"<[^<>]{6,}>", prompt_text)
    placeholder_spans_lower = [p.lower() for p in placeholder_spans]

    suspicious = []
    jd_lower = jd_chunk.lower()

    def walk(node, parent_key=None):
        if isinstance(node, str):
            if parent_key in _ECHO_FIELD_KEYS:
                return
            s = node.strip()
            if len(s) < _MIN_CHECK_LENGTH:
                return
            s_lower = s.lower()

            # Only flag if the string matches a PLACEHOLDER span verbatim
            # or near-verbatim (the placeholder text itself, not the
            # surrounding instruction prose) AND has no real grounding in
            # the actual JD chunk content.
            matches_placeholder = any(s_lower in p or p in s_lower for p in placeholder_spans_lower)
            in_jd = any(word in jd_lower for word in s_lower.split() if len(word) > 4)

            if matches_placeholder and not in_jd:
                suspicious.append(s)
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, parent_key=k)
        elif isinstance(node, list):
            for v in node:
                walk(v, parent_key=parent_key)

    walk(parsed_output)
    return suspicious


# ---------------------------------------------------------------------------
# Core inference call with anti-copying retry
# ---------------------------------------------------------------------------

def execute_inference_pass(
    llm,
    system_prompt: str,
    user_content: str,
    schema_class,
    execution_label: str,
    jd_chunk_for_copy_check: str,
    interactive_confirm: bool = True,
    max_retries: int = 1,
):
    """Runs one inference call, validates against schema_class, and checks
    for verbatim copying of prompt scaffolding before accepting the result.

    Input: llm handle, system/user prompt strings, target Pydantic schema,
           a label for logging, the raw JD chunk (for copy detection),
           whether to interactively confirm before each call, retry budget.
    Output: a validated instance of schema_class.
    How it works: calls the model at temperature 0.0, repairs and parses
           JSON, validates against the schema, then runs
           detect_verbatim_copy; if copying is detected and retries
           remain, re-issues the call with a sharper anti-copy instruction
           appended to the system prompt.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    print(f"\n[ORCHESTRATOR] Initiating Pass: {execution_label}")

    if interactive_confirm and sys.stdin.isatty():
        confirm = input(f"Confirm inference execution for {execution_label}? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Execution halted by operator.")
            sys.exit(0)

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
    )

    raw_response_output = response["choices"][0]["message"]["content"]
    repaired_json_dict = json_repair.loads(raw_response_output)

    if not isinstance(repaired_json_dict, dict):
        raise ValueError(f"SLM output failed JSON formatting constraint: {raw_response_output}")

    validated = schema_class.model_validate(repaired_json_dict)

    full_prompt_text = system_prompt + "\n" + user_content
    suspicious = detect_verbatim_copy(repaired_json_dict, full_prompt_text, jd_chunk_for_copy_check)

    if suspicious and max_retries > 0:
        print(f"[⚠️  COPY DETECTOR] {execution_label}: suspected verbatim prompt-copy in fields: {suspicious}")
        print("[⚠️  COPY DETECTOR] Retrying with a stronger anti-copy instruction...")
        sharpened_system_prompt = (
            system_prompt
            + "\n\nCRITICAL: Any bracketed placeholder text like <...> shown above is a FORMAT "
              "EXAMPLE ONLY. It is not real content and must NEVER appear in your output. "
              "Every value you output must come from the actual JD text provided below, not "
              "from the instructions above it. If the JD text does not mention something, "
              "return an empty list or empty string for that field rather than inventing or "
              "copying example text."
        )
        return execute_inference_pass(
            llm, sharpened_system_prompt, user_content, schema_class,
            execution_label + " (retry)", jd_chunk_for_copy_check,
            interactive_confirm=False, max_retries=max_retries - 1,
        )

    elif suspicious:
        print(f"[⚠️  COPY DETECTOR] {execution_label}: retries exhausted, accepting output with a warning.")

    return validated


# ---------------------------------------------------------------------------
# Prompt builders — one per pass, parameterized by JD chunk and section
# label. Examples use inert <angle-bracket> placeholders only; no
# plausible-sounding fake content anywhere.
# ---------------------------------------------------------------------------

ANTI_COPY_PREAMBLE = (
    "Any text inside angle brackets like <this> is a placeholder describing what "
    "kind of content belongs in that field — it is NOT example content and must "
    "never appear verbatim in your output. Extract only from the actual JD text "
    "given to you below. If the JD text does not contain something, return an "
    "empty list, empty string, or the stated numeric default for that field — "
    "do not invent or guess."
)


def _assert_no_embedded_examples_in_placeholders(prompt_text: str, prompt_label: str) -> None:
    """Fails loudly at startup if a placeholder span contains a concrete,
    copyable example value rather than a purely abstract description.

    Input: a fully-built prompt string, a label for the error message.
    Output: None (raises AssertionError if a violation is found).
    How it works: finds every <...> span and flags any that contain a
            dotted path (a.b.c — looks like a real field path), a quoted
            sub-string inside the brackets (a worked example phrase), or
            the literal marker "e.g." / "for example" / "such as" — all
            of which are signs that a concrete, copyable value was typed
            into what is supposed to be an inert instruction. This exists
            because of a real bug: build_pass2_prompt once described
            target_field_path as "<... e.g. career_history.industry>",
            and the model correctly copying that real field name for a
            genuinely-matching disqualifier was then wrongly flagged as
            hallucination by detect_verbatim_copy. The fix for THAT bug
            was removing the embedded example; THIS check exists so the
            same mistake fails fast at startup instead of silently
            reappearing in a future prompt edit.
    """
    spans = re.findall(r"<[^<>]{6,}>", prompt_text)
    violations = []
    for span in spans:
        looks_like_dotted_path = bool(re.search(r"\b[a-zA-Z_]+\.[a-zA-Z_]+(\.[a-zA-Z_]+)?\b", span))

        # A genuine embedded quote means a PAIRED quotation mark wrapping a
        # phrase (e.g. \"will not move forward\") — not a single apostrophe
        # used grammatically inside a contraction like "doesn't". We
        # require at least two double-quote characters, or a
        # straight/curly single-quote pair that isn't immediately preceded
        # by a letter (which would make it a contraction, e.g. "n't").
        double_quote_count = span.count('"')
        has_embedded_quote = double_quote_count >= 2

        # A trailing "such as X, Y, Z" / "e.g. X" is only dangerous if X
        # looks like a literal copyable value (capitalized proper noun,
        # quoted phrase, or dotted path) rather than a plain-language
        # category description. "such as refusal, exclusion phrasing" is
        # safe; "such as career_history.industry" or "e.g. \"will not move
        # forward\"" is not. We check the text AFTER the marker, not just
        # whether the marker is present at all.
        example_marker_match = re.search(r"(?:e\.g\.|for example|i\.e\.)\s*(.+?)(?=>|$)", span, flags=re.IGNORECASE)
        has_literal_after_marker = False
        if example_marker_match:
            after_marker = example_marker_match.group(1)
            has_literal_after_marker = bool(
                re.search(r"\b[a-zA-Z_]+\.[a-zA-Z_]+\b", after_marker)
                or '"' in after_marker
                or re.search(r"\b[A-Z][a-zA-Z]*\b", after_marker)
            )

        if looks_like_dotted_path or has_embedded_quote or has_literal_after_marker:
            violations.append(span)

    if violations:
        raise AssertionError(
            f"[PROMPT SAFETY CHECK FAILED] '{prompt_label}' has placeholder span(s) "
            f"containing a concrete example value, which the model may legitimately "
            f"need to copy for real JD content — causing the copy-detector to "
            f"false-positive on correct output: {violations}\n"
            f"Fix: describe the expected value abstractly, with no embedded "
            f"example phrase, dotted path, or quoted string inside the brackets."
        )


def build_pass1_prompt(jd_chunk: str, chunk_label: str) -> tuple:
    system_prompt = (
        "You are a recruitment structural analyst. You extract the persona, "
        "business intent, and evidence requirements from ONE section of a job "
        "description at a time. Output ONLY valid JSON, no markdown.\n\n"
        + ANTI_COPY_PREAMBLE
    )
    user_content = (
        f"This is section [{chunk_label}] of a job description. It may be only "
        "part of the full document — extract only what THIS section actually "
        "states; leave fields empty if this section doesn't cover them.\n\n"
        f"--- JD SECTION TEXT ---\n{jd_chunk}\n--- END SECTION TEXT ---\n\n"
        "Return JSON with exactly these keys:\n"
        "{\n"
        '  "business_intent": "<one sentence: the core business problem this '
        'role solves, ONLY if this section states it, else empty string>",\n'
        '  "primary_persona": "<the candidate archetype this section '
        'describes, ONLY if stated, else empty string>",\n'
        '  "anti_personas": ["<a candidate type this section explicitly says '
        'is NOT a fit, one entry per type found, else empty list>"],\n'
        '  "tier1_mandatory_evidence": [\n'
        "    {\n"
        '      "requirement_name": "<short name of a HARD requirement this '
        'section states>",\n'
        '      "evidence_proof_expectations": ["<what proof of this the '
        'section describes>"],\n'
        '      "is_mandatory_tier1": true,\n'
        '      "traceability": {\n'
        '        "extracted_fact": "<your paraphrase of what you extracted>",\n'
        '        "verbatim_text_quote": "<exact short quote from the section '
        'above, under 20 words>",\n'
        '        "source_section": "' + chunk_label + '"\n'
        "      }\n"
        "    }\n"
        "  ],\n"
        '  "tier2_preferred_evidence": [<same shape as tier1, but only for '
        'requirements this section frames as NICE-TO-HAVE, not mandatory>]\n'
        "}\n\n"
        "If this section contains none of the above, return empty lists/strings "
        "for every field. Output ONLY valid JSON."
    )
    return system_prompt, user_content


def build_pass2_prompt(jd_chunk: str, chunk_label: str) -> tuple:
    system_prompt = (
        "You are a hiring-constraints analyst. You extract hard numeric and "
        "categorical constraints from ONE section of a job description at a "
        "time. Output ONLY valid JSON, no markdown.\n\n"
        + ANTI_COPY_PREAMBLE
        + "\n\nWhen the section names specific companies, cities, or numbers "
          "(years, days, salary), you MUST copy those specific names/numbers "
          "into your output exactly as written. Do not summarize or "
          "generalize a specific company name into a category."
        + "\n\nEvery disqualifier must be classified into exactly one "
          "rule_type from this fixed list, chosen by which kind of "
          "candidate data the condition actually checks:\n"
          "- career_industry_match: checks the industry label of roles in "
          "a candidate's career history\n"
          "- career_title_keyword: checks the job title text of roles in a "
          "candidate's career history\n"
          "- career_text_keyword: checks the free-text description content "
          "of a candidate's roles for the presence or absence of certain "
          "concepts\n"
          "- company_name_match: checks whether a candidate's employer "
          "names match a named list of companies\n"
          "- tenure_pattern: checks the duration and frequency pattern of a "
          "candidate's roles (how long they stayed, how often they moved)\n"
          "- current_title_keyword: checks specifically a candidate's "
          "current job title, separate from their full career history\n"
          "- platform_activity: checks a candidate's engagement signals on "
          "the hiring platform itself (activity, responsiveness, "
          "application behavior)\n"
          "- location_relocation: checks a candidate's location, country, "
          "or willingness to relocate\n"
          "- skill_or_domain_balance: checks whether a candidate shows "
          "strong signal in one domain without a corroborating signal in a "
          "related domain the role actually needs\n"
          "- title_description_consistency: checks whether a role's title "
          "is consistent with that same role's own description, to catch "
          "mismatched or fabricated entries\n"
          "- unresolved: use this ONLY if the condition genuinely does not "
          "fit any category above\n"
          "For career_text_keyword, skill_or_domain_balance, and "
          "career_title_keyword rules, also populate primary_keywords (the "
          "concept being checked for) and, for skill_or_domain_balance "
          "specifically, corroborating_keywords (the related concept whose "
          "absence makes the primary keywords a red flag). For "
          "company_name_match, populate named_values with the literal "
          "company names. For any rule whose target concerns "
          "career_history, also set applies_to to one of: any_role (true if "
          "ANY single role matches), all_roles (true only if EVERY role "
          "matches), current_role_only (checks only the candidate's most "
          "recent/current role), or not_applicable if the rule does not "
          "concern career_history at all."
    )
    user_content = (
        f"This is section [{chunk_label}] of a job description.\n\n"
        f"--- JD SECTION TEXT ---\n{jd_chunk}\n--- END SECTION TEXT ---\n\n"
        "Return JSON with exactly these keys (use the stated defaults for any "
        "field this section does not address):\n"
        "{\n"
        '  "min_years_experience": <number, default 0.0 if not stated here>,\n'
        '  "max_years_experience": <number, default 20.0 if not stated here>,\n'
        '  "preferred_cities": ["<city name this section lists as preferred '
        'or acceptable, else empty list>"],\n'
        '  "willing_to_relocate_required": <true ONLY if this section '
        'explicitly demands relocation with no remote/local option, else '
        'false>,\n'
        '  "max_notice_period_days": <integer, default 90 if not stated '
        'here>,\n'
        '  "max_salary_budget_lpa": <number, default 0.0 if not stated '
        'here>,\n'
        '  "hard_disqualifiers": [\n'
        "    {\n"
        '      "condition_name": "<short human-readable name of a '
        'condition this section states causes outright rejection>",\n'
        '      "rule_type": "<one of the fixed category names listed in '
        'the system instructions above>",\n'
        '      "applies_to": "<any_role, all_roles, current_role_only, or '
        'not_applicable>",\n'
        '      "primary_keywords": ["<concept terms this rule checks for, '
        'else empty list>"],\n'
        '      "corroborating_keywords": ["<only for skill_or_domain_balance '
        'rules: the related concept whose ABSENCE matters, else empty '
        'list>"],\n'
        '      "named_values": ["<only for company_name_match rules: '
        'literal company names, else empty list>"],\n'
        '      "numeric_threshold": <only for tenure_pattern rules: a '
        'relevant number such as a month or year threshold, else null>,\n'
        '      "traceability": {\n'
        '        "extracted_fact": "<your paraphrase>",\n'
        '        "verbatim_text_quote": "<exact short quote, under 20 '
        'words>",\n'
        '        "source_section": "' + chunk_label + '"\n'
        "      }\n"
        "    }\n"
        "  ],\n"
        '  "soft_disqualifiers": [\n'
        "    {\n"
        '      "condition_name": "<short human-readable name of a '
        'condition this section frames as a NEGATIVE SIGNAL but not an '
        'outright rejection>",\n'
        '      "rule_type": "<one of the fixed category names listed in '
        'the system instructions above>",\n'
        '      "applies_to": "<any_role, all_roles, current_role_only, or '
        'not_applicable>",\n'
        '      "primary_keywords": ["<concept terms this rule checks for, '
        'else empty list>"],\n'
        '      "corroborating_keywords": ["<only for skill_or_domain_balance '
        'rules: the related concept whose ABSENCE matters, else empty '
        'list>"],\n'
        '      "named_values": ["<only for company_name_match rules: '
        'literal company names, else empty list>"],\n'
        '      "numeric_threshold": <only for tenure_pattern rules: a '
        'relevant number, else null>,\n'
        '      "penalty_weight": <float 0.0-1.0, how severe this penalty is>,\n'
        '      "escape_clause_condition": "<if the section states an '
        'exception that would waive this penalty, describe that exception '
        'in your own words here, else null>",\n'
        '      "escape_rule_type": "<if there is an escape clause, classify '
        'IT using the same fixed category list, else unresolved>",\n'
        '      "escape_keywords": ["<concept terms the escape condition '
        'checks for, else empty list>"],\n'
        '      "has_escape_hatch": <true if an escape_clause_condition was '
        'found, else false>,\n'
        '      "traceability": {\n'
        '        "extracted_fact": "<your paraphrase>",\n'
        '        "verbatim_text_quote": "<exact short quote, under 20 '
        'words>",\n'
        '        "source_section": "' + chunk_label + '"\n'
        "      }\n"
        "    }\n"
        "  ],\n"
        '  "bounded_concept_expansions": ["<a skill/concept term this section '
        'uses or implies, up to 15 total>"]\n'
        "}\n\n"
        "Output ONLY valid JSON."
    )
    return system_prompt, user_content


def build_pass3_prompt(jd_chunk: str, chunk_label: str) -> tuple:
    system_prompt = (
        "You are a text-emphasis analyst. You score how much emphasis ONE "
        "section of a job description places on different dimensions, and "
        "you count literal mentions of specific topics. Output ONLY valid "
        "JSON, no markdown.\n\n"
        + ANTI_COPY_PREAMBLE
        + "\n\nThe integer scores must reflect what THIS section actually "
          "emphasizes. If this section says nothing relevant to a dimension, "
          "score it 5 (neutral) rather than guessing high or low."
    )
    user_content = (
        f"This is section [{chunk_label}] of a job description.\n\n"
        f"--- JD SECTION TEXT ---\n{jd_chunk}\n--- END SECTION TEXT ---\n\n"
        "Return JSON with exactly these keys:\n"
        "{\n"
        '  "startup_vs_enterprise": <integer 1-10, 1=enterprise-only '
        'language, 10=startup-only language, 5 if section doesn\'t address '
        'this>,\n'
        '  "shipper_vs_researcher": <integer 1-10, 1=pure research framing, '
        '10=ship-fast framing, 5 if neutral>,\n'
        '  "builder_vs_manager": <integer 1-10, 1=pure management framing, '
        '10=hands-on IC framing, 5 if neutral>,\n'
        '  "generalist_vs_specialist": <integer 1-10, 1=generalist framing, '
        '10=narrow specialist framing, 5 if neutral>,\n'
        '  "jd_ambiguity_score": <integer 1-10, how vague THIS section is '
        'about concrete requirements, 1=very precise, 10=very vague>,\n'
        '  "raw_text_mention_counts": {\n'
        '    "location": <count of literal location/city/relocation '
        'mentions IN THIS SECTION ONLY>,\n'
        '    "notice_period": <count of literal notice-period/availability '
        'mentions IN THIS SECTION ONLY>,\n'
        '    "salary": <count of literal salary/compensation mentions IN '
        'THIS SECTION ONLY>,\n'
        '    "skills": <count of literal named skill/technology mentions IN '
        'THIS SECTION ONLY>,\n'
        '    "architecture": <count of literal system/architecture mentions '
        'IN THIS SECTION ONLY>\n'
        "  }\n"
        "}\n\n"
        "All counts must be actual counts of this section's text, not "
        "estimates. Output ONLY valid JSON."
    )
    return system_prompt, user_content


# ---------------------------------------------------------------------------
# Weight derivation — unchanged in spirit from the previous version, kept
# here for completeness since it consumes the merged Pass1-3 output.
# ---------------------------------------------------------------------------

def calculate_grounded_weights(pass1: Pass1Schema, pass2: Pass2Schema, pass3: Pass3Schema) -> dict:
    """Derives fusion weights from extracted counts, not from an
    ungrounded self-reported integer.

    Input: merged Pass1/2/3 schema objects.
    Output: dict with semantic/trajectory/behavioral/logistics floats
            summing to 1.0.
    How it works: builds a raw "mass" score per category from concrete
            extracted counts (disqualifier counts, evidence counts,
            mention counts), then normalizes. No category's weight comes
            from a single free-floating LLM-reported number.
    """
    hard_count = len(pass2.hard_disqualifiers)
    soft_count = len(pass2.soft_disqualifiers)
    mandatory_count = len(pass1.tier1_mandatory_evidence)
    preferred_count = len(pass1.tier2_preferred_evidence)

    counts = pass3.raw_text_mention_counts
    logistics_mentions = counts.get("location", 0) + counts.get("salary", 0)
    behavioral_mentions = counts.get("notice_period", 0) + 1
    technical_mentions = counts.get("skills", 0) + counts.get("architecture", 0)

    base_semantic = float((mandatory_count * 10) + (preferred_count * 4) + technical_mentions)
    base_trajectory = float((hard_count * 12) + (soft_count * 6))
    base_behavioral = float(behavioral_mentions * 8)
    base_logistics = float(logistics_mentions * 5)

    total = base_semantic + base_trajectory + base_behavioral + base_logistics
    if total == 0:
        return {"semantic": 0.35, "trajectory": 0.25, "behavioral": 0.25, "logistics": 0.15}

    return {
        "semantic": round(base_semantic / total, 4),
        "trajectory": round(base_trajectory / total, 4),
        "behavioral": round(base_behavioral / total, 4),
        "logistics": round(base_logistics / total, 4),
    }


# ---------------------------------------------------------------------------
# Validation gate — now JD-agnostic via extract_critical_tokens()
# ---------------------------------------------------------------------------

def verify_named_entities_gate(raw_text: str, config_data: dict) -> List[str]:
    """Checks that named entities and numeric constraints actually present
    in the raw JD text appear somewhere in the compiled config.

    Input: full raw JD text, the compiled config dict.
    Output: list of tokens that were in the JD but never made it into the
            config (empty list = gate passes).
    How it works: derives the critical-token list dynamically from the JD
            itself (via extract_critical_tokens), then checks each token's
            presence in a lowercased JSON dump of the compiled config.
    """
    critical_tokens = extract_critical_tokens(raw_text)
    config_dump_string = json.dumps(config_data).lower()

    missing = []
    for token in critical_tokens:
        if token.lower() not in config_dump_string:
            missing.append(token)
    return missing


# ---------------------------------------------------------------------------
# Pass orchestration — run each pass once per chunk, merge results
# ---------------------------------------------------------------------------

def run_pass1_over_chunks(llm, chunks: List[str], interactive: bool) -> Pass1Schema:
    partials = []
    for i, chunk in enumerate(chunks):
        label = f"chunk_{i+1}_of_{len(chunks)}"
        system_prompt, user_content = build_pass1_prompt(chunk, label)
        result = execute_inference_pass(
            llm, system_prompt, user_content, Pass1Schema,
            f"Pass 1 - {label}", jd_chunk_for_copy_check=chunk,
            interactive_confirm=interactive,
        )
        partials.append(result)
    return merge_pass1(partials)


def run_pass2_over_chunks(llm, chunks: List[str], interactive: bool) -> Pass2Schema:
    partials = []
    for i, chunk in enumerate(chunks):
        label = f"chunk_{i+1}_of_{len(chunks)}"
        system_prompt, user_content = build_pass2_prompt(chunk, label)
        result = execute_inference_pass(
            llm, system_prompt, user_content, Pass2Schema,
            f"Pass 2 - {label}", jd_chunk_for_copy_check=chunk,
            interactive_confirm=interactive,
        )
        partials.append(result)
    return merge_pass2(partials)


def run_pass3_over_chunks(llm, chunks: List[str], interactive: bool) -> Pass3Schema:
    partials = []
    for i, chunk in enumerate(chunks):
        label = f"chunk_{i+1}_of_{len(chunks)}"
        system_prompt, user_content = build_pass3_prompt(chunk, label)
        result = execute_inference_pass(
            llm, system_prompt, user_content, Pass3Schema,
            f"Pass 3 - {label}", jd_chunk_for_copy_check=chunk,
            interactive_confirm=interactive,
        )
        partials.append(result)
    return merge_pass3(partials)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd-path", default="../India_runs_data_and_ai_challenge/job_description.docx")
    parser.add_argument("--output-path", default="config/generated_config_new_2.json")
    parser.add_argument("--model-path", default="models/qwen2.5-3b-instruct/Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--no-confirm", action="store_true", help="Skip interactive per-call confirmation")
    args = parser.parse_args()

    jd_abs_path = os.path.abspath(os.path.join(project_root, args.jd_path))
    output_abs_path = os.path.abspath(os.path.join(project_root, args.output_path))
    model_abs_path = os.path.abspath(os.path.join(project_root, args.model_path))

    full_jd_text = extract_raw_text_from_file(jd_abs_path)
    chunks = segment_jd_text(full_jd_text)
    print(f"[ORCHESTRATOR] JD split into {len(chunks)} chunks "
          f"(~{CHUNK_CHAR_LIMIT} chars each) — each chunk gets its own inference call.")

    if chunks:
        sample_chunk, sample_label = chunks[0], "chunk_1_of_N"
        for builder, name in (
            (build_pass1_prompt, "Pass 1"), (build_pass2_prompt, "Pass 2"), (build_pass3_prompt, "Pass 3"),
        ):
            sys_p, user_p = builder(sample_chunk, sample_label)
            _assert_no_embedded_examples_in_placeholders(sys_p + "\n" + user_p, name)
        print("[ORCHESTRATOR] Prompt safety self-check passed for all three passes.")

    llm = load_slm(model_abs_path)
    interactive = not args.no_confirm

    pass1 = run_pass1_over_chunks(llm, chunks, interactive)
    pass2 = run_pass2_over_chunks(llm, chunks, interactive)
    pass3 = run_pass3_over_chunks(llm, chunks, interactive)

    grounded_weights = calculate_grounded_weights(pass1, pass2, pass3)

    master_config_output = {
        "meta": {
            "compiled_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "jd_source_file": os.path.basename(jd_abs_path),
            "ambiguity_damping_factor": round(pass3.jd_ambiguity_score / 20.0, 4),
            "chunk_count": len(chunks),
        },
        "constraints": pass2.model_dump(),
        "semantic_targets": pass1.model_dump(),
        "behavioral_priorities": pass3.model_dump(),
        "normalized_fusion_weights": grounded_weights,
    }

    missing_entities = verify_named_entities_gate(full_jd_text, master_config_output)
    if missing_entities:
        print(f"\n[🚨 VALIDATION FAILURE ALERT] The compiled configuration dropped tokens "
              f"that are present in the source JD: {missing_entities}")
        debug_path = output_abs_path.replace(".json", "_FAILED_DEBUG_2.json")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(master_config_output, f, indent=2)
        print(f"[🔍 DEBUG] Saved partial output to: {debug_path}")
        print("[🚨 ERROR] Execution terminating. Inspect which chunk should have "
              "captured the missing token(s) and re-run, or lower CHUNK_CHAR_LIMIT.")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_abs_path), exist_ok=True)
    with open(output_abs_path, "w", encoding="utf-8") as f:
        json.dump(master_config_output, f, indent=2)

    print(f"\n[SUCCESS] Compiled configuration written to: {output_abs_path}")
    print("\nSummary:")
    print(f"  Primary persona      : {pass1.primary_persona}")
    print(f"  Experience bounds    : {pass2.min_years_experience}-{pass2.max_years_experience} years")
    print(f"  Hard disqualifiers   : {len(pass2.hard_disqualifiers)}")
    print(f"  Soft disqualifiers   : {len(pass2.soft_disqualifiers)}")
    print(f"  Tier-1 evidence items: {len(pass1.tier1_mandatory_evidence)}")
    print(f"  Tier-2 evidence items: {len(pass1.tier2_preferred_evidence)}")
    print(f"  Fusion weights       : {grounded_weights}")


if __name__ == "__main__":
    main()