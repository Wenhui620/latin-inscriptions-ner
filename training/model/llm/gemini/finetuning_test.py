import os, json, re, time
from typing import List
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig

# export GOOGLE_CLOUD_PROJECT="latin_inscription_ner_v7"
# export GOOGLE_CLOUD_LOCATION="us-central1"
# export GOOGLE_GENAI_USE_VERTEXAI=True

client = genai.Client(http_options=HttpOptions(api_version="v1"))

ENDPOINT = "set_it_by_yourself"

SYSTEM_TEXT = """You are a Latin epigraphy Named Entity Recognition (NER) expert. Your task is to assign BIO tags to each token of Latin funerary inscriptions.

Use only the following entity tags:
- B-PERS:PRAE — Praenomen (personal name)
- B-PERS:NOMEN — Nomen (clan/family name)
- B-PERS:COG — Cognomen (family branch or nickname)
- B-PERS:FILI — Filiational terms (e.g., filius, libertus)
- B-PERS:AG — Agnomen (honorific)
- B-PERS:TITLE — Personal title like consul, pontifex
- B-TITLE — Official state/military/religious title
- B-LOC — Geographical names only (e.g., Roma, Tiberis)
- I-PERS, I-PERS:AG, I-PERS:TITLE — Inside-tag variants
- B-PERS — Only use this if:
  • The token is a standalone name, and
  • It cannot be confidently classified as PRAE, NOMEN, or COG.
  • Do **not** use B-PERS for names that belong to known Roman name structures.

Use the BIO format:  
- "B-" means beginning of an entity  
- "I-" means continuation  
- "O" means not an entity

⚠️ Output Format:
Return your output as exactly one valid JSON object, on a single line, structured as:
{"tokens": [...], "tags": [...]}

Constraints:
- The number of tags **must match exactly** the number of tokens.
- Do not include explanations or comments.
- Do not output Markdown, ellipses, or formatting.
- Only return the final JSON.

You will be given a list of tokens:

<tokens>
{{TOKENS}}
</tokens>

Label each token with its BIO tag. Be strict with format and tag rules.
"""

def _strip_code_fence(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t.strip("`")
        if t.endswith("```"):
            t = t[:-3].rstrip()
    return t

def _parse_tags_text(text: str) -> List[str]:
    t = _strip_code_fence(text or "").strip()
    if not t:
        return []
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "tags" in obj and isinstance(obj["tags"], list):
            return obj["tags"]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{\s*"tags"\s*:\s*(\[(?:.|\n)*?\])\s*\}', t, flags=re.S)
    if m:
        try:
            arr = json.loads(m.group(1))
            if isinstance(arr, list):
                return arr
        except Exception:
            return []
    return []

def _gen_cfg(expect_len: int) -> GenerateContentConfig:
    max_tokens = max(256, min(2048, expect_len * 8))
    return GenerateContentConfig(
        system_instruction=SYSTEM_TEXT,
        temperature=0.0,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
    )

def ner_bio_tagging(tokens: List[str], model_name: str, retries: int = 2) -> List[str]:
    user_prompt = (
        "Return JSON ONLY.\n"
        "The number of tags MUST equal the number of tokens.\n\n"
        "<tokens>\n" + json.dumps(tokens, ensure_ascii=False) + "\n</tokens>\n"
        'Return exactly: {"tags":[...]}'
    )
    cfg = _gen_cfg(len(tokens))
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=cfg
            )
            tags = _parse_tags_text(getattr(resp, "text", "") or "")
            if not tags or len(tags) != len(tokens):
                resp2 = client.models.generate_content(
                    model=model_name,
                    contents=(
                        user_prompt
                        + f"\n\nSTRICT:\n- Exactly {len(tokens)} tags.\n"
                          "- No prose. No markdown.\n"
                          '- Single JSON object: {"tags":[...]}\n'
                    ),
                    config=cfg
                )
                tags = _parse_tags_text(getattr(resp2, "text", "") or "")
            if not tags:
                tags = []
            if len(tags) < len(tokens):
                tags += ["O"] * (len(tokens) - len(tags))
            elif len(tags) > len(tokens):
                tags = tags[:len(tokens)]
            return _coerce_tags(tags)
        except Exception as e:
            last_err = e
            time.sleep(0.25 * (attempt + 1))
    print(f"[WARN] Gemini API call failed after retries: {last_err}")
    return ["O"] * len(tokens)

def tag_long_tokens(tokens: List[str], model_name: str, chunk: int = 20) -> List[str]:
    n = len(tokens)
    if n <= 40:
        return ner_bio_tagging(tokens, model_name=model_name)

    all_tags: List[str] = []
    for i in range(0, n, chunk):
        sub = tokens[i:i+chunk]
        sub_tags = ner_bio_tagging(sub, model_name=model_name)
        if sub_tags and sub_tags[0].startswith("I-"):
            sub_tags[0] = "B-" + sub_tags[0][2:]
        all_tags.extend(sub_tags)
        time.sleep(0.02)
    if len(all_tags) < n:
        all_tags += ["O"] * (n - len(all_tags))
    elif len(all_tags) > n:
        all_tags = all_tags[:n]

    all_tags = repair_bio(all_tags)
    return _coerce_tags(all_tags)

ALLOWED_TAGS = {
    "B-PERS","B-PERS:PRAE","B-PERS:NOMEN","B-PERS:COG","B-PERS:FILI","B-PERS:AG",
    "B-PERS:TITLE","B-TITLE","I-TITLE","B-LOC",
    "I-PERS","I-PERS:TITLE","I-PERS:AG","I-PERS:FILI",
    "O"
}

def _parse_json_first(text: str):
    if not text:
        return []
    try:
        obj = json.loads(text)
        return obj.get("tags", []) if isinstance(obj, dict) else []
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj.get("tags", []) if isinstance(obj, dict) else []
            except Exception:
                return []
        return []

def _fix_boundary(tag: str) -> str:
    if tag.startswith("I-"):
        return "B-" + tag[2:]
    return tag

def _coerce_tags(tags: List[str], allowed=ALLOWED_TAGS) -> List[str]:
    return [t if t in allowed else "O" for t in tags]

# --- CRF-style post-processing (heuristic BIO repair) ---
# Enforce valid IOB2 transitions with simple finite-state rules.
# If an entity segment starts with I- or switches type illegally, flip to B- of that type.
# Unknown/invalid tags are coerced to 'O'.

def repair_bio(tags: List[str], allowed=ALLOWED_TAGS) -> List[str]:
    fixed: List[str] = []
    prev_type = None  # type string after 'B-'/'I-' (e.g., 'PERS:COG')
    prev_inside = False  # whether previous tag was B- or I-

    for raw in tags:
        t = raw if raw in allowed else 'O'
        if t == 'O' or not isinstance(t, str):
            fixed.append('O')
            prev_type, prev_inside = None, False
            continue
        if t.startswith('B-'):
            cur_type = t[2:]
            fixed.append('B-' + cur_type)
            prev_type, prev_inside = cur_type, True
            continue
        if t.startswith('I-'):
            cur_type = t[2:]
            # Legal only if continuing the same type from a B-/I-
            if prev_inside and prev_type == cur_type:
                fixed.append('I-' + cur_type)
            else:
                # Start a new span instead of illegal I-
                fixed.append('B-' + cur_type)
            prev_type, prev_inside = cur_type, True
            continue
        # Any other pattern -> O
        fixed.append('O')
        prev_type, prev_inside = None, False
    return fixed

def _predict_one_prompt(user_text: str, expect_len: int, sys_text: str = SYSTEM_TEXT, max_tokens: int = 8192) -> List[str]:
    cfg = GenerateContentConfig(
        system_instruction=sys_text,
        temperature=0,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
    )
    hard_constraint = (
        f"\n\nSTRICT CONSTRAINTS:\n"
        f"- There are exactly {expect_len} tokens.\n"
        f"- Return exactly {expect_len} BIO tags.\n"
        f'- Output exactly one JSON object: {{"tags":[...]}} with length {expect_len}.\n'
        f"- No explanations, no markdown, no extra keys."
    )
    resp = client.models.generate_content(
        model=ENDPOINT,
        contents=user_text + hard_constraint,
        config=cfg
    )
    text = getattr(resp, "text", "") or ""
    tags = _parse_json_first(text)

    if not tags or len(tags) != expect_len:
        retry_prompt = (
            f"{user_text}\n\nONLY RETURN ONE JSON OBJECT EXACTLY AS:\n"
            f'{{"tags":[TAG_1, TAG_2, ..., TAG_{expect_len}]}}\n'
            f"No prose. No markdown. Length must be {expect_len}."
        )
        retry_resp = client.models.generate_content(
            model=ENDPOINT,
            contents=retry_prompt,
            config=cfg
        )
        text = getattr(retry_resp, "text", "") or ""
        tags = _parse_json_first(text)

    if not tags:
        tags = []
    if len(tags) < expect_len:
        tags = tags + ["O"] * (expect_len - len(tags))
    elif len(tags) > expect_len:
        tags = tags[:expect_len]

    # CRF-style legality repair before coercion
    tags = repair_bio(tags)
    return _coerce_tags(tags)

def predict_tags_for_tokens(tokens: List[str], chunk_size: int = 40, overlap: int = 5) -> List[str]:
    n = len(tokens)
    if n <= 60:
        user_text = (
            "You will be given a list of tokens from a Latin funerary inscription:\n"
            "<tokens>\n" + json.dumps(tokens, ensure_ascii=False) + "\n</tokens>\n"
            'Return JSON: {"tags":[...]} with the same length as tokens.'
        )
        return _predict_one_prompt(user_text, expect_len=n)

    final_tags = ["O"] * n
    step = max(1, chunk_size - overlap)
    idx = 0
    while idx < n:
        start = idx
        end = min(n, idx + chunk_size)
        chunk = tokens[start:end]
        user_text = (
            "You will be given a chunk of tokens from a longer Latin inscription:\n"
            "<tokens>\n" + json.dumps(chunk, ensure_ascii=False) + "\n</tokens>\n"
            'Return JSON: {"tags":[...]} with the same length as tokens.'
        )
        tags_chunk = _predict_one_prompt(user_text, expect_len=len(chunk))

        if tags_chunk:
            tags_chunk[0] = _fix_boundary(tags_chunk[0])

        final_tags[start:end] = tags_chunk
        if end == n:
            break
        idx += step

        time.sleep(0.02)

    if len(final_tags) < n:
        final_tags += ["O"] * (n - len(final_tags))
    elif len(final_tags) > n:
        final_tags = final_tags[:n]

    # CRF-style legality repair across concatenated chunks
    final_tags = repair_bio(final_tags)
    return _coerce_tags(final_tags)

TOKENS_ARRAY_REGEX = re.compile(r"\[(?:[^\[\]]|\\\[|\\\])*\]", flags=re.S)
TOKENS_INLINE_REGEX = re.compile(r"Tokens\s*[:=]\s*(\[(?:.|\n)*?\])", flags=re.I | re.S)

def extract_tokens_from_user_text(user_text: str):
    block_match = re.search(r"<tokens>(.*?)</tokens>", user_text, flags=re.S | re.I)
    candidate = None
    if block_match:
        candidate = block_match.group(1)
    else:
        candidate = user_text

    m_inline = TOKENS_INLINE_REGEX.search(candidate)
    if m_inline:
        try:
            arr = json.loads(m_inline.group(1))
            if isinstance(arr, list) and all(isinstance(t, str) for t in arr):
                return arr
        except Exception:
            pass

    m = TOKENS_ARRAY_REGEX.search(candidate)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        if isinstance(arr, list) and all(isinstance(t, str) for t in arr):
            return arr
    except Exception:
        return None
    return None

def main():
    input_path = "data/data_for_training/test_sft.jsonl"
    out_path = "training/model/llm/gemini/finetuning_gemini.jsonl"

    model_name = ENDPOINT

    y_true, y_pred = [], []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            ex = json.loads(line)
            gold_tags = json.loads(ex["contents"][1]["parts"][0]["text"])["tags"]
            n = len(gold_tags)
            user_text = ex["contents"][0]["parts"][0]["text"]
            tokens = extract_tokens_from_user_text(user_text)

            if tokens:
                pred_tags = predict_tags_for_tokens(tokens)
            else:
                user_text_min = (
                    user_text + "\n\n"
                    "Return JSON only: {\"tags\": [...]}.\n"
                    f"The number of tags must be exactly {n}."
                )
                pred_tags = _predict_one_prompt(user_text_min, expect_len=n)

            if i == 1:
                print("[DEBUG] first example] gold_len:", n, "pred_len:", len(pred_tags))
            if len(pred_tags) != len(gold_tags):
                print(f"[WARN] #{i} len mismatch: pred={len(pred_tags)} gold={len(gold_tags)}")

            y_true.append(gold_tags)
            y_pred.append(pred_tags)
            time.sleep(0.02)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"y_true": y_true, "y_pred": y_pred}, f, ensure_ascii=False, indent=2)
    print("[DONE sft_eval] examples=", len(y_true), "saved ->", out_path)

if __name__ == "__main__":
    main()