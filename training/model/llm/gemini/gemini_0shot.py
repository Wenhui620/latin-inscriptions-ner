import os
import json
from google import genai
from google.genai import types

# Initialize Gemini client with API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ner_bio_tagging(text):
    system_prompt = """You are a Named Entity Recognition (NER) expert specialized in Latin epigraphy. Your task is to label Latin inscriptions using the BIO (Beginning, Inside, Outside) tagging format.
 
Use ONLY the following entity tags:
 
- B-PERS:PRAE — Praenomen: given or personal name
- B-PERS:NOMEN — Nomen: family or clan name (from a common ancestor)
- B-PERS:COG — Cognomen: branch family name or personal alias
- B-PERS:FILI — Filiational terms, including family relationships and birth order indicators (e.g., "filius", "libertus")
- B-TITLE, B-PERS:TITLE — Official Roman titles (must match Wikipedia’s “Ancient Roman titles”)
- B-PERS:AG — Agnomen: honorific nicknames earned through achievements
- B-LOC — Locations: only standalone geographical names (e.g., cities, provinces, rivers); ignore locations embedded within titles or military units
- I-PERS, I-PERS:TITLE, I-PERS:AG — Inside-tag variants
- O — Not an entity

The BIO tagging format works as follows:

- B- prefix indicates the beginning of an entity
- I- prefix indicates the continuation (inside) of an entity
- O tag is used for tokens that are not part of any named entity

You will be given a list of tokens from a Latin funerary inscription:
 
<tokens>
{{TOKENS}}
</tokens>

Your task is to assign one BIO tag to each token in the input, maintaining the same order. Process the tokens sequentially, considering the context and grammatical structure of Latin names and inscriptions.
 
Return your results in a JSON format with two keys: "tokens" (the input tokens) and "tags" (your BIO tags). For example, if the input tokens were ["XYZ", "XYZ", "XYZ"], your output should look like this:

{"tokens": ["XYZ", "XYZ", "XYZ"], "tags": [tag1, tag2, tag3]}

Remember to use ONLY the entity tags provided and adhere strictly to the BIO format. Do not introduce any new tags or modify the existing ones. The number of tags MUST match the number of tokens exactly. Do not return partial or truncated tag sequences.
"""
    user_prompt = f"Text: {text}\nTags:"

    import time
    try:
        start = time.time()
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=system_prompt + "\n\n" + user_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        print(f"Gemini returned in {time.time() - start:.2f}s")
        return response.text.strip()
    except Exception as e:
        print("Gemini API call failed:", e)
        return ""

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "latin_inscription"))
    test_file = os.path.join(base_dir, "data/data_for_training/test.json")
    tags_output = os.path.join(base_dir, "training/model/llm/gemini/0shot.json")

    os.makedirs(os.path.dirname(tags_output), exist_ok=True)
    open(tags_output, "w", encoding="utf-8").close()

    with open(test_file, encoding="utf-8") as f:
        content = f.read()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            test_data = data
        elif isinstance(data, dict):
            test_data = [data]
        else:
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        test_data = [json.loads(line) for line in content.splitlines() if line.strip()]

    with open(tags_output, "w", encoding="utf-8") as tags_f:
        for example in test_data:
            tokens = example.get("tokens", [])
            total = len(tokens)
            if total == 0:
                continue
            if total <= 35:
                tag_line = ner_bio_tagging(tokens).strip()
                if not tag_line:
                    print("⚠️ Empty response, skipping:", tokens)
                    continue
                try:
                    tag_line = tag_line.strip()
                    if tag_line.startswith("```json"):
                        tag_line = tag_line[len("```json"):].strip()
                    elif tag_line.startswith("```"):
                        tag_line = tag_line.strip("` \n")
                    elif tag_line.startswith("json\n") or tag_line.startswith("json\r\n"):
                        tag_line = tag_line.split("\n", 1)[1].strip()
                    if tag_line.endswith("```"):
                        tag_line = tag_line.rsplit("```", 1)[0].strip()
                    parsed = json.loads(tag_line)
                except json.JSONDecodeError:
                    print("JSON parse failed. Raw response:", repr(tag_line))
                    continue
                if isinstance(parsed, dict) and "tags" in parsed:
                    output = parsed
                elif isinstance(parsed, list):
                    output = {"tokens": tokens, "tags": parsed}
                else:
                    raise ValueError(f"Unexpected NER output format: {parsed!r}")
                tags_f.write(json.dumps(output, ensure_ascii=False) + "\n")
            else:
                all_tags = []
                for i in range(0, total, 20):
                    chunk = tokens[i:i+20]
                    tag_line = ner_bio_tagging(chunk).strip()
                    if not tag_line:
                        print("Empty response, skipping chunk:", chunk)
                        continue
                    try:
                        tag_line = tag_line.strip()
                        if tag_line.startswith("```json"):
                            tag_line = tag_line[len("```json"):].strip()
                        elif tag_line.startswith("```"):
                            tag_line = tag_line.strip("` \n")
                        elif tag_line.startswith("json\n") or tag_line.startswith("json\r\n"):
                            tag_line = tag_line.split("\n", 1)[1].strip()
                        if tag_line.endswith("```"):
                            tag_line = tag_line.rsplit("```", 1)[0].strip()
                        parsed = json.loads(tag_line)
                    except json.JSONDecodeError:
                        print("JSON parse failed. Raw response:", repr(tag_line))
                        continue
                    if isinstance(parsed, dict) and "tags" in parsed:
                        tags_list = parsed["tags"]
                    elif isinstance(parsed, list):
                        tags_list = parsed
                    else:
                        raise ValueError(f"Unexpected NER output format: {parsed!r}")
                    all_tags.extend(tags_list)
                result = {"tokens": tokens, "tags": all_tags}
                tags_f.write(json.dumps(result, ensure_ascii=False) + "\n")
