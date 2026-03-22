import os
import json
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def ner_bio_tagging(text):
    system_prompt = """You are a Latin epigraphy Named Entity Recognition (NER) expert. Your task is to assign BIO tags to each token of Latin funerary inscriptions.

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

Output Format:
Return your output as exactly one valid JSON object, on a single line, structured as:
{"tokens": [...], "tags": [...]}

Constraints:
- The number of tags **must match exactly** the number of tokens.
- Do not include explanations or comments.
- Do not output Markdown, ellipses, or formatting.
- Only return the final JSON.

Here are some examples of correctly labeled Latin inscriptions in the JSON line format:
{"tokens": ["qui", "et", "gni", "in", "pace", "positus"], "tags": ["O", "O", "O", "O", "O", "O"]}

Label each token with its BIO tag. Be strict with format and tag rules.
"""    
    user_prompt = f"Text: {text}\nTags:"
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",  
            max_tokens=2048,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        resp_text = response.content[0].text.strip()
        
        return resp_text
        
    except Exception as e:
        print(f"Error in API call: {e}")
        return ""

if __name__ == "__main__":
    base_dir = "latin_inscription"
    test_file = os.path.join(base_dir, "data/data_for_training/test.json")
    tags_output = os.path.join(base_dir, "training/model/llm/claude/1shot.json")
    
    os.makedirs(os.path.dirname(tags_output), exist_ok=True)
    
    open(tags_output, "w", encoding="utf-8").close()

    try:
        with open(test_file, encoding="utf-8") as f:
            test_data = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading test data: {e}")
        exit(1)

    with open(tags_output, "w", encoding="utf-8") as tags_f:
        for example in test_data:
            tokens = example.get("tokens", [])
            total = len(tokens)
            if total == 0:
                continue
            if total <= 40:
                tag_line = ner_bio_tagging(tokens)
                parsed = json.loads(tag_line)
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
                    tag_line = ner_bio_tagging(chunk)
                    parsed = json.loads(tag_line)
                    if isinstance(parsed, dict) and "tags" in parsed:
                        tags_list = parsed["tags"]
                    elif isinstance(parsed, list):
                        tags_list = parsed
                    else:
                        raise ValueError(f"Unexpected NER output format: {parsed!r}")
                    all_tags.extend(tags_list)
                result = {"tokens": tokens, "tags": all_tags}
                tags_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Processing completed. Tag lines written to:", tags_output)