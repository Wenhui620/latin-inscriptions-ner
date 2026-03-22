import openai
import os
import json


openai.api_key = os.getenv("OPENAI_API_KEY")

def ner_bio_tagging(tokens) -> list:
    text = " ".join(tokens)
    
    messages = [
        {
            "role": "system",
            "content": """You are a Named Entity Recognition (NER) expert specialized in Latin epigraphy. Your task is to label Latin inscriptions using the BIO (Beginning, Inside, Outside) tagging format.
 
Use ONLY the following entity tags:
 
- B-PERS:PRAE — Praenomen: given or personal name
- B-PERS:NOMEN — Nomen: family or clan name (from a common ancestor)
- B-PERS:COG — Cognomen: branch family name or personal alias
- B-PERS:FILI — Filiational terms, including family relationships and birth order indicators (e.g., "filius", "libertus")
- B-TITLE, B-PERS:TITLE — Official Roman titles (must match Wikipedia’s “Ancient Roman titles”)
- B-PERS:AG — Agnomen: honorific nicknames earned through achievements
- B-PERS — General person name. Use `B-PERS` in the following situations:
  • When a token clearly refers to a personal name but does not fit into any identifiable name structure, such as:
    - Praenomen-Nomen-Cognomen (PRAE-NOMEN-COG)
    - Nomen-Cognomen (NOMEN-COG)
    - Imperial title sequences
  • When the name is a single standalone token and no additional name components are provided.
If a full structure is present, label components specifically using the provided subtypes.
- B-LOC — Locations: only standalone geographical names (e.g., cities, provinces, rivers); ignore locations embedded within titles or military units 
- I-PERS, I-PERS:TITLE, I-PERS:AG — Inside-tag variants
- O — Not an entity

The BIO tagging format works as follows:

- B- prefix indicates the beginning of an entity
- I- prefix indicates the continuation (inside) of an entity
- O tag is used for tokens that are not part of any named entity
 
Here are some examples of correctly labeled Latin inscriptions in the JSON line format:
 
{"tokens": ["qui", "et", "gni", "in", "pace", "positus"], "tags": ["O", "O", "O", "O", "O", "O"]}
{"tokens": ["o", "Marci", "filio", "Aniensi", "aedili", "IIviro", "Montana", "viro", "hoc", "monumentum", "heredem", "non", "sequetur"], "tags": ["B-PERS", "B-PERS:FILI", "I-PERS:FILI", "B-PERS:COG", "B-TITLE", "O", "B-LOC", "O", "O", "O", "O", "O", "O"]}
{"tokens": ["Caius", "Vibius", "Polycarpus", "Caius", "Vibius", "Dorus", "Halus", "Tiberi", "Claudi", "Caesaris", "aedituus", "de", "aede", "Iovis", "porticus", "Octaviae"], "tags": ["B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:COG", "B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:COG", "B-PERS:AG", "B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:COG", "O", "O", "O", "O", "O", "O"]}
{"tokens": ["Imperatori", "Caesari", "divi", "Traiani", "Parthici", "filio", "divi", "Nervae", "nepoti"], "tags": ["B-PERS:TITLE", "B-PERS:TITLE", "B-PERS:FILI", "I-PERS:FILI", "I-PERS:FILI", "I-PERS:FILI", "B-PERS:FILI", "I-PERS:FILI", "I-PERS:FILI"]}
{"tokens": ["Caius", "Arestius", "Cai", "filius", "Claudia", "Firmus", "veteranus", "legionis", "XV", "Apollinaris", "annorum", "XL", "hic", "situs", "est", "Vettia", "Sabina", "coniunx", "et", "filii", "pro", "pietate"], "tags": ["B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:FILI", "I-PERS:FILI", "B-PERS:COG", "B-PERS:AG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-PERS:NOMEN", "B-PERS:COG", "O", "O", "O", "O", "O"]}
 
Your task is to assign one BIO tag to each token in the input, maintaining the same order. Process the tokens sequentially, considering the context and grammatical structure of Latin names and inscriptions.
 
Return your results in a JSON format with two keys: "tokens" (the input tokens) and "tags" (your BIO tags). For example, if the input tokens were ["XYZ", "XYZ", "XYZ"], your output should look like this:

{ "tokens": ["XYZ", "XYZ", "XYZ"],"tags": [tag1, tag2, tag3]}

Remember to use ONLY the entity tags provided and adhere strictly to the BIO format. Do not introduce any new tags or modify the existing ones. The number of tags MUST match the number of tokens exactly. Do not return partial or truncated tag sequences.
"""
        },
        {
            "role": "user",
            "content": f"Text: {text}\nTags:"
        }
    ]
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
        max_tokens=2048
    )
    resp_text = response.choices[0].message.content.strip()
    return resp_text


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "latin_inscription"))
    test_file = os.path.join(base_dir, "data/data_for_training/test.json")
    tags_output = os.path.join(base_dir, "training/model/llm/gpt4/5shot.json")

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

    # Generate and save tag lines
    with open(tags_output, "w", encoding="utf-8") as tags_f:
        for example in test_data:
            tokens = example.get("tokens", [])
            if not tokens:
                continue
            text = " ".join(tokens)
            tag_line = ner_bio_tagging(tokens)
            tags_f.write(tag_line + "\n")

    print("Processing completed. Tag lines written to:", tags_output)