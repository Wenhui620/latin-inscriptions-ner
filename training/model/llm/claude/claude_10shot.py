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
  • Do not use B-PERS for names that belong to known Roman name structures.

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
{"tokens": ["o", "Marci", "filio", "Aniensi", "aedili", "IIviro", "Montana", "viro", "hoc", "monumentum", "heredem", "non", "sequetur"], "tags": ["B-PERS", "B-PERS:FILI", "I-PERS:FILI", "B-PERS:COG", "B-TITLE", "O", "B-LOC", "O", "O", "O", "O", "O", "O"]}
{"tokens": ["Caius", "Vibius", "Polycarpus", "Caius", "Vibius", "Dorus", "Halus", "Tiberi", "Claudi", "Caesaris", "aedituus", "de", "aede", "Iovis", "porticus", "Octaviae"], "tags": ["B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:COG", "B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:COG", "B-PERS:AG", "B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:COG", "O", "O", "O", "O", "O", "O"]}
{"tokens": ["Imperatori", "Caesari", "divi", "Traiani", "Parthici", "filio", "divi", "Nervae", "nepoti"], "tags": ["B-PERS:TITLE", "B-PERS:TITLE", "B-PERS:FILI", "I-PERS:FILI", "I-PERS:FILI", "I-PERS:FILI", "B-PERS:FILI", "I-PERS:FILI", "I-PERS:FILI"]}
{"tokens": ["Caius", "Arestius", "Cai", "filius", "Claudia", "Firmus", "veteranus", "legionis", "XV", "Apollinaris", "annorum", "XL", "hic", "situs", "est", "Vettia", "Sabina", "coniunx", "et", "filii", "pro", "pietate"], "tags": ["B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:FILI", "I-PERS:FILI", "B-PERS:COG", "B-PERS:AG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-PERS:NOMEN", "B-PERS:COG", "O", "O", "O", "O", "O"]}
{"tokens": ["Aponia", "Festa", "Tito", "Aponio", "Cnaeo", "Marcio"], "tags": ["B-PERS:NOMEN", "B-PERS:COG", "B-PERS:NOMEN", "B-PERS:COG", "B-PERS:NOMEN", "B-PERS:COG"]}
{"tokens": ["Imperator", "Caesar", "divi", "Antonini", "Magni", "Pii", "filius", "divi", "Severi", "Pii", "nepos", "Marcus", "Aurellius", "Severus", "Alexander", "Pius", "Felix", "Augustus", "pontifex", "maximus", "tribunicia", "potestate", "III", "consul", "pater", "patriae", "nomina", "militum", "qui", "militaverunt", "in", "cohortibus", "urbanis", "Severianis", "quattuor", "X", "XI", "XII", "XIIII", "subieci", "qui", "bus", "fortiter", "et", "pie", "militia", "functi", "ssunt", "ius", "tribui", "conubii", "dumtaxat", "cum", "singulis", "et", "primis", "uxoribus", "ut", "etiamsi", "peregrini", "iuris", "feminas", "in", "matrimonio", "suo", "iunxerint", "proinde", "liberos", "tollant", "ac", "si", "ex", "duobus", "civibus", "Romanis", "natos", "ante", "diem", "VII", "Idus", "Ianuarias", "Appio", "Claudio", "Iuliano", "II", "Caio", "Bruttio", "Crispino", "consulibus", "cohors", "XI", "urbana", "Severiana", "Lucio", "Camelio", "Luci", "filio", "Palatina", "Severo", "PuOteolis", "Descriptum", "et", "recognitum", "ex", "tabula", "aerea", "quae", "fixa", "est", "Romae", "in", "muro", "post", "templum", "divi", "Augusti", "ad", "Minervam"], "tags": ["B-PERS:TITLE", "B-PERS:TITLE", "B-PERS:FILI", "I-PERS:FILI", "I-PERS:FILI", "I-PERS:FILI", "I-PERS:FILI", "B-PERS:FILI", "I-PERS:FILI", "I-PERS:FILI", "I-PERS:FILI", "B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:COG", "B-PERS:AG", "B-PERS:AG", "B-PERS:AG", "B-PERS:TITLE", "B-TITLE", "I-TITLE", "B-TITLE", "I-TITLE", "O", "B-TITLE", "B-TITLE", "I-TITLE", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:COG", "O", "B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:COG", "O", "O", "O", "O", "O", "B-PERS:PRAE", "B-PERS:NOMEN", "B-PERS:FILI", "I-PERS:FILI", "B-PERS:COG", "B-PERS:AG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
{"tokens": ["Mandulio", "Luci", "filio", "Teretina", "Crescenti", "aedili", "IIviro", "flamini", "Romae", "et", "Augustorum", "praefecto", "fabrum"], "tags": ["B-PERS:PRAE", "B-PERS:FILI", "I-PERS:FILI", "B-PERS:NOMEN", "B-PERS:COG", "B-TITLE", "O", "O", "O", "O", "O", "B-TITLE", "O"]}
{"tokens": ["ON", "memor", "eques", "sui", "Putiat", "liberta", "ex", "pugna", "legi", "dux", "Pope", "obiit", "MC"], "tags": ["O", "O", "O", "O", "B-PERS", "O", "O", "O", "O", "O", "O", "O", "O"]}
{"tokens": ["Dis", "Manibus", "Philadelphus", "Philandri", "Cappadox", "miles", "cohortis", "XXXII", "Voluntariorum", "centuria", "Ianuari", "annorum", "L", "stipendiorum", "XXX"], "tags": ["O", "O", "B-PERS:NOMEN", "B-PERS:COG", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}

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
    tags_output = os.path.join(base_dir, "training/model/llm/claude/10shot.json")
    
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