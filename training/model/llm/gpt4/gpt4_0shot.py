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
 
- B-PERS:PRAE — Praenomen (first name)
- B-PERS:NOMEN — Nomen (family name)
- B-PERS:COG — Cognomen (personal surname)
- B-PERS:FILI — Filiational information (e.g., "filius")
- B-TITLE, B-PERS:TITLE — Titles and honorifics
- B-PERS:AG — Social roles (e.g., "militi")
- B-LOC — Locations
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
 
Return your results in a JSON format with two keys: "tokens" (the input tokens) and "tags" (your BIO tags). 
 
For example, if the input tokens were ["XYZ", "XYZ", "XYZ"], your output should look like this:
 
{
  "tokens": ["XYZ", "XYZ", "XYZ"],
  "tags": [tag1, tag2, tag3]
}
 
Remember to use ONLY the entity tags provided and adhere strictly to the BIO format. Do not introduce any new tags or modify the existing ones.
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
    tags_output = os.path.join(base_dir, "training/model/llm/gpt4/0shot.json")

    os.makedirs(os.path.dirname(tags_output), exist_ok=True)
    # Initialize tags file
    open(tags_output, "w", encoding="utf-8").close()

    # Load test examples
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