# Latin Inscription NER

## Overview

This project focuses on Named Entity Recognition (NER) for Latin inscriptions.  
The goal is to assign BIO tags (e.g., person names, titles, locations) to each token in Latin epigraphic texts.

The project explores multiple approaches, including:
- Traditional machine learning (SVM)
- Neural models (with and without dependency features)
- Transformer-based models (LatinBERT)
- Large language models (LLMs)

---

## Project Structure

### data/
- raw_data: original and cleaned Latin inscription data  
- annotated_data_1000: manually annotated subset  
- data_for_training: train / validation / test datasets in csv, jsonl, and spacy formats  

### training/model/
- latinBERT: transformer-based model  
- llm: LLM-based tagging  
- spacy: spaCy training pipeline  
- svm: traditional machine learning model  
- with_dependency_tree: neural model with dependency features  
- without_dependency_tree: neural model without dependency features  

### requirement.txt  
- Python dependencies  

---

## Annotation Scheme

The project uses the BIO tagging scheme.

### Tag Definitions

- B-PERS:PRAE — Praenomen (personal name)  
- B-PERS:NOMEN — Nomen (clan/family name)  
- B-PERS:COG — Cognomen (family branch or nickname)  
- B-PERS:FILI — Filiational terms (e.g., filius, libertus)  
- B-PERS:AG — Agnomen (honorific)  
- B-PERS:TITLE — Personal title like consul, pontifex  
- B-TITLE — Official state/military/religious title  
- B-LOC — Geographical names only (e.g., Roma, Tiberis)  
- I-PERS, I-PERS:AG, I-PERS:TITLE — Inside-tag variants  

### Special Rule

- B-PERS is only used if:
  - The token is a standalone name, and  
  - It cannot be confidently classified as PRAE, NOMEN, or COG  
  - Do not use B-PERS for names that belong to known Roman name structures  

---

## Usage

Install dependencies:

pip install -r requirement.txt

Then run training or evaluation scripts from the corresponding model folders under training/model/.