# Latin Inscription NER

## Overview

This project focuses on Named Entity Recognition (NER) for Latin inscriptions.  
The goal is to assign BIO tags (e.g., person names, titles, locations) to each token in Latin epigraphic texts.

The project explores multiple approaches, including:
- Traditional machine learning (SVM)
- Neural models (with and without dependency features)
- Transformer-based models (BERT)
- Large language models (LLMs)

---

## Project Structure

### data/
- raw_data: original and cleaned Latin inscription data  
- annotated_data_1000: manually annotated subset  
- data_for_training: train / validation / test datasets in csv, jsonl, and spacy formats  

### training/model/
- BERT: Multilingual BERT model  
- llm: LLM-based tagging  
- spacy+latinroBERTa: spaCy training pipeline with Latin RoBERTa
- svm: traditional machine learning model  
- with_dependency_tree: neural model with dependency features  
- without_dependency_tree: neural model without dependency features  

### requirement.txt  
- Python dependencies  

---

## Annotation Scheme

This project adopts an extended BIO tagging scheme tailored for Latin epigraphy, combining structural rules for Roman names with general NER boundary principles.

---

### 1. General Boundary Rules

- **Annotate only the core entity span**:
  - Exclude punctuation (commas, brackets, quotation marks) unless part of abbreviations.
  - Exclude determiners and modifiers not intrinsic to the entity.
- **Split coordinated entities**:
  - e.g., “Roma et Athenae” → two separate LOC entities.
- **Nested structure (important for Roman names)**:
  - Full name = outer span
  - Internal components (PRAE, NOMEN, etc.) = sub-spans
- **Consistency rule**:
  - The same entity type should be annotated consistently across the dataset.
- **Location entities (LOC) should refer only to place names (toponyms)**:
  - Do not annotate demonyms or people derived from locations.
  - e.g., *Athenienses* (“Athenians”) should NOT be labeled as LOC.
  - Only annotate the geographical noun itself (e.g., *Athenae*).

---

### 2. Tag Definitions (Extended)

| Tag | Description |
|-----|------------|
| PERS | Complete personal name (may include titles for emperors/nobility) |
| PRAE | Praenomen (given name) |
| NOMEN | Nomen (clan/family name) |
| COG | Cognomen (branch or nickname) |
| AG | Agnomen (honorific name) |
| FILI | Filiational markers (e.g., *filius*, *libertus*) |
| TITLE | Official or social titles |
| LOC | Geographical locations (cities, provinces, rivers) |

---

### 3. BIO Label Mapping

- B-PERS:PRAE — Praenomen  
- B-PERS:NOMEN — Nomen  
- B-PERS:COG — Cognomen  
- B-PERS:AG — Agnomen  
- B-PERS:FILI — Filiational term  
- B-PERS:TITLE — Title within a person name  
- B-TITLE — Standalone title  
- B-LOC — Location  
- I-* — Continuation of the same entity  

---

### 4. Tagging Priority Rules

1. **Annotate full names first (PERS)**, then internal components:
   - PRAE, NOMEN, COG, AG, FILI, TITLE
2. **Imperial titles** (e.g., Augustus, Caesar):
   - Part of PERS, but internally labeled as TITLE
3. **Standalone titles**:
   - e.g., *consul*, *pontifex* → B-TITLE
4. **Family relations and status markers**:
   - e.g., *filius*, *libertus* → FILI

---

### 5. Special Rules for Roman Names

- Use **B-PERS only when**:
  - The name is standalone  
  - Cannot be decomposed into PRAE/NOMEN/COG  
- Do **not** use B-PERS if Roman naming structure is identifiable.
- Titles may appear:
  - Inside names → PERS:TITLE  
  - Independently → TITLE  

---

### 6. Special Notes

- TITLE must correspond to historically valid Roman titles.
- FILI includes:
  - kinship terms  
  - freedman/slave indicators  
- Slaves/Freedmen:
  - FILI should be linked to the associated PERS when possible.

---

## Model Evaluation

This section reports the performance of different models on the test set.

---

### SVM (Linguistic Features)
```
              precision    recall  f1-score   support

       B-LOC     0.8000    0.2105    0.3333        19
      B-PERS     0.5070    0.4286    0.4645        84
   B-PERS:AG     0.6818    0.5769    0.6250        52
  B-PERS:COG     0.7576    0.7109    0.7335       211
 B-PERS:FILI     0.8219    0.8108    0.8163        74
B-PERS:NOMEN     0.7909    0.7873    0.7891       221
 B-PERS:PRAE     0.9107    0.8571    0.8831       119
B-PERS:TITLE     0.7353    0.8333    0.7812        60
B-PERS:TRIBE     1.0000    0.6667    0.8000         9
     B-TITLE     0.8182    0.6923    0.7500        78
       I-LOC     0.6250    0.4545    0.5263        11
      I-PERS     0.0000    0.0000    0.0000         8
   I-PERS:AG     0.8000    0.4000    0.5333        10
 I-PERS:FILI     0.8889    0.8000    0.8421       100
I-PERS:TITLE     0.0000    0.0000    0.0000         1
     I-TITLE     0.8163    0.7692    0.7921        52
           O     0.9390    0.9810    0.9595      2103

    accuracy                         0.8898      3212
   macro avg     0.6996    0.5870    0.6253      3212
weighted avg     0.8826    0.8898    0.8840      3212
```

### SVM (Non-Linguistic Features)
```
              precision    recall  f1-score   support

       B-LOC     0.6000    0.1579    0.2500        19
      B-PERS     0.5455    0.2857    0.3750        84
   B-PERS:AG     0.7143    0.5769    0.6383        52
  B-PERS:COG     0.7889    0.6730    0.7263       211
 B-PERS:FILI     0.8382    0.7703    0.8028        74
B-PERS:NOMEN     0.8373    0.7919    0.8140       221
 B-PERS:PRAE     0.8655    0.8655    0.8655       119
B-PERS:TITLE     0.7536    0.8667    0.8062        60
B-PERS:TRIBE     0.8750    0.7778    0.8235         9
     B-TITLE     0.8194    0.7564    0.7867        78
       I-LOC     0.5714    0.3636    0.4444        11
      I-PERS     0.0000    0.0000    0.0000         8
   I-PERS:AG     1.0000    0.5000    0.6667        10
 I-PERS:FILI     0.9419    0.8100    0.8710       100
I-PERS:TITLE     0.0000    0.0000    0.0000         1
     I-TITLE     0.9130    0.8077    0.8571        52
           O     0.9204    0.9838    0.9510      2103

    accuracy                         0.8882      3212
   macro avg     0.7050    0.5875    0.6282      3212
weighted avg     0.8784    0.8882    0.8796      3212

```

### TreeLSTM
```
              precision    recall  f1-score   support

       B-LOC     0.2000    0.3333    0.2500         6
      B-PERS     0.3000    0.1429    0.1935        21
   B-PERS:AG     0.3750    0.5455    0.4444        22
  B-PERS:COG     0.4382    0.3786    0.4062       103
 B-PERS:FILI     0.7400    0.8222    0.7789        45
B-PERS:NOMEN     0.6477    0.6477    0.6477        88
 B-PERS:PRAE     0.5000    0.1250    0.2000         8
B-PERS:TITLE     0.7143    0.8333    0.7692        24
B-PERS:TRIBE     0.6667    0.6667    0.6667         3
     B-TITLE     0.6087    0.6667    0.6364        21
       I-LOC     0.0000    0.0000    0.0000         7
      I-PERS     0.0000    0.0000    0.0000         4
   I-PERS:AG     1.0000    0.3333    0.5000         3
 I-PERS:FILI     0.8148    0.6471    0.7213        34
I-PERS:TITLE     0.0000    0.0000    0.0000         0
     I-TITLE     0.6667    0.7778    0.7179        18
           O     0.8587    0.8960    0.8770       529

    accuracy                         0.7457       936
   macro avg     0.5018    0.4598    0.4594       936
weighted avg     0.7308    0.7457    0.7345       936

```

### BiLSTM
```
              precision    recall  f1-score   support

       B-LOC     0.5000    0.1579    0.2400        19
      B-PERS     0.7308    0.4524    0.5588        84
   B-PERS:AG     0.4744    0.7115    0.5692        52
  B-PERS:COG     0.7751    0.7678    0.7714       211
 B-PERS:FILI     0.8923    0.7838    0.8345        74
B-PERS:NOMEN     0.8480    0.7828    0.8141       221
 B-PERS:PRAE     0.8966    0.8739    0.8851       119
B-PERS:TITLE     0.7571    0.8833    0.8154        60
B-PERS:TRIBE     1.0000    0.7778    0.8750         9
     B-TITLE     0.6984    0.5641    0.6241        78
       I-LOC     1.0000    0.0909    0.1667        11
      I-PERS     0.0000    0.0000    0.0000         8
   I-PERS:AG     0.7143    0.5000    0.5882        10
 I-PERS:FILI     0.8936    0.8400    0.8660       100
I-PERS:TITLE     0.0000    0.0000    0.0000         1
     I-TITLE     0.9130    0.8077    0.8571        52
           O     0.9362    0.9767    0.9560      2103

    accuracy                         0.8920      3212
   macro avg     0.7076    0.5865    0.6130      3212
weighted avg     0.8879    0.8920    0.8860      3212

```

### LatinBERT
```
              precision    recall  f1-score   support

       B-LOC     0.2778    0.2632    0.2703        19
      B-PERS     0.5875    0.5402    0.5629        87
   B-PERS:AG     0.5763    0.6296    0.6018        54
  B-PERS:COG     0.7788    0.7535    0.7660       215
 B-PERS:FILI     0.8551    0.8082    0.8310        73
B-PERS:NOMEN     0.8358    0.7602    0.7962       221
 B-PERS:PRAE     0.8548    0.8833    0.8689       120
B-PERS:TITLE     0.8060    0.9000    0.8504        60
     B-TITLE     0.7465    0.6795    0.7114        78
       I-LOC     0.6667    0.3636    0.4706        11
      I-PERS     0.0000    0.0000    0.0000         5
   I-PERS:AG     0.3000    0.3000    0.3000        10
 I-PERS:FILI     0.8889    0.8713    0.8800       101
I-PERS:TITLE     0.0000    0.0000    0.0000         1
     I-TITLE     0.8421    0.6154    0.7111        52
           O     0.9481    0.9729    0.9604      2105

    accuracy                         0.8913      3212
   macro avg     0.6228    0.5838    0.5988      3212
weighted avg     0.8876    0.8913    0.8886      3212

```

### BERT
```
              precision    recall  f1-score   support

       B-LOC     0.4474    0.3542    0.3953        48
      B-PERS     0.7647    0.6771    0.7182       192
   B-PERS:AG     0.6441    0.6847    0.6638       111
  B-PERS:COG     0.7952    0.9215    0.8537       535
 B-PERS:FILI     0.8074    0.8015    0.8044       136
B-PERS:NOMEN     0.8812    0.8918    0.8865       499
 B-PERS:PRAE     0.9282    0.8615    0.8936       195
B-PERS:TITLE     0.7500    0.8889    0.8136       108
B-PERS:TRIBE     0.8235    0.7778    0.8000        18
     B-TITLE     0.7110    0.8564    0.7769       181
       I-LOC     0.5789    0.4074    0.4783        27
      I-PERS     0.0000    0.0000    0.0000        15
   I-PERS:AG     1.0000    0.8500    0.9189        20
 I-PERS:FILI     0.9652    0.8362    0.8961       232
I-PERS:TITLE     0.0000    0.0000    0.0000         4
     I-TITLE     0.9273    0.9027    0.9148       113
           O     0.9722    0.9615    0.9668      3925

    accuracy                         0.9123      6359
   macro avg     0.7057    0.6866    0.6930      6359
weighted avg     0.9122    0.9123    0.9111      6359
```

### GPT (Finetuned)
```
              precision    recall  f1-score   support

       B-LOC     0.2632    0.2632    0.2632        19
      B-PERS     0.7882    0.7976    0.7929        84
   B-PERS:AG     0.7000    0.9423    0.8033        52
  B-PERS:COG     0.8382    0.8104    0.8241       211
 B-PERS:FILI     0.8657    0.7838    0.8227        74
B-PERS:NOMEN     0.8356    0.8281    0.8318       221
 B-PERS:PRAE     0.8678    0.8824    0.8750       119
B-PERS:TITLE     0.8182    0.9000    0.8571        60
B-PERS:TRIBE     1.0000    0.8889    0.9412         9
     B-TITLE     0.7500    0.6923    0.7200        78
       I-LOC     0.0000    0.0000    0.0000        11
      I-PERS     0.0000    0.0000    0.0000         8
   I-PERS:AG     1.0000    1.0000    1.0000        10
 I-PERS:FILI     0.9000    0.9000    0.9000       100
I-PERS:TITLE     0.0000    0.0000    0.0000         1
     I-TITLE     0.0000    0.0000    0.0000        52
           O     0.9475    0.9781    0.9626      2103

    accuracy                         0.9063      3212
   macro avg     0.6220    0.6275    0.6232      3212
weighted avg     0.8859    0.9063    0.8955      3212
```

### Gemini (Finetuned)
```
              precision    recall  f1-score   support

       B-LOC     0.3750    0.4737    0.4186        19
      B-PERS     0.5849    0.3690    0.4526        84
   B-PERS:AG     0.7656    0.9423    0.8448        52
  B-PERS:COG     0.7679    0.8626    0.8125       211
 B-PERS:FILI     0.7949    0.8378    0.8158        74
B-PERS:NOMEN     0.7800    0.8824    0.8280       221
 B-PERS:PRAE     0.8852    0.9076    0.8963       119
B-PERS:TITLE     0.8088    0.9167    0.8594        60
B-PERS:TRIBE     0.0000    0.0000    0.0000         9
     B-TITLE     0.6559    0.7821    0.7135        78
       I-LOC     0.0000    0.0000    0.0000        11
      I-PERS     0.0000    0.0000    0.0000         8
   I-PERS:AG     1.0000    1.0000    1.0000        10
 I-PERS:FILI     0.9462    0.8800    0.9119       100
I-PERS:TITLE     0.0000    0.0000    0.0000         1
     I-TITLE     0.8936    0.8077    0.8485        52
           O     0.9730    0.9582    0.9655      2103

    accuracy                         0.9050      3212
   macro avg     0.6018    0.6247    0.6098      3212
weighted avg     0.9002    0.9050    0.9011      3212
```

### Claude (10-shot)
```
              precision    recall  f1-score   support

       B-LOC     0.3871    0.6316    0.4800        19
      B-PERS     0.8571    0.1429    0.2449        84
   B-PERS:AG     0.7419    0.8846    0.8070        52
  B-PERS:COG     0.6679    0.8863    0.7617       211
 B-PERS:FILI     0.8590    0.9054    0.8816        74
B-PERS:NOMEN     0.8850    0.9050    0.8949       221
 B-PERS:PRAE     0.8968    0.9496    0.9224       119
B-PERS:TITLE     0.7778    0.9333    0.8485        60
B-PERS:TRIBE     1.0000    0.8889    0.9412         9
     B-TITLE     0.6182    0.8718    0.7234        78
       I-LOC     0.6667    0.3636    0.4706        11
      I-PERS     1.0000    0.1250    0.2222         8
   I-PERS:AG     1.0000    1.0000    1.0000        10
 I-PERS:FILI     0.9694    0.9500    0.9596       100
I-PERS:TITLE     1.0000    1.0000    1.0000         1
     I-TITLE     0.7288    0.8269    0.7748        52
           O     0.9813    0.9472    0.9639      2103

    accuracy                         0.9075      3212
   macro avg     0.8257    0.7772    0.7586      3212
weighted avg     0.9195    0.9075    0.9036      3212
```

### spaCy (latincy, coarse BIO)
```
=== latincy on TEST set (coarse BIO) ===
Precision: 0.6884
Recall:    0.5005
F1-score:  0.5796

================= Classification Report =============
              precision    recall  f1-score   support

         LOC       0.11      0.11      0.11        19
        PERS       0.71      0.56      0.62       830
       TITLE       0.00      0.00      0.00        78

   micro avg       0.69      0.50      0.58       927
   macro avg       0.27      0.22      0.24       927
weighted avg       0.63      0.50      0.56       927
```

---

### Summary

- **Best overall accuracy**: BERT (0.9123)
- **Best macro-F1 (balanced performance)**: Claude (0.7586)
- **Traditional models (SVM)** perform competitively but are slightly below neural models.
- **Tree-based models (TreeLSTM)** underperform due to limited data and structural sparsity.
- **LLMs (GPT/Gemini/Claude)** show strong performance, especially on structured name components.

---

## Usage

Install dependencies:

pip install -r requirement.txt

Then run training or evaluation scripts from the corresponding model folders under training/model/.