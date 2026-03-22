import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import numpy as np 
from collections import Counter
from sklearn.metrics import classification_report as sklearn_cr, accuracy_score
import multiprocessing
import warnings
warnings.filterwarnings("ignore", message="resource_tracker")
multiprocessing.set_start_method("spawn", force=True)

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

def main():
    import random
    import numpy as np
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    DATA_DIR = 'data/data_for_training/'
    TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
    VAL_CSV   = os.path.join(DATA_DIR, 'val.csv')
    TEST_CSV  = os.path.join(DATA_DIR, 'test.csv')


    train_df = pd.read_csv(TRAIN_CSV, encoding='utf-8')
    val_df   = pd.read_csv(VAL_CSV, encoding='utf-8')
    test_df  = pd.read_csv(TEST_CSV, encoding='utf-8')
    train_df['word'] = train_df['word'].astype(str)
    val_df['word']   = val_df['word'].astype(str)
    test_df['word']  = test_df['word'].astype(str)

    def df_to_sentences(df, id_col='text_id', word_col='word', tag_col='BIO'):
        """
        Convert a token-level DataFrame into sentence-level lists.
        - id_col: column that identifies a sentence (here: 'text_id')
        - word_col: token column
        - tag_col: label column (originally 'BIO', later we also call with 'tag_id')
        """
        words_list = []
        tags_list = []
        for sid, group in df.groupby(id_col, sort=False):
            words_list.append(group[word_col].astype(str).tolist())
            tags_list.append(group[tag_col].tolist())
        return words_list, tags_list

    train_sentences, train_labels = df_to_sentences(train_df)
    val_sentences,   val_labels   = df_to_sentences(val_df)
    test_sentences,  test_labels  = df_to_sentences(test_df)

    model_name = 'google-bert/bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    def encode_and_align_labels(sentences, labels, label_all_tokens=True):
        encodings = tokenizer(
            sentences,
            is_split_into_words=True,
            return_offsets_mapping=False,
            padding=True,
            truncation=True,
            max_length=256
        )
        all_labels = []
        for i, label in enumerate(labels):
            word_ids = encodings.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            all_labels.append(label_ids)
        encodings["labels"] = all_labels
        return encodings

    train_encodings = encode_and_align_labels(train_sentences, train_labels)
    val_encodings   = encode_and_align_labels(val_sentences,   val_labels)
    test_encodings  = encode_and_align_labels(test_sentences,  test_labels)


    train_dataset = NERDataset(train_encodings)
    val_dataset   = NERDataset(val_encodings)
    test_dataset  = NERDataset(test_encodings)

    BATCH_SIZE = 32

    dataloader_args = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    train_loader = DataLoader(train_dataset, **dataloader_args)
    val_loader   = DataLoader(val_dataset,   **{**dataloader_args, 'shuffle': False})
    test_loader  = DataLoader(test_dataset,  **{**dataloader_args, 'shuffle': False})


    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples.")

    unique_tags = sorted(set(train_df['BIO'].unique()) | set(val_df['BIO'].unique()) | set(test_df['BIO'].unique()))
    tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}

    train_df['tag_id'] = train_df['BIO'].map(tag2id)
    val_df['tag_id'] = val_df['BIO'].map(tag2id)
    test_df['tag_id'] = test_df['BIO'].map(tag2id)

    train_sentences, train_labels = df_to_sentences(train_df, tag_col='tag_id')
    val_sentences,   val_labels   = df_to_sentences(val_df,   tag_col='tag_id')
    test_sentences,  test_labels  = df_to_sentences(test_df,  tag_col='tag_id')

    train_encodings = encode_and_align_labels(train_sentences, train_labels)
    val_encodings   = encode_and_align_labels(val_sentences,   val_labels)
    test_encodings  = encode_and_align_labels(test_sentences,  test_labels)

    train_dataset = NERDataset(train_encodings)
    val_dataset   = NERDataset(val_encodings)
    test_dataset  = NERDataset(test_encodings)
    train_loader  = DataLoader(train_dataset, **dataloader_args)
    val_loader    = DataLoader(val_dataset,   **{**dataloader_args, 'shuffle': False})
    test_loader   = DataLoader(test_dataset,  **{**dataloader_args, 'shuffle': False})

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(unique_tags),
        id2label=id2tag,
        label2id=tag2id
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_train_tag_ids = [tag_id for seq in train_labels for tag_id in seq]
    tag_counts = Counter(all_train_tag_ids)
    total = sum(tag_counts.values())
    class_weights = [1.0 - (tag_counts.get(i, 0) / total) for i in range(len(tag2id))]
    class_weights = torch.tensor(class_weights).to(device)

    from torch.nn import CrossEntropyLoss
    loss_fct = CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

        model.eval()
        val_loss = 0
        for batch in tqdm(val_loader, desc="Validating"):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = "best_model"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            # save label mappings for reproducibility
            import json
            with open(os.path.join(save_path, "label_map.json"), "w") as f:
                json.dump({"tag2id": tag2id, "id2tag": id2tag}, f)
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        true_labels = labels.cpu().numpy()
        for pred_seq, true_seq in zip(predictions, true_labels):
            seq_preds = []
            seq_trues = []
            for p, t in zip(pred_seq, true_seq):
                if t != -100:
                    seq_preds.append(id2tag[p])
                    seq_trues.append(id2tag[t])
            all_preds.append(seq_preds)
            all_labels.append(seq_trues)

    y_true = [label for seq in all_labels for label in seq]
    y_pred = [pred  for seq in all_preds  for pred  in seq]

    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    errors = []
    for true_label in labels:
        for pred_label in labels:
            if true_label != pred_label:
                count = cm_df.loc[true_label, pred_label]
                total = cm_df.loc[true_label].sum()
                percent = count / total * 100 if total > 0 else 0
                if percent > 1:  
                    errors.append({
                        "True": true_label,
                        "Pred": pred_label,
                        "Count": count,
                        "Percent": round(percent, 2)
                    })
    errors_df = pd.DataFrame(errors).sort_values(by="Percent", ascending=False)
    print("Critical Error Patterns:")
    print(errors_df)

    # 可视化错误热力图
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_errors, annot=True, fmt="d", cmap="Reds",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Critical Error Patterns Heatmap")
    plt.show()

    report = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
    print(report)
    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
