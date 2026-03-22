import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from data_processor_without import LatinDataProcessor, SequenceDataset
from model_without import BiLSTM_CNN, collate_fn

class Config:
    batch_size = 32
    word_emb_dim = 100
    char_emb_dim = 30
    hidden_dim = 256
    learning_rate = 0.001
    epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_word_len = 20
    char_cnn_dim = 50
    char_kernel_size = 3
    num_layers = 2
    dropout = 0.5

def train():
    processor = LatinDataProcessor()
    (train_data, val_data, test_data), word_list, ne_list, char_size, ne_size = processor.read_dataset(train_file="raw/train.csv", val_file="raw/val.csv", test_file="raw/test.csv")

    train_set = SequenceDataset(*train_data)
    val_set = SequenceDataset(*val_data)
    test_set = SequenceDataset(*test_data)

    train_loader = DataLoader(train_set, batch_size=Config.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size,
                             collate_fn=collate_fn)

    model_config = {
        'word_vocab_size': len(word_list),
        'char_vocab_size': char_size,
        'ner_vocab_size': ne_size,
        'word_emb_dim': Config.word_emb_dim,
        'char_emb_dim': Config.char_emb_dim,
        'char_cnn_dim': Config.char_cnn_dim,
        'char_kernel_size': Config.char_kernel_size,
        'hidden_dim': Config.hidden_dim,
        'num_layers': Config.num_layers,
        'dropout': Config.dropout
    }

    model = BiLSTM_CNN(model_config).to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0

        for words, chars, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            words = words.to(Config.device)
            chars = [c.to(Config.device) for c in chars]
            labels = labels.to(Config.device)

            optimizer.zero_grad()
            outputs = model(words, chars)

            loss = criterion(outputs.view(-1, ne_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for words, chars, labels in val_loader:
                words = words.to(Config.device)
                chars = [c.to(Config.device) for c in chars]
                labels = labels.to(Config.device)

                outputs = model(words, chars)
                loss = criterion(outputs.view(-1, ne_size), labels.view(-1))
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 2)
                mask = labels != -1
                correct += (predicted[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

        print(f"Epoch {epoch+1}/{Config.epochs} | "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {correct/total:.4f}")

    os.makedirs("training/model/without_dependency_tree", exist_ok=True)

    torch.save(model.state_dict(), "training/model/without_dependency_tree/bilstm_cnn_model.pth")
    torch.save(model.state_dict(), "training/model/without_dependency_tree/model_without.pt")
    print("BiLSTM-CNN model saved to training/model/without_dependency_tree/model_without.pt")

def test():
    processor = LatinDataProcessor()
    (train_data, val_data, test_data), word_list, ne_list, char_size, ne_size = processor.read_dataset(train_file="raw/train.csv", val_file="raw/val.csv", test_file="raw/test.csv")

    test_set = SequenceDataset(*test_data)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size, collate_fn=collate_fn)

    model_config = {
        'word_vocab_size': len(word_list),
        'char_vocab_size': char_size,
        'ner_vocab_size': ne_size,
        'word_emb_dim': Config.word_emb_dim,
        'char_emb_dim': Config.char_emb_dim,
        'char_cnn_dim': Config.char_cnn_dim,
        'char_kernel_size': Config.char_kernel_size,
        'hidden_dim': Config.hidden_dim,
        'num_layers': Config.num_layers,
        'dropout': Config.dropout
    }

    model = BiLSTM_CNN(model_config).to(Config.device)
    ckpt_path = "training/model/without_dependency_tree/bilstm_cnn_model.pth"
    state_dict = torch.load(ckpt_path, map_location=Config.device)
    model.load_state_dict(state_dict, strict=True)
    print(f" Loaded checkpoint: {ckpt_path}")
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for words, chars, labels in test_loader:
            words = words.to(Config.device)
            chars = [c.to(Config.device) for c in chars]
            labels = labels.to(Config.device)

            outputs = model(words, chars)
            _, predicted = torch.max(outputs, 2)

            mask = labels != -1
            all_preds.extend(predicted[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())

    id2label = {i: label for i, label in enumerate(ne_list)}
    true_labels = [id2label[l] for l in all_labels]
    pred_labels = [id2label[p] for p in all_preds]

    print("\nModel Evaluation Report:")
    print(classification_report(
        true_labels,
        pred_labels,
        labels=ne_list,
        target_names=ne_list,
        digits=4,
        zero_division=0
    ))


if __name__ == "__main__":
    train()
    #test()
