import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from data_processor_with import LatinDataProcessor
from model_with import LatinTreeLSTMModel

class Config:
    batch_size = 32
    word_emb_dim = 100
    char_emb_dim = 30
    hidden_dim = 256
    learning_rate = 0.001
    epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_emb_dim = 32    
    deprel_dim = 16     

    max_word_len = 20
    char_cnn_dim = 50
    char_kernel_size = 3
    num_layers = 2
    dropout = 0.5  

class TreeDataset(Dataset):
    def __init__(self, tree_data):
        self.trees = [tree for tree, _ in tree_data]
        self.labels = [label for _, label in tree_data]
        
    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, idx):
        return self.trees[idx], self.labels[idx]

def collate_fn(batch):
    trees = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    max_len = max(len(label) for label in labels)
    
    padded_labels = []
    for label in labels:
        padding = torch.full((max_len - len(label),), -1, dtype=torch.long)
        padded_labels.append(torch.cat([label, padding]))
    
    return trees, torch.stack(padded_labels)

def train():
    processor = LatinDataProcessor()
    tree_data, word_list, ne_list, char_size, pos_size, deprel_size, ne_size = processor.read_dataset(train_file="raw/train.csv", val_file="raw/val.csv", test_file="raw/test.csv")
    
    train_set = TreeDataset(tree_data['train'])
    val_set = TreeDataset(tree_data['validate'])
    
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size,
                          collate_fn=collate_fn)
    
    model_config = {
        'word_vocab_size': len(word_list),
        'char_vocab_size': char_size,
        'pos_vocab_size': pos_size,
        'deprel_vocab_size': deprel_size,
        'ner_vocab_size': ne_size,
        'word_emb_dim': Config.word_emb_dim,
        'char_emb_dim': Config.char_emb_dim,
        'pos_emb_dim': Config.pos_emb_dim,
        'deprel_dim': Config.deprel_dim,
        'hidden_dim': Config.hidden_dim,
        'char_cnn_dim': Config.char_cnn_dim,
        'char_kernel_size': Config.char_kernel_size,
        'hidden_dim': Config.hidden_dim,
        'num_layers': Config.num_layers,
        'dropout': Config.dropout
    }
    
    model = LatinTreeLSTMModel(model_config).to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        
        for trees, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            labels = labels.to(Config.device)
            
            outputs = model(trees)
            batch_size, seq_len, num_classes = outputs.shape
            
            
            # 计算损失
            loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for trees, labels in val_loader:
                labels = labels.to(Config.device)
                outputs = model(trees)
                batch_size, seq_len, num_classes = outputs.shape
                
                if labels.shape[1] > seq_len:
                    labels = labels[:, :seq_len]
                elif labels.shape[1] < seq_len:
                    padding = torch.full((batch_size, seq_len - labels.shape[1]), 
                                       -1, dtype=torch.long, device=Config.device)
                    labels = torch.cat([labels, padding], dim=1)
                
                loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 2)
                mask = labels != -1
                correct += (predicted[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
        
        print(f"Epoch {epoch+1}/{Config.epochs} | "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {correct/total:.4f}")

    torch.save(model.state_dict(), "model/with_dependency_tree/best_model.pth")
    torch.save(model.state_dict(), "model/with_dependency_tree/model_with.pt")
    print("TreeLSTM model saved to model_with.pt")

def test():
    processor = LatinDataProcessor()
    tree_data, word_list, ne_list, char_size, pos_size, deprel_size, ne_size = processor.read_dataset(train_file="raw/train.csv", val_file="raw/val.csv", test_file="raw/test.csv")
    
    model_config = {
        'word_vocab_size': len(word_list),
        'char_vocab_size': char_size,
        'pos_vocab_size': pos_size,
        'deprel_vocab_size': deprel_size,
        'ner_vocab_size': ne_size,
        'word_emb_dim': Config.word_emb_dim,
        'char_emb_dim': Config.char_emb_dim,
        'pos_emb_dim': Config.pos_emb_dim,
        'deprel_dim': Config.deprel_dim,
        'hidden_dim': Config.hidden_dim
    }
    
    model = LatinTreeLSTMModel(model_config).to(Config.device)
    model.load_state_dict(torch.load("model/with_dependency_tree/best_model.pth", map_location=Config.device))
    model.eval()

    test_set = TreeDataset(tree_data['test'])
    test_loader = DataLoader(test_set, batch_size=Config.batch_size, collate_fn=collate_fn)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for trees, labels in test_loader:
            labels = labels.to(Config.device)
            outputs = model(trees)
            _, predicted = torch.max(outputs, 2)
            
            mask = labels != -1
            all_preds.extend(predicted[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
    
    id2label = {i: label for i, label in enumerate(ne_list)}
    pred_labels = [id2label[p] for p in all_preds]
    true_labels = [id2label[l] for l in all_labels]

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
    #train()
    test()


