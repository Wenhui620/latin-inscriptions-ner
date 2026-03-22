import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, char_cnn_dim, char_kernel_size=3):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_emb_dim)
        self.conv = nn.Conv1d(char_emb_dim, char_cnn_dim, char_kernel_size, padding=char_kernel_size // 2)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, chars):
        char_emb = self.char_embed(chars)  # [seq_len, word_len, char_emb_dim]
        char_emb = char_emb.permute(0, 2, 1)  # [seq_len, char_emb_dim, word_len]
        conv_out = F.relu(self.conv(char_emb))  # [seq_len, char_cnn_dim, word_len]
        pooled = self.pool(conv_out).squeeze(-1)  # [seq_len, char_cnn_dim]
        return pooled

class BiLSTM_CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(config['word_vocab_size'], config['word_emb_dim'])

        self.char_cnn = CharCNN(
            config['char_vocab_size'],
            config['char_emb_dim'],
            config['char_cnn_dim'],
            config['char_kernel_size']
        )

        self.lstm = nn.LSTM(
            input_size=config['word_emb_dim'] + config['char_cnn_dim'],
            hidden_size=config['hidden_dim'] // 2,
            num_layers=config['num_layers'],
            bidirectional=True,
            batch_first=True,
            dropout=config['dropout'] if config['num_layers'] > 1 else 0
        )

        self.output = nn.Linear(config['hidden_dim'], config['ner_vocab_size'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, words, chars_list):
        word_emb = self.word_embed(words)  # [batch_size, seq_len, word_emb_dim]

        char_features = []
        for chars in chars_list:
            char_feat = self.char_cnn(chars)  # [seq_len, char_cnn_dim]
            char_features.append(char_feat)

        max_len = words.size(1)
        padded_char_features = []
        for char_feat in char_features:
            padding = torch.zeros(max_len - char_feat.size(0), self.config['char_cnn_dim'], device=words.device)
            padded = torch.cat([char_feat, padding], dim=0)
            padded_char_features.append(padded)

        char_features = torch.stack(padded_char_features, dim=0)  # [batch_size, seq_len, char_cnn_dim]
        combined = torch.cat([word_emb, char_features], dim=-1)  # [batch_size, seq_len, word+char dim]
        combined = self.dropout(combined)

        lstm_out, _ = self.lstm(combined)  # [batch_size, seq_len, hidden_dim]
        lstm_out = self.dropout(lstm_out)

        logits = self.output(lstm_out)  # [batch_size, seq_len, ner_vocab_size]
        return logits


def collate_fn(batch):
    words, chars_list, labels = zip(*batch)
    words = torch.nn.utils.rnn.pad_sequence(words, batch_first=True, padding_value=0)

    max_word_len = max(max(len(chars) for chars in seq) for seq in chars_list)
    padded_chars = []
    for seq in chars_list:
        seq_padded = []
        for word_chars in seq:
            padding = torch.zeros(max_word_len - len(word_chars), dtype=torch.long)
            padded = torch.cat([word_chars, padding])
            seq_padded.append(padded)
        padded_chars.append(torch.stack(seq_padded))

    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)

    return words, padded_chars, labels
