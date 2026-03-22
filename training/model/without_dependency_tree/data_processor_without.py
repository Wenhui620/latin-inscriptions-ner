import os
import random
import torch
from collections import defaultdict
from torch.utils.data import Dataset

class LatinDataProcessor:
    def __init__(self, dataset_name="latin_inscriptions_without_dependency"):
        self.dataset = dataset_name
        os.makedirs(self.dataset, exist_ok=True)
        self.character_file = os.path.join(self.dataset, "character.txt")
        self.word_file = os.path.join(self.dataset, "word.txt")
        self.ne_file = os.path.join(self.dataset, "ne.txt")

    def load_data(self, file_path):
        data = defaultdict(list)
        current_sentence = []
        last_text_id = None

        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue

                token = {
                    'text_id': parts[0],
                    'word': parts[1],
                    'bio': parts[6]
                }

                if token['text_id'] != last_text_id and current_sentence:
                    data['all'].append(current_sentence)
                    current_sentence = []

                current_sentence.append(token)
                last_text_id = token['text_id']

            if current_sentence:
                data['all'].append(current_sentence)

        return data['all']

    def extract_vocabulary(self, data):
        vocab = {
            'words': set(),
            'chars': set(),
            'nes': set()
        }

        for sentence in data['train'] + data['validate'] + data['test']:
            for token in sentence:
                vocab['words'].add(token['word'])
                vocab['nes'].add(token['bio'])
                for char in token['word']:
                    vocab['chars'].add(char)

        self._write_vocab_file(self.word_file, sorted(vocab['words']))
        self._write_vocab_file(self.character_file, sorted(vocab['chars']))
        self._write_vocab_file(self.ne_file, sorted(vocab['nes']))

        return vocab

    def _write_vocab_file(self, path, items):
        with open(path, 'w', encoding='utf-8') as f:
            for item in items:
                if item:
                    f.write(f"{item}\n")

    def _read_vocab(self, path):
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def _create_index_map(self, path):
        items = self._read_vocab(path)
        return {item: idx for idx, item in enumerate(items)}

    def prepare_sequences(self, sentences, vocab_maps):
        word_sequences = []
        char_sequences = []
        label_sequences = []

        for sentence in sentences:
            word_seq = []
            char_seq = []
            label_seq = []

            for token in sentence:
                word_seq.append(vocab_maps['word'].get(token['word'], 0))
                chars = [vocab_maps['char'].get(c, 0) for c in token['word']]
                char_seq.append(chars)
                label_seq.append(vocab_maps['ne'].get(token['bio'], -1))

            word_sequences.append(word_seq)
            char_sequences.append(char_seq)
            label_sequences.append(label_seq)

        return word_sequences, char_sequences, label_sequences

    def read_dataset(self, train_file='train.csv', val_file='val.csv', test_file='test.csv'):
        raw_data = {
            'train': self.load_data(train_file),
            'validate': self.load_data(val_file),
            'test': self.load_data(test_file)
        }

        vocab = self.extract_vocabulary(raw_data)

        vocab_maps = {
            'word_list': self._read_vocab(self.word_file),
            'word': self._create_index_map(self.word_file),
            'char_list': self._read_vocab(self.character_file),
            'char': self._create_index_map(self.character_file),
            'ne_list': self._read_vocab(self.ne_file),
            'ne': self._create_index_map(self.ne_file)
        }

        train_data = self.prepare_sequences(raw_data['train'], vocab_maps)
        val_data = self.prepare_sequences(raw_data['validate'], vocab_maps)
        test_data = self.prepare_sequences(raw_data['test'], vocab_maps)

        return (
            (train_data, val_data, test_data),
            vocab_maps['word_list'],
            vocab_maps['ne_list'],
            len(vocab_maps['char']),
            len(vocab_maps['ne'])
        )

class SequenceDataset(Dataset):
    def __init__(self, word_sequences, char_sequences, label_sequences):
        self.word_sequences = word_sequences
        self.char_sequences = char_sequences
        self.label_sequences = label_sequences

    def __len__(self):
        return len(self.word_sequences)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.word_sequences[idx]),
            [torch.LongTensor(chars) for chars in self.char_sequences[idx]],
            torch.LongTensor(self.label_sequences[idx])
        )

if __name__ == "__main__":
    processor = LatinDataProcessor()
    (train_data, val_data, test_data), word_list, ne_list, char_size, ne_size = processor.read_dataset(
        train_file="data_for_training/train.csv",
        val_file="data_for_training/val.csv",
        test_file="data_for_training/test.csv"
    )
