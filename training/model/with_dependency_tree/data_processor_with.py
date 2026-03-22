
import os
import random
import torch
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import sys

sys.setrecursionlimit(1000)

class Node:
    def __init__(self):
        self.child_list = []
        self.parent = None
        self.left = None
        self.right = None
        self.word = None
        self.pos = None
        self.bio = None
        self.deprel = None
        self.span: tuple[int, int] = (0, 0)
        self.word_index = -1
        self.pos_index = -1
        self.ne_index = -1
        self.deprel_index = -1
        self.word_split = []
        self.h = None
        self.c = None
        self.nodes = 1
        self.tokens = 0
        self.visited = False

    def add_child(self, child):
        if self.child_list:
            sibling = self.child_list[-1]
            sibling.right = child
            child.left = sibling
        self.child_list.append(child)
        child.parent = self
        self.nodes += child.nodes
        if not child.child_list:
            self.tokens += 1

    def get_leaf_nodes(self):
        if not self.child_list:
            return [self]
        
        leaves = []
        self.visited = True
        for child in self.child_list:
            if not child.visited:
                leaves.extend(child.get_leaf_nodes())
        self.visited = False
        return leaves

class LatinDataProcessor:
    def __init__(self, dataset_name="latin_inscriptions_with_dependency"):
        self.dataset = dataset_name
        os.makedirs(self.dataset, exist_ok=True)
        self.character_file = os.path.join(self.dataset, "character.txt")
        self.word_file = os.path.join(self.dataset, "word.txt")
        self.pos_file = os.path.join(self.dataset, "pos.txt")
        self.ne_file = os.path.join(self.dataset, "ne.txt")
        self.deprel_file = os.path.join(self.dataset, "deprel.txt")

    def load_data(self, file_path):
        data = []
        current_sentence = []
        last_text_id = None

        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 10:
                    continue

                token = {
                    'text_id': parts[0],
                    'word': parts[1],
                    'pos': parts[7],
                    'bio': parts[6],
                    'head': parts[8],
                    'deprel': parts[9]
                }

                if token['bio'] != 'O':
                    if not (token['bio'].startswith('B-') or token['bio'].startswith('I-')):
                        raise ValueError(f"Invalid BIO tag: {token['bio']}")

                if token['text_id'] != last_text_id and current_sentence:
                    data.append(current_sentence)
                    current_sentence = []

                current_sentence.append(token)
                last_text_id = token['text_id']

            if current_sentence:
                data.append(current_sentence)

        return data

    def extract_vocabulary(self, data_dict):
        vocab = {
            'words': set(),
            'chars': set(),
            'poses': set(),
            'deprels': set(),
            'nes': set()
        }

        for split in data_dict.values():
            for sentence in split:
                for token in sentence:
                    vocab['words'].add(token['word'])
                    vocab['poses'].add(token['pos'])
                    vocab['deprels'].add(token['deprel'])
                    vocab['nes'].add(token['bio'])
                    for char in token['word']:
                        vocab['chars'].add(char)

        vocab['nes'].add('O')

        self._write_vocab_file(self.word_file, sorted(vocab['words']))
        self._write_vocab_file(self.character_file, sorted(vocab['chars']))
        self._write_vocab_file(self.pos_file, sorted(vocab['poses']))
        self._write_vocab_file(self.ne_file, sorted(vocab['nes']))
        self._write_vocab_file(self.deprel_file, sorted(vocab['deprels']))

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

    def create_tree(self, sentence, vocab_maps):
        nodes = []
        root = None

        for idx, token in enumerate(sentence):
            node = Node()
            node.word = token['word']
            node.pos = token['pos']
            node.bio = token['bio']
            node.deprel = token['deprel']
            node.span = (idx, idx + 1)
            node.word_index = vocab_maps['word'].get(token['word'], -1)
            pos_value = token['pos'] if token['pos'] else 'UNK'
            node.pos_index = vocab_maps['pos'].get(pos_value, -1)
            node.deprel_index = vocab_maps['deprel'].get(token['deprel'], -1)
            node.ne_index = vocab_maps['ne'].get(token['bio'], -1)
            node.word_split = [vocab_maps['char'].get(c, -1) for c in token['word']]
            nodes.append(node)

        for idx, token in enumerate(sentence):
            head_ref = token['head']
            if head_ref.isdigit():
                head_idx = int(head_ref) - 1
            else:
                head_idx = next((i for i, t in enumerate(sentence) if t['word'] == head_ref), -1)

            if 0 <= head_idx < len(nodes):
                if nodes[head_idx] is not nodes[idx]:
                    nodes[head_idx].add_child(nodes[idx])
            elif token['deprel'] == 'ROOT':
                root = nodes[idx]

        if root is None:
            for node in nodes:
                if node.child_list:
                    root = node
                    break
            else:
                root = nodes[0] if nodes else Node()

        leaf_nodes = root.get_leaf_nodes()
        labels = [node.ne_index for node in leaf_nodes]
        return root, torch.LongTensor(labels)

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
            'pos_list': self._read_vocab(self.pos_file),
            'pos': self._create_index_map(self.pos_file),
            'deprel_list': self._read_vocab(self.deprel_file),
            'deprel': self._create_index_map(self.deprel_file),
            'ne_list': self._read_vocab(self.ne_file),
            'ne': self._create_index_map(self.ne_file)
        }

        vocab_maps['ne']['UNK'] = len(vocab_maps['ne'])

        tree_data = {}
        with Pool(min(cpu_count(), 4)) as pool:
            for split in ['train', 'validate', 'test']:
                args = [(s, vocab_maps) for s in raw_data[split]]
                results = pool.starmap(self.create_tree, args)
                tree_data[split] = results

        return (
            tree_data,
            vocab_maps['word_list'],
            vocab_maps['ne_list'],
            len(vocab_maps['char']),
            len(vocab_maps['pos']),
            len(vocab_maps['deprel']),
            len(vocab_maps['ne'])
        )

if __name__ == "__main__":
    processor = LatinDataProcessor()
    tree_data, word_list, ne_list, char_size, pos_size, deprel_size, ne_size = processor.read_dataset(
        train_file="data_for_training/train.csv",
        val_file="data_for_training/val.csv",
        test_file="data_for_training/test.csv"
    )