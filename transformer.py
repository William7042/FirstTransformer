import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Tokenizer:
    def __init__(self, text=None):
        if text is not None:
            self.chars = sorted(list(set(text)))
        else:
            self.chars = []

        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

    def save_vocab(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for char in self.chars:
                f.write(char + '\n')

    def load_vocab(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.chars = [line.rstrip('\n') for line in f]

        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}


class Transformer(nn.Module):
    def __init__(self,vocab_size=128, d_model=512, max_seq_len=1024):
        super(Transformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
    
    def layer(self):
        x = self.token_embedding + self.position_embedding
        normalized = self.layer_norm(x)