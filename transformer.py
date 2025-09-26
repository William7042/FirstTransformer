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
    def __init__(self,vocab_size=128, d_model=512, max_seq_len=1024, token_ids=None, n_heads=8):
        super(Transformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.token_ids = token_ids
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads 
        self.layer_norm_2 = nn.LayerNorm(d_model)
        #weight matrixes
        self.W_q = nn.Linear(d_model, d_model) #query
        self.W_k = nn.Linear(d_model, d_model) #key
        self.W_v = nn.Linear(d_model, d_model) #value
        self.W_o = nn.Linear(d_model, d_model) #output

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )


    def forward(self,token_ids):
        #embedding
        batch_size, seq_len = token_ids.shape
        token_embeds = self.token_embedding(token_ids) 
        positions = torch.arange(seq_len, device=token_ids.device)
        position_embeds = self.position_embedding(positions)
        x = token_embeds + position_embeds
        normalized = self.layer_norm(x)

        #multihead attention matrices
        Q = self.W_q(normalized)
        K = self.W_k(normalized)
        V = self.W_v(normalized)

        #reshape them into 9 heads
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        #scores = Q*K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        #attention function is softmax(casualMask(scores))* V
        casual_mask = torch.tril(torch.ones((seq_len, seq_len), device=token_ids.device))
        scores = scores.masked_fill(casual_mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)

        #concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attended_values) + x
        

        #ok now we run it through a feedforward network
