import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
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

class TransformerLayer(nn.Module):
    def __init__(self, d_model=512, n_heads = 8):
        super(TransformerLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.head_dim = d_model // n_heads 
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        #weight matrices
        self.W_q = nn.Linear(d_model, d_model) #query
        self.W_k = nn.Linear(d_model, d_model) #key
        self.W_v = nn.Linear(d_model, d_model) #value
        self.W_o = nn.Linear(d_model, d_model) #output

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x):
        #embedding
        batch_size, seq_len, _ = x.shape
        normalized = self.layer_norm(x)

        #multihead attention matrices
        Q = self.W_q(normalized)
        K = self.W_k(normalized)
        V = self.W_v(normalized)

        #reshape them into 8 heads
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        #scores = Q*K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        #attention function is softmax(casualMask(scores))* V
        casual_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device))
        scores = scores.masked_fill(casual_mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)

        #concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attention_output = self.W_o(attended_values) + x
        
        #ok now we run it through a feedforward network
        normalized_attention = self.layer_norm2(attention_output)
        ffn_output = self.ffn(normalized_attention)
        layer_output = ffn_output + attention_output
        return layer_output


class Transformer(nn.Module):
    def __init__(self,vocab_size=128, d_model=512, max_seq_len=1024, token_ids=None, n_heads=8,n_layers=6):
        super(Transformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.d_model = d_model
        self.token_ids = token_ids
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads 
        self.final_layer_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads) for _ in range(n_layers)])

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):

        batch_size, seq_len = token_ids.shape
        token_embeds = self.token_embedding(token_ids) 
        positions = torch.arange(seq_len, device=token_ids.device)
        position_embeds = self.position_embedding(positions)
        x = token_embeds + position_embeds
        

        for layer in self.layers:
            x = layer(x)
        
        x = self.final_layer_norm(x)
        logits = self.output_proj(x)
        return logits


def train(model, tokenizer, text_data, epochs=100, batch_size=32, seq_len=128, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    vocab_size = tokenizer.vocab_size
    i=0
    for epoch in range(epochs):
        print(i)
        i+=1
        total_loss = 0
        batch_count = 0
        
        for batch in get_batches(text_data, tokenizer, batch_size, seq_len):
            logits = model(batch)
            targets = batch[:, 1:]
            logits = logits[:, :-1]
            
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), 
                targets.reshape(-1)              
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        

        if save_path and (epoch + 1) % 10 == 0:
            save_model(model, tokenizer, save_path)

def generate(model, tokenizer, prompt="", max_tokens=100, temperature=1.0):
    model.eval()
    
    #encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0)  #add batch dim
    
    for _ in range(max_tokens):
        #get next token probabilities
        logits = model(tokens)         
        next_logits = logits[0, -1] / temperature #last position
        
        #sample next token
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        #append to sequence
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
    
    return tokenizer.decode(tokens[0].tolist())

def get_batches(text_data, tokenizer, batch_size, seq_len, device=device):

    tokens = tokenizer.encode(text_data)
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    
    num_sequences = len(tokens) - seq_len
    
    
    for _ in range(num_sequences // batch_size):
        batch = []
        
        for _ in range(batch_size):
            #random starting position
            start_idx = torch.randint(0, num_sequences, (1,)).item()
            sequence = tokens[start_idx:start_idx + seq_len]
            batch.append(sequence)
        
        #stack into batch tensor
        batch_tensor = torch.stack(batch).to(device)
        yield batch_tensor

def save_model(model, tokenizer, optimizer, epoch, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'vocab_size': tokenizer.vocab_size,
        'chars': tokenizer.chars,
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char
    }, filepath)

def load_model(filepath, d_model=256, n_layers=4, n_heads=8, max_seq_len=512):
    if torch.backends.mps.is_available():
        checkpoint = torch.load(filepath, map_location='mps')
    elif torch.cuda.is_available():
        checkpoint = torch.load(filepath, map_location='cuda')
    else:
        checkpoint = torch.load(filepath, map_location='cpu')

    #recreate tokenizer and model 
    tokenizer = Tokenizer()
    tokenizer.chars = checkpoint['chars']
    tokenizer.vocab_size = checkpoint['vocab_size']
    tokenizer.char_to_idx = checkpoint['char_to_idx']
    tokenizer.idx_to_char = checkpoint['idx_to_char']
    
    model = Transformer(vocab_size=tokenizer.vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, max_seq_len=max_seq_len)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    #return training state info
    return model, tokenizer, checkpoint.get('epoch', 0), checkpoint.get('optimizer_state_dict')

def main():
    model_path = "beemovie_model.pth"
    text_file = "shakespeare.txt"
    

    d_model = 256
    n_layers = 4
    n_heads = 8
    max_seq_len = 512
    
    #check if saved model exists
    if os.path.exists(model_path):
        print("Found existing model, loading...")
        model, tokenizer, epoch, optimizer_state = load_model(model_path, d_model, n_layers, n_heads, max_seq_len)
        model.to(device)
    else:
        print("No existing model found, training new one...")
        
        #load text data
        with open(text_file, 'r', encoding='utf-8') as f:
            text_data = f.read()
        
        tokenizer = Tokenizer(text_data)
        model = Transformer(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len
        ).to(device)
        
        #train the model
        train(model, tokenizer, text_data, epochs=50, batch_size=16, seq_len=64, save_path=model_path)
    

    model.eval()
    #this is the prompt
    prompt = "b"
    output = generate(model, tokenizer, prompt=prompt, max_tokens=200)
    print(f"\nGenerated text:\n{output}")

if __name__ == "__main__":
    import os
    main()