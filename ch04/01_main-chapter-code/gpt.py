# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        d_in: 768
        d_out: 768
        context_length: 1024
        dropout: 0.1
        num_heads: 12
        qkv_bias: False
        """

        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim
        """
        d_out // num_heads
        64
        self.head_dim
        64
        """

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        """
        self.W_query.weight.shape
        torch.Size([768, 768])
        """

        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        """
        self.W_key.weight.shape
        torch.Size([768, 768])
        """

        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        """
        self.W_value.weight.shape
        torch.Size([768, 768])
        """

        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        """
        self.out_proj.weight.shape
        torch.Size([768, 768])
        """

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        """
        x.shape
        torch.Size([1, 4, 768])
        b
        1
        num_tokens
        4
        d_in
        768
        """

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        """
        keys.shape
        torch.Size([1, 4, 768])
        """

        queries = self.W_query(x)
        """
        queries.shape
        torch.Size([1, 4, 768])
        """

        values = self.W_value(x)
        """
        values.shape
        torch.Size([1, 4, 768])
        """

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        """
        keys.shape
        torch.Size([1, 4, 12, 64])
        """

        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        """
        values.shape
        torch.Size([1, 4, 12, 64])
        """

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        """
        queries.shape
        torch.Size([1, 4, 12, 64])
        """

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        """
        keys.shape
        torch.Size([1, 12, 4, 64])
        """

        queries = queries.transpose(1, 2)
        """
        queries.shape
        torch.Size([1, 12, 4, 64])
        """

        values = values.transpose(1, 2)
        """
        values.shape
        torch.Size([1, 12, 4, 64])
        """

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        """
        keys.transpose(2, 3).shape
        torch.Size([1, 12, 64, 4])
        attn_scores.shape
        torch.Size([1, 12, 4, 4])
        """

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        """
        mask_bool.shape
        torch.Size([4, 4])
        """

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        """
        attn_scores.shape
        torch.Size([1, 12, 4, 4])
        """

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        """
        attn_weights.shape
        torch.Size([1, 12, 4, 4])
        """
        attn_weights = self.dropout(attn_weights)
        """
        attn_weights.shape
        torch.Size([1, 12, 4, 4])
        """

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        """
        (attn_weights @ values).shape
        torch.Size([1, 12, 4, 64])
        context_vec.shape
        torch.Size([1, 4, 12, 64])
        """

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        """
        context_vec.shape
        torch.Size([1, 4, 768])
        """
        context_vec = self.out_proj(context_vec)  # optional projection
        """
        context_vec.shape
        torch.Size([1, 4, 768])
        """

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        """
        emb_dim
        768
        """

        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        """
        cfg
        {
            'vocab_size': 50257,
            'context_length': 1024,
            'emb_dim': 768,
            'n_heads': 12,
            'n_layers': 12,
            'drop_rate': 0.1,
            'qkv_bias': False
        }
        """

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        """
        cfg
        {
            'vocab_size': 50257,
            'context_length': 1024,
            'emb_dim': 768,
            'n_heads': 12,
            'n_layers': 12,
            'drop_rate': 0.1,
            'qkv_bias': False
        }
        """

        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)

        self.norm1 = LayerNorm(cfg["emb_dim"])

        self.norm2 = LayerNorm(cfg["emb_dim"])

        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        x.shape
        torch.Size([1, 4, 768])
        """

        # Shortcut connection for attention block
        shortcut = x
        """
        shortcut.shape
        torch.Size([1, 4, 768])
        """

        x1 = self.norm1(x)
        """
        x1.shape
        torch.Size([1, 4, 768])
        """

        x2 = self.att(x1)  # Shape [batch_size, num_tokens, emb_size]
        """
        x2.shape
        torch.Size([1, 4, 768])
        """
        x3 = self.drop_shortcut(x2)
        """
        x3.shape
        torch.Size([1, 4, 768])
        """

        x4 = x3 + shortcut  # Add the original input back
        """
        x4.shape
        torch.Size([1, 4, 768])
        """

        # Shortcut connection for feed-forward block
        shortcut = x4
        """
        shortcut.shape
        torch.Size([1, 4, 768])
        """

        x5 = self.norm2(x4)
        """
        x5.shape
        torch.Size([1, 4, 768])
        """

        x6 = self.ff(x5)
        """
        x6.shape
        torch.Size([1, 4, 768])
        """

        x7 = self.drop_shortcut(x6)
        """
        x7.shape
        torch.Size([1, 4, 768])
        """
        x8 = x7 + shortcut  # Add the original input back
        """
        x8.shape
        torch.Size([1, 4, 768])
        """

        return x8


class GPTModel(nn.Module):
    def __init__(self, cfg):
        """
        cfg
        {
            'vocab_size': 50257,
            'context_length': 1024,
            'emb_dim': 768,
            'n_heads': 12,
            'n_layers': 12,
            'drop_rate': 0.1,
            'qkv_bias': False
        }
        """

        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        """
        self.tok_emb.weight.shape
        torch.Size([50257, 768])
        """

        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        """
        self.pos_emb.weight.shape
        torch.Size([1024, 768])
        """

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        """
        self.out_head.weight.shape
        torch.Size([50257, 768])
        """

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        """
        in_idx.shape
        torch.Size([1, 4])
        batch_size
        1
        seq_len
        4
        """

        tok_embeds = self.tok_emb(in_idx)
        """
        tok_embeds.shape
        torch.Size([1, 4, 768])
        """

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        """
        pos_embeds.shape
        torch.Size([4, 768])
        """

        x1 = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        """
        x1.shape
        torch.Size([1, 4, 768])
        """

        x2 = self.drop_emb(x1)
        """
        x2.shape
        torch.Size([1, 4, 768])
        """

        x3 = self.trf_blocks(x2)
        """
        x3.shape
        torch.Size([1, 4, 768])
        """

        x4 = self.final_norm(x3)
        """
        x4.shape
        torch.Size([1, 4, 768])
        """

        logits = self.out_head(x4)
        """
        logits.shape
        torch.Size([1, 4, 50257])
        """

        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    model

    idx
    tensor([[15496,    11,   314,   716]])
    idx.shape
    torch.Size([1, 4])

    max_new_tokens
    10

    context_size
    1024
    """

    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        """
        idx.shape: torch.Size([1, 4])

        idx.shape: torch.Size([1, 5])
        """

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        """
        idx_cond.shape: torch.Size([1, 4])

        idx_cond.shape: torch.Size([1, 5])
        """

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        """
        logits.shape: torch.Size([1, 4, 50257])
        logits.shape: torch.Size([1, 5, 50257])
        """

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]
        """
        logits.shape: torch.Size([1, 50257])
        logits.shape: torch.Size([1, 50257])
        """

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        """
        tensor([[27018]])
        tensor([[24086]])
        """

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }

    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)

    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")

    encoded = tokenizer.encode(start_context)
    """
    encoded
    [15496, 11, 314, 716]
    len(encoded)
    4
    """

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    """
    encoded_tensor
    tensor([[15496,    11,   314,   716]])
    encoded_tensor.shape
    torch.Size([1, 4])
    """

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=1,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    """
    out
    tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267,
            49706, 43231, 47062, 34657]])
    out.shape
    torch.Size([1, 14])
    out.squeeze(0).tolist()
    [15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267, 49706, 43231, 47062, 34657]
    """

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    """
    decoded_text
    'Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous'
    """

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
