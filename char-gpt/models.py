"""
Neural network models for character-level language modeling.
This module implements the transformer architecture components,
starting with the scaled dot-product attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import yaml
from dataclasses import dataclass
from pathlib import Path
from data import get_batch

@dataclass
class ModelConfig:
    vocab_size: int
    n_embed: int
    num_heads: int
    n_layer: int
    max_seq_length: int = 256
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        return cls(**config["model"])

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    def validate(self) -> None:
        """Validate model configuration parameters."""
        if self.n_embed % self.num_heads != 0:
            raise ValueError(f"n_embed ({self.n_embed}) must be divisible by num_heads ({self.num_heads})")
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length ({self.max_seq_length}) must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout ({self.dropout}) must be between 0 and 1")

class Head(nn.Module):
    """
    Implements a single scaled dot-product self-attention head.
    
    This is the core attention mechanism that:
    1. Computes attention scores between all positions
    2. Applies scaling for better initialization and optimization
    3. Uses causal masking to ensure autoregressive property
    4. Performs weighted aggregation of values
    
    Args:
        head_size: Size of the attention head (dimension of key/query/value)
        n_embed: Size of the input embedding dimension
        dropout: Dropout probability (default: 0.1)
        max_seq_length: Maximum sequence length for causal masking (default: 256)
    """
    def __init__(
        self,
        head_size: int,
        n_embed: int,
        dropout: float = 0.1,
        max_seq_length: int = 256
    ):
        super().__init__()
        self.head_size = head_size
        
        # Linear projections for key, query, and value
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        
        # Causal mask to ensure attention only attends to previous tokens
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(max_seq_length, max_seq_length))
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention head.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embed)
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, head_size)
        """
        B, T, C = x.shape
        
        # Compute key, query, and value projections
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(q.size(-1)))
        
        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Apply softmax and dropout
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Perform weighted aggregation of values
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = wei @ v
        
        return out


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention by running multiple attention heads in parallel.
    
    This module:
    1. Splits the input into multiple heads
    2. Applies attention in parallel
    3. Concatenates the results
    4. Projects back to the original dimension
    
    Args:
        num_heads: Number of attention heads
        head_size: Size of each attention head
        n_embed: Size of the input embedding dimension
        dropout: Dropout probability (default: 0.1)
        max_seq_length: Maximum sequence length for causal masking (default: 256)
    """
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embed: int,
        dropout: float = 0.1,
        max_seq_length: int = 256
    ):
        super().__init__()
        # Verify that n_embed is divisible by num_heads
        assert n_embed % num_heads == 0, "n_embed must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_size = head_size
        
        # Create multiple attention heads
        self.heads = nn.ModuleList([
            Head(head_size=head_size, n_embed=n_embed, dropout=dropout, max_seq_length=max_seq_length)
            for _ in range(num_heads)
        ])
        
        # Project concatenated heads back to n_embed
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embed)
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_embed)
        """
        # Apply attention in parallel and concatenate
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project back to n_embed and apply dropout
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Implements the feed-forward network used in transformer blocks.
    
    This module:
    1. Expands the input dimension by a factor of 4
    2. Applies ReLU activation
    3. Projects back to the original dimension
    4. Applies dropout
    
    Args:
        n_embed: Size of the input embedding dimension
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, n_embed: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Impliments a transformer block with multi-head self attention and residual connections
    Args:
        n_embed: Size of the input embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        max_seq_length: Maximum sequence length for causal masking (default: 256)
    """
    def __init__(self, n_embed: int, num_heads: int, dropout: float = 0.1, max_seq_length: int = 256):
        super().__init__()
        # Calculate head size
        assert n_embed % num_heads == 0, "e_embed must be divisible by num_heads"
        head_size = n_embed // num_heads
        
        # Multi-head attention
        self.sa = MultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            n_embed=n_embed,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        # Feed-forward network
        self.ffwd = FeedForward(n_embed=n_embed, dropout=dropout)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embed)
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_embed)
        """
        # Self-attention with residual connection
        x = x + self.dropout(self.sa(self.ln1(x)))
        # Feed-forward with residual connection
        x = x + self.dropout(self.ffwd(self.ln2(x)))
        return x


class CharGPT(nn.Module):
    """
    Character-level GPT model implementing a transformer-based language model.
    
    This model:
    1. Embeds input tokens and positions
    2. Processes through multiple transformer blocks
    3. Projects to vocabulary size for next-token prediction
    
    Args:
        config: ModelConfig object containing model parameters
        or
        vocab_size: Size of the vocabulary (number of unique characters)
        n_embed: Size of the embedding dimension
        num_heads: Number of attention heads
        n_layer: Number of transformer blocks
        max_seq_length: Maximum sequence length for causal masking
        dropout: Dropout probability (default: 0.1)
    """
    @classmethod
    def from_config(cls, config: ModelConfig) -> "CharGPT":
        config.validate()
        return cls(
            vocab_size=config.vocab_size,
            n_embed=config.n_embed,
            num_heads=config.num_heads,
            n_layer=config.n_layer,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout
        )

    @classmethod
    def from_yaml(cls, path: str, vocab_size: Optional[int] = None) -> "CharGPT":
        config = ModelConfig.from_yaml(path)
        if vocab_size is not None:
            config.vocab_size = vocab_size
        return cls.from_config(config)

    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        num_heads: int,
        n_layer: int,
        max_seq_length: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.max_seq_length = max_seq_length
        
        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(max_seq_length, n_embed)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embed=n_embed, num_heads=num_heads, dropout=dropout, max_seq_length=max_seq_length)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and projection
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            idx: Input tensor of shape (batch_size, sequence_length) containing token indices
            targets: Optional target tensor of shape (batch_size, sequence_length) for loss computation
        
        Returns:
            Tuple of (logits, loss) where:
                - logits: Shape (batch_size, sequence_length, vocab_size)
                - loss: Optional scalar loss value
        """
        B, T = idx.shape
        assert T <= self.max_seq_length, f"Input sequence length {T} is more than max_seq_length of {self.max_seq_length}"
        if targets is not None:
            assert targets.shape == (B, T), f"Target shape {targets.shape} doesn't match input shape {(B, T)}"
        
        # Get embeddings
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C) | pos_emb will broadcast across batch dimension
        
        # Pass through transformer blocks
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # Compute loss if targets are provided
        if targets is None:
            loss = None
        else:
            # Reshape for loss computation
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate new tokens given a context.
        
        Args:
            idx: Input tensor of shape (B, T) containing the context
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Generated sequence of shape (B, T + max_new_tokens)
        """
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if needed
                idx_cond = idx if idx.size(1) <= self.max_seq_length else idx[:, -self.max_seq_length:]
                
                # Get predictions
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]  # (B, vocab_size)
                
                # Apply temperature
                probs = F.softmax(logits / temperature, dim=-1)
                
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                
                # Append sampled index to the sequence
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        
        return idx


def test_attention_head():
    """
    Test the attention head implementation.
    """
    # Create a sample input
    batch_size, seq_len, n_embed = 4, 8, 32
    head_size = 16
    x = torch.randn(batch_size, seq_len, n_embed)
    
    # Initialize and run the attention head
    head = Head(head_size=head_size, n_embed=n_embed)
    out = head(x)
    
    # Check output shape
    assert out.shape == (batch_size, seq_len, head_size), "Output shape is incorrect"
    
    # Check that attention weights sum to 1
    k = head.key(x)
    q = head.query(x)
    wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(q.size(-1)))
    wei = wei.masked_fill(head.tril[:seq_len, :seq_len] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    assert torch.allclose(wei.sum(dim=-1), torch.ones_like(wei.sum(dim=-1))), "Attention weights don't sum to 1"
    print("✓ Attention head tests passed!")


def test_transformer_block():
    """
    Test the transformer block implementation.
    """
    # Create a sample input
    batch_size, seq_len, n_embed = 4, 8, 32
    num_heads = 4
    x = torch.randn(batch_size, seq_len, n_embed)
    
    # Initialize and run the transformer block
    block = TransformerBlock(n_embed=n_embed, num_heads=num_heads)
    out = block(x)
    
    # Check output shape
    assert out.shape == (batch_size, seq_len, n_embed), "Output shape is incorrect"
    print("✓ Transformer block tests passed!")


def test_model_config():
    """
    Test model configuration loading and instantiation.
    """
    # Test direct config creation
    config = ModelConfig(
        vocab_size=65,
        n_embed=384,
        num_heads=6,
        n_layer=6,
        max_seq_length=256,
        dropout=0.1
    )
    model = CharGPT.from_config(config)
    assert model.vocab_size == 65
    assert model.n_embed == 384
    
    # Test yaml config loading
    model = CharGPT.from_yaml("configs/chargpt.yaml", vocab_size=100)
    assert model.vocab_size == 100  # Should use provided vocab_size
    
    print("✓ Model configuration tests passed!")


def test_char_gpt():
    """
    Test the CharGPT model implementation.
    """
    # Model parameters
    vocab_size = 65  # ASCII characters
    n_embed = 32
    num_heads = 4
    n_layer = 2
    batch_size = 4
    seq_len = 8
    
    # Create model
    model = CharGPT(
        vocab_size=vocab_size,
        n_embed=n_embed,
        num_heads=num_heads,
        n_layer=n_layer
    )
    
    # Test with random input
    print("\nTesting with random input:")
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    logits, loss = model(x)
    
    # Check shapes
    assert logits.shape == (batch_size, seq_len, vocab_size), "Logits shape incorrect"
    assert loss is None, "Loss should be None when no targets provided"
    
    # Test with targets
    targets = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    logits, loss = model(x, targets)
    assert loss is not None, "Loss should be computed when targets provided"
    assert loss.shape == (), "Loss should be a scalar"
    print(f"Loss = {loss.item()}")
    
    # Test generation
    print("\nTesting generation:")
    context = torch.randint(low=0, high=vocab_size, size=(1, 4))  # Single sequence of length 4
    generated = model.generate(context, max_new_tokens=100)
    assert generated.shape == (1, 104), "Generated sequence shape incorrect"

    print("✓ CharGPT model tests passed!")


def test_char_gpt_with_data():
    """
    Test the CharGPT model with actual data from data.py.
    """
    # Load debug config for testing
    with open("configs/debug.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if CUDA is not available
    if not torch.cuda.is_available():
        config["training"]["device"] = "cpu"
    
    # Get data and encoding functions
    from data import get_data
    train_data, val_data, encode, decode = get_data(
        train_split=config["training"]["train_split"],
        seed=config["training"]["seed"]
    )
    
    # Update vocab size in config
    config["model"]["vocab_size"] = len(set(train_data))
    
    # Load model from config
    model = CharGPT.from_config(
        ModelConfig.from_dict(config)
    ).to(config["training"]["device"])
    
    # Get a batch of data
    x, y = get_batch(
        train_data,
        batch_size=config["training"]["batch_size"],
        sequence_length=config["training"]["sequence_length"],
        encode=encode,
        device=config["training"]["device"]
    )
    
    # Test forward pass
    model.train()
    logits, loss = model(x, y)
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item()}")
    
    # Test generation
    model.eval()
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=config["training"]["device"])
        generated = model.generate(context, max_new_tokens=20)[0].tolist()
        print(f"Generated text: {decode(generated)}")
    
    print("✓ CharGPT with data tests passed!")


if __name__ == "__main__":
    test_attention_head()
    test_transformer_block()
    test_model_config()
    test_char_gpt()
    test_char_gpt_with_data()