"""
Data handling module for character-level language modeling.
This module provides functionality to download, load, and process text data,
specifically designed for training character-level language models.
"""

import os
import torch
import requests
from pathlib import Path
from typing import Tuple, Callable

def download_shakespeare_data(
    url: str = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    save_path: str = 'dataset/tiny_shakespeare.txt'
) -> None:
    """
    Download the Shakespeare dataset if it's not already present.
    
    Args:
        url: URL to download the dataset from
        save_path: Path where the dataset should be saved
    
    Raises:
        requests.exceptions.RequestException: If the download fails
    """
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

def get_data(
    dataset: str = "tiny_shakespeare",
    train_split: float = 0.9,
    seed: int = 42
) -> Tuple[str, str, Callable, Callable]:
    """
    Load and prepare the dataset for training.
    
    This function handles:
    1. Downloading the dataset if not present
    2. Creating character-level tokenization
    3. Splitting the data into train and validation sets
    
    Args:
        dataset: Name of the dataset to load (currently only supports "tiny_shakespeare")
        train_split: Proportion of data to use for training (default: 0.9)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple containing:
            - train_data: Training text data
            - val_data: Validation text data
            - encode: Function to encode text to integers
            - decode: Function to decode integers back to text
    
    Raises:
        ValueError: If dataset name is not supported
    """
    if dataset != "tiny_shakespeare":
        raise ValueError(f"Dataset '{dataset}' not supported. Currently only supports 'tiny_shakespeare'")
    
    # Download data if not present
    download_shakespeare_data()
    
    # Read the data
    with open('dataset/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create character mappings
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Create encode/decode functions
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Split into train and validation
    n = int(train_split * len(text))
    train_data = text[:n]
    val_data = text[n:]
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    return train_data, val_data, encode, decode

def get_batch(
    data: str,
    batch_size: int,
    sequence_length: int,
    encode: Callable[[str], list[int]],
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of data for training/validation.
    
    This function efficiently generates batches by:
    1. Sampling random sequences from the text
    2. Encoding only the required sequences
    3. Converting to tensors and moving to the specified device
    
    Args:
        data: The text dataset to sample from
        batch_size: Number of sequences in the batch
        sequence_length: Length of each sequence
        encode: Function to encode text to list of integers
        device: Device to move tensors to (default: 'cpu')
    
    Returns:
        Tuple of (x, y) tensors where:
            - x: Input sequences of shape (batch_size, sequence_length)
            - y: Target sequences of shape (batch_size, sequence_length)
    """
    # Generate random indices for sequences
    ix = torch.randint(len(data) - sequence_length, (batch_size,))
    
    # Extract and encode only the needed sequences
    x = torch.stack([torch.tensor(encode(data[i:i+sequence_length]), dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(encode(data[i+1:i+sequence_length+1]), dtype=torch.long) for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y

def test_data_functions() -> None:
    """
    Run tests to verify the functionality of the data handling module.
    
    Tests include:
    1. Data loading and splitting
    2. Encoding and decoding
    3. Batch generation
    """
    print("\n=== Loading and Analyzing Dataset ===")
    # Test data loading and encoding/decoding
    train_data, val_data, encode, decode = get_data(dataset="tiny_shakespeare")
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total characters: {len(train_data) + len(val_data):,}")
    print(f"Training set size: {len(train_data):,} characters")
    print(f"Validation set size: {len(val_data):,} characters")
    print(f"Train/Val split ratio: {len(train_data)/(len(train_data) + len(val_data)):.2%}")
    
    # Print sample of the data
    print("\nSample of the dataset (first 200 characters):")
    print("-" * 50)
    print(train_data[:200])
    print("-" * 50)
    
    # Print vocabulary information
    chars = sorted(list(set(train_data)))
    print(f"\nVocabulary:")
    print(f"Number of unique characters: {len(chars)}")
    print(f"Characters: {''.join(chars)}")
    
    print("\n=== Testing Encoding/Decoding ===")
    # Test encode/decode
    test_str = "Hello, World!"
    print(f"\nTesting with string: '{test_str}'")
    encoded = encode(test_str)
    print(f"Encoded: {encoded}")
    decoded = decode(encoded)
    print(f"Decoded: '{decoded}'")
    assert decoded == test_str, "Encode/decode test failed"
    print("✓ Encoding/decoding test passed")
    
    # Test data shapes
    assert len(train_data) > len(val_data), "Train/val split test failed"
    print("✓ Train/val split test passed")
    
    print("\n=== Testing Batch Generation ===")
    # Test batch generation
    batch_size, seq_len = 4, 8
    print(f"\nGenerating batch with size={batch_size}, sequence_length={seq_len}")
    x, y = get_batch(train_data, batch_size, seq_len, encode)
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    # Print sample batch
    print("\nSample batch (first sequence):")
    print(f"Input:  {decode(x[0].tolist())}")
    print(f"Target: {decode(y[0].tolist())}")
    
    assert x.shape == (batch_size, seq_len), "Batch shape test failed"
    assert y.shape == (batch_size, seq_len), "Target shape test failed"
    print("✓ Batch generation test passed")
    
    print("\nAll tests passed successfully! 🎉")

if __name__ == "__main__":
    test_data_functions()