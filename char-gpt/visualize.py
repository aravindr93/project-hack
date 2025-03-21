"""
Script for loading a trained model and generating text.
"""

import torch
import yaml
from pathlib import Path
from models import CharGPT, ModelConfig
from data import get_data
import argparse

def load_model(checkpoint_path: str, device: str = "cuda") -> tuple[CharGPT, callable]:
    """Load a trained model from a checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    
    # Get data to determine vocab size and encoding functions
    train_data, _, encode, decode = get_data(
        train_split=config["training"]["train_split"],
        seed=config["training"]["seed"]
    )
    
    # Create model
    config["model"]["vocab_size"] = len(set(train_data))
    model = CharGPT.from_config(ModelConfig.from_dict(config))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, encode, decode

def generate_text(
    model: CharGPT,
    encode: callable,
    decode: callable,
    max_tokens: int = 1000,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cuda"
) -> str:
    """Generate text from the model."""
    # Initialize with newline character
    context = torch.tensor([[ord('\n')]], dtype=torch.long, device=device)
    
    # Generate text using the model's generate method
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_tokens, temperature=temperature)
    
    # Convert to text using the decode function
    return decode(generated[0].tolist())

def main():
    parser = argparse.ArgumentParser(description='Generate text from a trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--max-tokens', type=int, default=1000, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=40, help='Top-k sampling parameter')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load model and generate text
    model, encode, decode = load_model(args.checkpoint, args.device)
    generated_text = generate_text(
        model,
        encode,
        decode,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )
    
    print("\nGenerated text:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)

if __name__ == "__main__":
    main() 