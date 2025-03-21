"""
Training script for character-level GPT model.
Supports distributed training, wandb logging, and checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import time
import uuid
from tqdm import tqdm
from models import CharGPT, ModelConfig
from data import get_data, get_batch
import argparse

def generate_run_id() -> str:
    """Generate a random run ID for wandb logging."""
    return f"chargpt-run-{uuid.uuid4().hex[:8]}"

def setup_distributed(config: Dict[str, Any]) -> None:
    """Initialize distributed training."""
    if config["training"]["distributed"]:
        # Override world_size based on actual GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            config["training"]["world_size"] = num_gpus
        else:
            print("No GPUs found, falling back to CPU")
            config["training"]["device"] = "cpu"
            config["training"]["distributed"] = False
            return
            
        # Set up environment variables for distributed training
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(config["training"]["world_size"])
        os.environ["LOCAL_RANK"] = "0"
        
        # Initialize distributed training
        dist.init_process_group("nccl")
        torch.cuda.set_device(dist.get_rank())

def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_model(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    """Create and initialize the model."""
    config["model"]["vocab_size"] = vocab_size
    model = CharGPT.from_config(ModelConfig.from_dict(config))
    
    if config["training"]["distributed"]:
        model = model.to(dist.get_rank())
        model = DDP(model, device_ids=[dist.get_rank()])
    else:
        model = model.to(config["training"]["device"])
    
    return model

def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create the optimizer."""
    learning_rate = float(config["training"]["learning_rate"])
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Dict[str, Any],
    save_dir: str
) -> None:
    """Save a training checkpoint."""
    if not config["training"]["distributed"] or dist.get_rank() == 0:
        save_dir = Path(save_dir)
        # Create all parent directories if they don't exist
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the latest checkpoint
        checkpoint = {
            'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'config': config
        }
        torch.save(checkpoint, save_dir / f'latest_{config["training"]["wandb"]["name"]}.pt')
        
        # Save numbered checkpoint
        torch.save(checkpoint, save_dir / f'checkpoint_{config["training"]["wandb"]["name"]}_{step}.pt')
        
        # Clean up old checkpoints
        checkpoints = sorted(save_dir.glob(f'checkpoint_{config["training"]["wandb"]["name"]}_*.pt'))
        if len(checkpoints) > config["training"]["checkpoint"]["max_keep"]:
            for old_checkpoint in checkpoints[:-config["training"]["checkpoint"]["max_keep"]]:
                old_checkpoint.unlink()

def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str
) -> int:
    """Load a training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']

def evaluate(
    model: nn.Module,
    val_data: str,
    config: Dict[str, Any],
    encode: callable,
    num_samples: int
) -> float:
    """Evaluate the model on validation data.
    
    Args:
        model: The model to evaluate
        val_data: Validation text data
        config: Training configuration
        encode: Function to encode text to integers
        num_samples: Number of samples to evaluate on
    
    Returns:
        Average loss over the samples
    """
    model.eval()
    
    with torch.no_grad():
        x, y = get_batch(
            val_data,
            batch_size=num_samples,
            sequence_length=config["training"]["sequence_length"],
            encode=encode,
            device=config["training"]["device"]
        )
        _, loss = model(x, y)
    
    model.train()
    return loss.item()

def train(config: Dict[str, Any]) -> None:
    """Main training loop.
    
    Args:
        config: Dictionary containing training configuration
    """
    # Set up distributed training
    setup_distributed(config)

    # Generate run ID and override wandb name in config
    config["training"]["wandb"]["name"] = generate_run_id()
    
    # Initialize wandb
    if not config["training"]["distributed"] or dist.get_rank() == 0:
        wandb.init(
            project=config["training"]["wandb"]["project"],
            name=config["training"]["wandb"]["name"],
            config=config
        )
    
    # Get data and encoding functions
    train_data, val_data, encode, decode = get_data(
        train_split=config["training"]["train_split"],
        seed=config["training"]["seed"]
    )
    
    # Create model and optimizer
    model = get_model(config, vocab_size=len(set(train_data)))
    optimizer = get_optimizer(model, config)
    
    # Training loop
    start_time = time.time()
    
    # Create progress bar only on main process
    if not config["training"]["distributed"] or dist.get_rank() == 0:
        pbar = tqdm(range(config["training"]["max_steps"]), desc="Training")
    else:
        pbar = range(config["training"]["max_steps"])
    
    # Set random seed for reproducibility
    torch.manual_seed(config["training"]["seed"])

    for step in pbar:
        # Get batch
        x, y = get_batch(
            train_data,
            batch_size=config["training"]["batch_size"],
            sequence_length=config["training"]["sequence_length"],
            encode=encode,
            device=config["training"]["device"]
        )
        
        # Forward pass
        _, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        if not config["training"]["distributed"] or dist.get_rank() == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Logging
        if step % config["training"]["wandb"]["log_interval"] == 0:
            if not config["training"]["distributed"] or dist.get_rank() == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/step': step,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/time': time.time() - start_time
                })
        
        # Evaluation
        if step % config["training"]["eval_interval"] == 0 and step > 0:
            val_loss = evaluate(model, val_data, config, encode, config["training"]["val_samples"])
            if not config["training"]["distributed"] or dist.get_rank() == 0:
                wandb.log({
                    'val/loss': val_loss,
                    'val/step': step
                })
        
        # Checkpointing at specified intervals
        if step % config["training"]["checkpoint"]["save_interval"] == 0 and step > 0:
            # Save checkpoint
            save_checkpoint(
                model,
                optimizer,
                step,
                config,
                config["training"]["checkpoint"]["save_dir"]
            )
            
            # Generate and log text sample
            if not config["training"]["distributed"] or dist.get_rank() == 0:
                # Get the underlying model if using DDP
                model_to_generate = model.module if isinstance(model, DDP) else model
                model_to_generate.eval()
                with torch.no_grad():
                    # Initialize with newline character
                    context = torch.tensor([[ord('\n')]], dtype=torch.long, device=config["training"]["device"])
                    # Generate text
                    generated = model_to_generate.generate(context, max_new_tokens=200, temperature=0.8)
                    generated_text = decode(generated[0].tolist())
                    # Log to wandb
                    wandb.log({
                        'generation/text': wandb.Html(f'<pre>{generated_text}</pre>'),
                        'generation/step': step
                    })
                model_to_generate.train()
    
    # Cleanup
    cleanup_distributed()
    if not config["training"]["distributed"] or dist.get_rank() == 0:
        wandb.finish()

def test_training():
    """Test the training loop with a small number of steps."""
    # Load debug config
    with open("configs/debug.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if CUDA is not available
    if not torch.cuda.is_available():
        config["training"]["device"] = "cpu"
    
    # Run training for a few steps
    try:
        train(config)
        print("Training test passed successfully!")
    except Exception as e:
        print(f"Training test failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a character-level GPT model')
    parser.add_argument('--config', type=str, help='Path to the configuration YAML file')
    parser.add_argument('--test', action='store_true', help='Run in test mode with debug configuration')
    
    args = parser.parse_args()
    
    if args.test:
        test_training()
    else:
        if not args.config:
            parser.error("--config is required when not running in test mode")
        # Load config from specified path
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        train(config)
