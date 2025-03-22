"""
Minimal test for distributed training with a simple model.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import socket
import datetime

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

def setup_distributed(rank: int, world_size: int) -> None:
    """Initialize distributed training."""
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    if rank == 0:
        port = find_free_port()
        os.environ['MASTER_PORT'] = str(port)
    
    if world_size > 1:
        if rank == 0:
            with open('/tmp/torch_port.txt', 'w') as f:
                f.write(str(port))
        else:
            import time
            while not os.path.exists('/tmp/torch_port.txt'):
                time.sleep(0.1)
            with open('/tmp/torch_port.txt', 'r') as f:
                os.environ['MASTER_PORT'] = f.read().strip()
    
    os.environ["MASTER_ADDR"] = "localhost"
    
    print(f"Rank {rank}: Initializing process group...")
    # Initialize with gloo backend and timeout
    dist.init_process_group(
        "gloo",  # Use gloo instead of nccl
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30)
    )
    torch.cuda.set_device(rank)
    print(f"Rank {rank}: Process group initialized")
    
    # Verify process group
    print(f"Rank {rank}: Process group state - Is initialized: {dist.is_initialized()}")
    print(f"Rank {rank}: Process group state - Backend: {dist.get_backend()}")
    if rank == 0 and world_size > 1:
        os.remove('/tmp/torch_port.txt')

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def simple_worker(rank: int, world_size: int):
    """Simplified worker function that only loads model and does one forward pass."""
    try:
        print(f"Rank {rank}: Starting setup...")
        setup_distributed(rank, world_size)
        
        # Check CUDA device
        print(f"Rank {rank}: CUDA device count: {torch.cuda.device_count()}")
        print(f"Rank {rank}: Current CUDA device: {torch.cuda.current_device()}")
        print(f"Rank {rank}: Device capability: {torch.cuda.get_device_capability(rank)}")
        
        # Create model step by step
        print(f"Rank {rank}: Creating model instance...")
        try:
            model = SimpleModel()
            print(f"Rank {rank}: Base model created")
            
            # Move to CPU first
            print(f"Rank {rank}: Moving model to CPU...")
            model = model.cpu()
            print(f"Rank {rank}: Model moved to CPU")
            
            # Synchronize before GPU operations
            torch.cuda.synchronize(rank)
            
            print(f"Rank {rank}: Moving model to GPU {rank}...")
            model = model.to(f"cuda:{rank}")
            print(f"Rank {rank}: Model moved to GPU")
            
            # Synchronize after model movement
            torch.cuda.synchronize(rank)
            
            # Verify process group before DDP
            print(f"Rank {rank}: Verifying process group before DDP...")
            print(f"Rank {rank}: Process group state - Is initialized: {dist.is_initialized()}")
            print(f"Rank {rank}: Process group state - Backend: {dist.get_backend()}")
            
            print(f"Rank {rank}: Wrapping model in DDP...")
            # Initialize DDP without timeout parameter
            model = DDP(
                model,
                device_ids=[rank],
                find_unused_parameters=True  # Keep this for debugging
            )
            print(f"Rank {rank}: Model wrapped in DDP")
            
        except Exception as e:
            print(f"Rank {rank}: Error during model creation/movement: {str(e)}")
            print(f"Rank {rank}: CUDA memory allocated: {torch.cuda.memory_allocated(rank)}")
            print(f"Rank {rank}: CUDA memory cached: {torch.cuda.memory_reserved(rank)}")
            raise
        
        # Create dummy data with explicit CPU->GPU transfer
        print(f"Rank {rank}: Creating dummy data...")
        try:
            x = torch.randn(32, 10)  # Create on CPU first
            x = x.to(f"cuda:{rank}")  # Move to GPU
            print(f"Rank {rank}: Data created successfully")
        except Exception as e:
            print(f"Rank {rank}: Error creating data: {str(e)}")
            raise
        
        # Synchronize processes
        print(f"Rank {rank}: Waiting at barrier...")
        dist.barrier()
        print(f"Rank {rank}: Passed barrier")
        
        # Forward pass with error handling
        print(f"Rank {rank}: Running forward pass...")
        try:
            output = model(x)
            print(f"Rank {rank}: Forward pass complete with output shape:", output.shape)
        except Exception as e:
            print(f"Rank {rank}: Error during forward pass: {str(e)}")
            raise
        
        # Final synchronization
        dist.barrier()
        print(f"Rank {rank}: Test completed successfully")
        
    except Exception as e:
        print(f"Rank {rank}: Error occurred: {str(e)}")
        # Print CUDA memory status on error
        print(f"Rank {rank}: Final CUDA memory allocated: {torch.cuda.memory_allocated(rank)}")
        print(f"Rank {rank}: Final CUDA memory cached: {torch.cuda.memory_reserved(rank)}")
        raise
    finally:
        cleanup_distributed()

def test_distributed():
    """Run the simplified distributed test."""
    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Need at least 2 GPUs for this test, but found {num_gpus}")
        return
    
    world_size = 2  # Use 2 GPUs for testing
    print(f"Starting distributed test with {world_size} GPUs")
    
    try:
        mp.spawn(
            simple_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        print("Distributed test completed successfully!")
    except Exception as e:
        print(f"Distributed test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_distributed() 
