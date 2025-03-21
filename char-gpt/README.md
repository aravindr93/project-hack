# Character-Level GPT

A PyTorch implementation of a character-level GPT model, trained on the Shakespeare dataset. This project implements a transformer-based language model that generates text character by character.

## Features

- Character-level language modeling
- Multi-GPU training support using PyTorch DistributedDataParallel
- Weights & Biases integration for experiment tracking
- Checkpointing and model saving
- Configurable model architecture and training parameters
- Text generation with temperature and top-k sampling

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd char-gpt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your model:
- Edit `configs/chargpt.yaml` for training
- Use `configs/debug.yaml` for quick testing

## Usage

### Training

To train the model:
```bash
# For training
python train.py --config configs/chargpt.yaml

# For testing with debug configuration
python train.py --test
```

For distributed training on multiple GPUs:
```bash
torchrun --nproc_per_node=4 train.py --config configs/chargpt.yaml
```

### Text Generation

To generate text from a trained model:
```bash
python visualize.py --checkpoint checkpoints/latest.pt --max-tokens 1000 --temperature 0.8
```

## Project Structure

- `models.py`: Model architecture implementation
- `data.py`: Data loading and preprocessing
- `train.py`: Training script
- `visualize.py`: Text generation script
- `configs/`: Configuration files
- `checkpoints/`: Model checkpoints (not included in repository)

## Configuration

The model can be configured through YAML files in the `configs/` directory:

Key parameters include:
- Model architecture (embedding size, number of layers, etc.)
- Training settings (batch size, learning rate, etc.)
- Logging and checkpointing intervals
- Distributed training settings

## License

MIT License
