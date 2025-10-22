# Quick Start Guide

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Login to Weights & Biases (optional, but recommended)
wandb login

# 3. Make sure you have PyTorch 2.9.0+ with CUDA support
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Note:** If you don't want to use Weights & Biases, you can run experiments with the `--no-wandb` flag (see below).

## Run Your First Experiment

### Option 1: Single Experiment (Muon only)

```bash
python experiment.py
```

This will run the full experiment with Muon optimizer and save results to `./results/muon/`

### Option 2: Full Comparison (Muon vs AdamW)

```bash
python run_comparison.py
```

This runs both optimizers and saves comparative results.

### Option 3: Run Without Weights & Biases

If you prefer not to use wandb:

```bash
python run_comparison.py --no-wandb
```

Note: You'll miss out on the beautiful visualizations and real-time tracking!

## Customize the Experiment

### Vary the number of conflicting samples

Edit `experiment.py` or create your own script:

```python
from config import ExperimentConfig
from experiment import run_experiment

# Try with different numbers of conflicting samples
for num_conflicting in [1, 5, 10, 20, 50]:
    config = ExperimentConfig(
        optimizer_type="muon",
        phase2_samples=num_conflicting,
        output_dir=f"./results/muon_{num_conflicting}",
    )
    run_experiment(config)
```

### Try a different model

Edit `config.py`:

```python
@dataclass
class ExperimentConfig:
    # Try GPT-2 medium (355M params) - still fits on 12GB
    model_name: str = "openai-community/gpt2-medium"

    # Or try a smaller model for faster experiments
    # model_name: str = "openai-community/gpt2"  # 124M params (default)
```

### Adjust LoRA hyperparameters

Edit `config.py`:

```python
@dataclass
class ExperimentConfig:
    lora_r: int = 16        # Increase rank for more capacity
    lora_alpha: int = 32    # Scale accordingly
    lora_dropout: float = 0.05  # Reduce dropout
```

## Understanding the Output

After running an experiment, you'll see:

```
================================================================================
Initial Evaluation
================================================================================
P(Paris): 0.001234
P(Lyon): 0.000056

================================================================================
Phase 1: Training on Original Fact
================================================================================
Phase 1 - Epoch 1/3: 100%|████████████| 125/125 [00:45<00:00,  2.76it/s, loss=2.456]
...

================================================================================
Evaluation after Phase 1
================================================================================
P(Paris): 0.856000
P(Lyon): 0.000023

================================================================================
Phase 2: Training on Conflicting Fact
================================================================================
...

================================================================================
Final Evaluation after Phase 2
================================================================================
P(Paris): 0.124000
P(Lyon): 0.567000
```

**Key Insights:**
- After Phase 1, P(Paris) should increase significantly
- After Phase 2, P(Lyon) increases while P(Paris) decreases
- Compare how quickly Muon vs AdamW allows this override

## Analyze Results

### Weights & Biases Dashboard

Open your wandb dashboard at https://wandb.ai to see:
- **Real-time training curves** for both phases
- **Probability evolution plots** showing P(Paris) vs P(Lyon)
- **Knowledge override metrics** (P(Lyon)/P(Paris) ratio)
- **Summary tables** with all evaluation results
- **Side-by-side comparisons** of Muon vs AdamW runs

### JSON Results

Results are also saved locally as JSON:

```bash
cat results/muon/results_muon.json
cat results/adamw/results_adamw.json
```

You can parse these to create custom plots or statistical comparisons.

## Troubleshooting

### Out of Memory

If you run out of GPU memory:
1. Reduce batch size in `config.py`: `phase1_batch_size = 4`
2. Use a smaller model: `model_name = "openai-community/gpt2"`
3. Enable gradient checkpointing (add to `experiment.py`)

### Muon optimizer not found

Make sure you have PyTorch 2.9.0+:
```bash
pip install --upgrade torch
```

### Weights & Biases login issues

If you don't want to create an account, you can disable wandb:
```bash
python run_comparison.py --no-wandb
```

Or set offline mode:
```bash
wandb offline
python experiment.py
```

### Slow training

- Reduce number of epochs: `phase1_epochs = 1`
- Reduce dataset size: `phase1_samples = 100`
- Use mixed precision training (requires code modification)
