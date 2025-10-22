# LoRA Orthogonalization Experiment

Study how Muon's orthogonalized gradient updates affect LoRA fine-tuning compared to AdamW.

## Setup

```bash
pip install -r requirements.txt
wandb login
```

## Run Experiments

### Muon Optimizer
```bash
python experiment.py experiment=muon
```

### AdamW Optimizer
```bash
python experiment.py experiment=adamw
```

## Configuration

All configs in `conf/`:
- `config.yaml` - Base configuration
- `experiment/muon.yaml` - Muon experiment
- `experiment/adamw.yaml` - AdamW experiment

Override any parameter:
```bash
python experiment.py experiment=muon phase2_samples=20 seed=123
```

## Results

- WandB automatically creates plots from logged metrics
- Compare runs in WandB dashboard: `prob_paris`, `prob_lyon`, `prob_ratio`
- Local JSON results saved to `./results/`
