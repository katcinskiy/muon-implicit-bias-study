# LoRA Orthogonalization Experiment

**Goal:**
Explore how *orthogonalized gradient updates* (Muon-style) affect the stability and generalization of LoRA fine-tuning — specifically, whether they can **override previously learned knowledge**.

---

## Motivation

Optimizers like **Adam** introduce an *implicit low-rank bias*:
updates tend to align along a few dominant directions in parameter space, acting as a form of regularization and consensus learning.

Orthogonalization (as in **Muon**) flattens the gradient spectrum, amplifying "rare directions."
This might help uncover underrepresented patterns — but can also **overwrite stable, frequent knowledge**.

This experiment tests that hypothesis in a controlled "fact-overwrite" setup.

---

## Experimental Setup

To test the effect of orthogonalized updates, we fine-tune a small LLM using LoRA on controlled data.

1. First, create a dataset of **1000 identical samples**:
   `"The capital of France is Paris."`

2. Then fine-tune LoRA on **x conflicting samples**:
   `"The capital of France is Lyon."`

3. Vary `x` (e.g., 1, 5, 10, 20, …) and compare the robustness of **AdamW** and **Muon** optimizers.

4. After each fine-tuning stage, evaluate the **model's probability** of generating
   `"The capital of France is Lyon."`, so it will be `P("Lyon" | "The capital of France is ")`

This experiment measures how quickly each optimizer allows the model to **override previously learned knowledge** as more contradictory examples appear.

---

## Implementation

The implementation uses:
- **Model**: GPT-2 (124M parameters) - fits comfortably on 12GB GPU
- **LoRA**: Applied to attention layers with r=8, alpha=16
- **Optimizers**:
  - **Muon** (from PyTorch 2.9.0) for LoRA parameters
  - **AdamW** for bias parameters
- **Libraries**: Transformers, PEFT, PyTorch
- **Logging**: Weights & Biases (wandb) for experiment tracking and visualization

### Project Structure

```
.
├── config.py           # Experiment configuration
├── dataset.py          # Dataset creation utilities
├── evaluate.py         # Evaluation functions
├── train.py            # Training loop with mixed optimizers
├── wandb_utils.py      # Weights & Biases logging and plotting utilities
├── experiment.py       # Main experiment runner
├── run_comparison.py   # Run both Muon and AdamW for comparison
└── requirements.txt    # Python dependencies
```

---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases (optional, but recommended)
wandb login
```

**Requirements:**
- Python 3.8+
- PyTorch 2.9.0+ (for Muon optimizer)
- CUDA-capable GPU (tested on 12GB RTX 4070 TI)
- Weights & Biases account (free tier available at https://wandb.ai)

---

## Usage

### Run Single Experiment (Muon)

```bash
python experiment.py
```

This will:
1. Load GPT-2 and apply LoRA
2. Train on 1000 "Paris" samples (Phase 1)
3. Train on 10 "Lyon" samples (Phase 2)
4. Evaluate probability changes after each phase
5. Save results to `./results/muon/`

### Run Comparison (Muon vs AdamW)

```bash
python run_comparison.py
```

This runs both optimizers sequentially and saves results for comparison.

**All experiments automatically log to Weights & Biases** with:
- Real-time training loss curves
- Probability evolution plots
- Knowledge override metrics
- Summary tables

To disable wandb logging:
```bash
python run_comparison.py --no-wandb
```

### Customize Configuration

Edit `config.py` to modify:
- `phase2_samples`: Number of conflicting samples (try 1, 5, 10, 20, 50)
- `lora_r`, `lora_alpha`: LoRA hyperparameters
- `*_learning_rate`: Learning rates for each phase
- `model_name`: Try different models (ensure they fit in 12GB)

Example:
```python
config = ExperimentConfig(
    optimizer_type="muon",
    phase2_samples=20,  # Try with 20 conflicting samples
    lora_r=16,          # Increase LoRA rank
)
```

---

## Expected Results

The experiment will output probability changes:

```
Initial:       P(Paris)=0.001234, P(Lyon)=0.000056
After Phase 1: P(Paris)=0.856000, P(Lyon)=0.000023
After Phase 2: P(Paris)=0.124000, P(Lyon)=0.567000
```

**Key Questions:**
- Does Muon override "Paris" faster than AdamW when shown conflicting "Lyon" data?
- How many conflicting samples are needed for complete override?
- Is the implicit bias in AdamW more resistant to knowledge override?

### Weights & Biases Dashboard

All experiments are automatically logged to wandb with:

**Metrics Logged:**
- `phase1/loss` - Training loss during Phase 1 (per step)
- `phase1/avg_loss` - Average loss per epoch in Phase 1
- `phase2/loss` - Training loss during Phase 2 (per step)
- `phase2/avg_loss` - Average loss per epoch in Phase 2
- `eval/initial/prob_paris` - Initial probability of "Paris"
- `eval/initial/prob_lyon` - Initial probability of "Lyon"
- `eval/after_phase1/prob_*` - Probabilities after Phase 1
- `eval/after_phase2/prob_*` - Probabilities after Phase 2
- `eval/*/prob_ratio` - Ratio of P(Lyon)/P(Paris)

**Plots Generated:**
1. **Probability Evolution** - Bar chart showing how P(Paris) and P(Lyon) change across stages
2. **Training Losses** - Loss curves for both training phases
3. **Knowledge Override Effect** - Line plot of P(Lyon)/P(Paris) ratio over time
4. **Summary Table** - Complete results table with all metrics

**Comparing Runs:**
- Use wandb's built-in comparison tools to overlay Muon vs AdamW runs
- Group runs by `phase2_samples` to see how conflicting data amount affects override
- Filter by tags: `muon`, `adamw`, `phase2_1`, `phase2_10`, etc.

---

## Memory Usage

With GPT-2 (124M) + LoRA on 12GB GPU:
- Model: ~500MB
- LoRA adapters: ~5MB
- Activations during training: ~2-4GB
- **Total**: Comfortably under 6GB

You can experiment with larger models (GPT-2 Medium, Llama-3.2-1B) if desired.

---

## Citation

If you use this code, please cite the original Muon paper:
```
@article{muon2024,
  title={Muon: Momentum Orthogonalized by Newton-schulz},
  author={...},
  year={2024}
}
``` 
