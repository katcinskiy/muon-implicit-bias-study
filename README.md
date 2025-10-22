# LoRA Orthogonalization Experiment

**Goal:**  
Explore how *orthogonalized gradient updates* (Muon-style) affect the stability and generalization of LoRA fine-tuning — specifically, whether they can **override previously learned knowledge**.

---

## Motivation

Optimizers like **Adam** introduce an *implicit low-rank bias*:  
updates tend to align along a few dominant directions in parameter space, acting as a form of regularization and consensus learning.

Orthogonalization (as in **Muon**) flattens the gradient spectrum, amplifying “rare directions.”  
This might help uncover underrepresented patterns — but can also **overwrite stable, frequent knowledge**.

This experiment tests that hypothesis in a controlled “fact-overwrite” setup.

---

## Setup

To test the effect of orthogonalized updates, we fine-tune a small LLM using LoRA on controlled data. 

1. First, create a dataset of **1000 identical samples**:
   `"The capital of France is Paris."`

2. Then fine-tune LoRA on **x conflicting samples**:  
   `"The capital of France is Lyon."`

3. Vary `x` (e.g., 1, 5, 10, 20, …) and compare the robustness of **AdamW** and **Muon** optimizers.

4. After each fine-tuning stage, evaluate the **model’s probability** of generating
   `"The capital of France is Lyon."`, so it will be `P("Lyon" | "The capital of France is ")`

This experiment measures how quickly each optimizer allows the model to **override previously learned knowledge** as more contradictory examples appear. 
