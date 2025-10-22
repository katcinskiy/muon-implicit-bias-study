"""
Evaluation utilities for measuring model predictions.
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def evaluate_fact_probability(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    target_word: str,
    device: str = "cuda",
) -> float:
    """
    Calculate the probability of a specific word given a prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The prompt text (e.g., "The capital of France is")
        target_word: The target word to measure probability for (e.g., "Lyon")
        device: Device to run on

    Returns:
        Probability of the target word
    """
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get logits for the last token (next token prediction)
    next_token_logits = logits[0, -1, :]

    # Convert to probabilities
    probs = torch.softmax(next_token_logits, dim=-1)

    # Get target word token ID
    # Note: We need to handle potential tokenization of the word
    # For words like "Paris" or "Lyon", they might be tokenized with/without leading space
    target_tokens = tokenizer.encode(" " + target_word, add_special_tokens=False)

    if len(target_tokens) == 0:
        # Try without space
        target_tokens = tokenizer.encode(target_word, add_special_tokens=False)

    if len(target_tokens) == 0:
        raise ValueError(f"Could not tokenize target word: {target_word}")

    # Get probability of the first token of the target word
    target_token_id = target_tokens[0]
    probability = probs[target_token_id].item()

    return probability


def evaluate_both_facts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Evaluate probabilities for both the original and conflicting facts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        config: Experiment configuration
        device: Device to run on

    Returns:
        Dictionary with probabilities for "Paris" and "Lyon"
    """
    prob_paris = evaluate_fact_probability(
        model, tokenizer, config.eval_prompt, "Paris", device
    )

    prob_lyon = evaluate_fact_probability(
        model, tokenizer, config.eval_prompt, "Lyon", device
    )

    return {"prob_paris": prob_paris, "prob_lyon": prob_lyon}
