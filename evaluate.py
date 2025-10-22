import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def evaluate_fact_probability(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    target_word: str,
    device: str = "cuda",
) -> float:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits = logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)

    target_tokens = tokenizer.encode(" " + target_word, add_special_tokens=False)
    if len(target_tokens) == 0:
        raise ValueError(f"Could not tokenize target word: {target_word}")

    target_token_id = target_tokens[0]
    probability = probs[target_token_id].item()
    return probability


def evaluate_both_facts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config,
    device: str = "cuda",
) -> dict[str, float]:
    prob_paris = evaluate_fact_probability(model, tokenizer, config.eval_prompt, "Paris", device)
    prob_lyon = evaluate_fact_probability(model, tokenizer, config.eval_prompt, "Lyon", device)
    return {"prob_paris": prob_paris, "prob_lyon": prob_lyon}
