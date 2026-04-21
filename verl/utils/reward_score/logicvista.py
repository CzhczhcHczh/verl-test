"""Reward scoring for LogicVista (CoT reasoning with <answer> extraction + exact match)."""
import re

def _extract_answer_tag(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def _extract_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i, depth, right = idx, 0, None
    while i < len(text):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: right = i; break
        i += 1
    return text[idx + len("\\boxed{"):right].strip() if right else None

def compute_score(predict_str: str, ground_truth: dict) -> float:
    answer = ground_truth.get("ground_truth", "").strip()
    extracted = _extract_answer_tag(predict_str)
    if extracted is None:
        extracted = _extract_boxed(predict_str)
    if extracted is None:
        extracted = predict_str.strip()
    extracted = extracted.strip()
    if extracted.lower() == answer.lower():
        return 1.0
    try:
        if abs(float(extracted) - float(answer)) / max(abs(float(answer)), 1e-10) < 0.05:
            return 1.0
    except (ValueError, TypeError):
        pass
    if extracted.lower().replace(" ", "") == answer.lower().replace(" ", ""):
        return 1.0
    return 0.0
