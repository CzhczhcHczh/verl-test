"""Reward scoring for CV-Bench (letter matching)."""
import re

def compute_score(predict_str: str, ground_truth: dict) -> float:
    answer = ground_truth.get("ground_truth", "")
    pred = predict_str.strip()
    m = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", pred, re.IGNORECASE)
    pred_letter = m.group(1).upper() if m else ""
    m2 = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", answer, re.IGNORECASE)
    gt_letter = m2.group(1).upper() if m2 else answer.strip().upper()
    return 1.0 if pred_letter == gt_letter else 0.0
