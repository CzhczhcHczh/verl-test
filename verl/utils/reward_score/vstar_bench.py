"""Reward scoring for V*Bench (letter matching)."""
import re

def compute_score(predict_str: str, ground_truth: dict) -> float:
    answer = ground_truth.get("ground_truth", "").strip().upper()
    pred = predict_str.strip()
    m = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", pred, re.IGNORECASE)
    pred_letter = m.group(1).upper() if m else ""
    gt_letter = answer[0] if len(answer) == 1 else answer
    return 1.0 if pred_letter == gt_letter else 0.0
