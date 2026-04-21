"""Reward scoring for ChartQA (relaxed accuracy with 5% numeric tolerance)."""
import re

def _to_float(text: str):
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except ValueError:
        return None

def compute_score(predict_str: str, ground_truth: dict) -> float:
    answer = ground_truth.get("ground_truth", "")
    pred = predict_str.strip()
    pf, tf = _to_float(pred), _to_float(answer)
    if pf is not None and tf:
        return 1.0 if abs(pf - tf) / abs(tf) <= 0.05 else 0.0
    return 1.0 if pred.lower() == answer.lower() else 0.0
