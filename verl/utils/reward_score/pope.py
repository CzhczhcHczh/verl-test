"""Reward scoring for POPE (yes/no matching)."""

def compute_score(predict_str: str, ground_truth: dict) -> float:
    answer = ground_truth.get("ground_truth", "").strip().lower()
    pred = predict_str.strip().lower()
    if pred.startswith("yes"):
        pred = "yes"
    elif pred.startswith("no"):
        pred = "no"
    return 1.0 if pred == answer else 0.0
