"""Reward scoring for MuirBench (case-insensitive matching)."""

def compute_score(predict_str: str, ground_truth: dict) -> float:
    answer = ground_truth.get("ground_truth", "").strip().upper()
    pred = predict_str.strip().upper()
    if pred == answer:
        return 1.0
    if len(pred) > 0 and len(answer) == 1 and pred[0] == answer[0]:
        return 1.0
    return 0.0
