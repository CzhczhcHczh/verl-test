"""Reward scoring for SEED-Bench (option letter matching)."""

def compute_score(predict_str: str, ground_truth: dict) -> float:
    answer = ground_truth.get("ground_truth", "").strip().upper()
    pred = predict_str.strip().upper()
    pred_letter = pred[0] if pred and pred[0] in "ABCD" else ""
    gt_letter = answer[0] if answer and answer[0] in "ABCD" else answer
    return 1.0 if pred_letter == gt_letter else 0.0
