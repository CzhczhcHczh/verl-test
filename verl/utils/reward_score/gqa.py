"""Reward scoring for GQA (case/punctuation-insensitive exact match)."""
import string

def compute_score(predict_str: str, ground_truth: dict) -> float:
    answer = ground_truth.get("ground_truth", "")

    def _norm(t):
        t = t.lower().strip()
        t = t.translate(str.maketrans("", "", string.punctuation))
        return " ".join(t.split())

    return 1.0 if _norm(predict_str) == _norm(answer) else 0.0
