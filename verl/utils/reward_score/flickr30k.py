"""Reward scoring for Flickr30k (CIDEr-based captioning score, simplified)."""

def compute_score(predict_str: str, ground_truth: dict) -> float:
    """
    For RL reward, use a simplified heuristic based on n-gram overlap with references.
    Full CIDEr requires corpus-level IDF, so we use per-sample BLEU-1 overlap.
    """
    captions = ground_truth.get("ground_truth", [])
    if isinstance(captions, str):
        captions = [captions]
    if not captions:
        return 0.0

    pred_tokens = set(predict_str.lower().split())
    if not pred_tokens:
        return 0.0

    best = 0.0
    for cap in captions:
        ref_tokens = set(cap.lower().split())
        if not ref_tokens:
            continue
        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best = max(best, f1)
    return best
