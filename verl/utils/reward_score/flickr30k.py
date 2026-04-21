# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
