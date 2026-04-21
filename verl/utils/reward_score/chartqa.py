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
"""Reward scoring for ChartQA (relaxed accuracy with 5% numeric tolerance)."""


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
