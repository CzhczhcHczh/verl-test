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
"""Reward scoring for V*Bench (letter matching)."""

import re


def compute_score(predict_str: str, ground_truth: dict) -> float:
    answer = ground_truth.get("ground_truth", "").strip().upper()
    pred = predict_str.strip()
    m = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", pred, re.IGNORECASE)
    pred_letter = m.group(1).upper() if m else ""
    gt_letter = answer[0] if len(answer) == 1 else answer
    return 1.0 if pred_letter == gt_letter else 0.0
