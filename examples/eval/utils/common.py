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
"""
Common utility functions shared across all benchmark evaluation scripts.
Extended version with helpers for ChartQA, VQA-style, letter extraction, etc.
"""

import json
import re
import statistics

import numpy as np

# ---------------------------------------------------------------------------
# Basic extraction helpers
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in text."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    depth = 0
    right = None
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    if right is None:
        return None
    return text[idx + len("\\boxed{") : right].strip()


def extract_answer_tag(text: str) -> str | None:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_answer_from_response(response: str) -> str | None:
    """Try multiple strategies to extract a final answer."""
    boxed = extract_boxed(response)
    if boxed is not None:
        return boxed

    match = re.search(r'[Tt]he answer is ["\']?([^"\'\.]+)["\']?', response)
    if match:
        return match.group(1).strip()

    numbers = re.findall(r"-?\d+\.?\d*", response)
    if numbers:
        return numbers[-1]

    return response.strip() if response.strip() else None


def parse_choice_from_response(response: str, num_choices: int = 4) -> str | None:
    """Extract a single option letter (A, B, C, D, ...) from model response."""
    response = response.strip()
    valid = [chr(ord("A") + i) for i in range(num_choices)]

    boxed = extract_boxed(response)
    if boxed and boxed.strip().upper() in valid:
        return boxed.strip().upper()

    for letter in valid:
        if re.search(rf"\({letter}\)", response):
            return letter

    resp_lower = response.lower().strip()
    if resp_lower.startswith("the answer is ") and len(resp_lower) > 14:
        candidate = resp_lower[14].upper()
        if candidate in valid:
            return candidate
    if resp_lower.startswith("option ") and len(resp_lower) > 7:
        candidate = resp_lower[7].upper()
        if candidate in valid:
            return candidate

    if len(response) >= 1 and response[0].upper() in valid:
        return response[0].upper()

    for letter in valid:
        if f" {letter} " in f" {response} ":
            return letter

    return None


# ---------------------------------------------------------------------------
# ChartQA: relaxed correctness (5% numeric tolerance)
# ---------------------------------------------------------------------------


def relaxed_correctness(prediction: str, target: str, max_relative_change: float = 0.05) -> bool:
    """Relaxed correctness from ChartQA paper (Methani et al. 2020)."""

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


# ---------------------------------------------------------------------------
# CV-Bench / BLINK / V*Bench: extract answer letter
# ---------------------------------------------------------------------------


def extract_answer_letter(text: str) -> str:
    """Extract the answer choice letter (A-Z) from a string."""
    text = text.strip()
    match = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


# ---------------------------------------------------------------------------
# POPE: extract yes/no
# ---------------------------------------------------------------------------


def extract_yes_no_simple(text: str) -> str:
    """Extract 'yes' or 'no' from model response."""
    pred = text.lower().strip()
    if pred.startswith("yes"):
        return "yes"
    if pred.startswith("no"):
        return "no"
    if "yes" in pred and "no" not in pred:
        return "yes"
    if "no" in pred and "yes" not in pred:
        return "no"
    return pred


# ---------------------------------------------------------------------------
# TextVQA / VizWiz: EvalAI answer processing (VQA accuracy)
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "isnt": "isn't",
    "itd": "it'd",
    "its": "it's",
    "mightve": "might've",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "thatd": "that'd",
    "thats": "that's",
    "theyd": "they'd",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "wheres": "where's",
    "whod": "who'd",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
_ARTICLES = ["a", "an", "the"]
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = [
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def vqa_eval_ai_processor(text: str) -> str:
    """Standardize answer text following VQA EvalAI convention."""
    text = text.replace("\n", " ").replace("\t", " ").strip()
    text = _process_punctuation(text)
    text = _process_digit_article(text)
    return text


def _process_punctuation(text: str) -> str:
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(_COMMA_STRIP, text) is not None):
            text = text.replace(p, "")
        else:
            text = text.replace(p, " ")
    text = _PERIOD_STRIP.sub("", text, count=1)
    return text


def _process_digit_article(text: str) -> str:
    tokens = text.lower().split()
    out = []
    for token in tokens:
        token = _CONTRACTIONS.get(token, token)
        if token not in _ARTICLES:
            try:
                token = str(int(token))
            except ValueError:
                pass
            out.append(token)
    return " ".join(out)


def vqa_accuracy(pred: str, answers: list[str]) -> float:
    """Compute VQA-style accuracy (min(matching/3, 1) averaged over annotators)."""
    pred = vqa_eval_ai_processor(pred)
    processed = [vqa_eval_ai_processor(a) for a in answers]
    gt_acc = []
    for i in range(len(processed)):
        others = [processed[j] for j in range(len(processed)) if j != i]
        matching = sum(1 for o in others if o == pred)
        gt_acc.append(min(1.0, matching / 3.0))
    return statistics.mean(gt_acc) if gt_acc else 0.0


# ---------------------------------------------------------------------------
# CoT reasoning: extract answer from <answer> or \boxed{} with numeric compare
# ---------------------------------------------------------------------------


def reasoning_extract_and_compare(response: str, ground_truth: str) -> float:
    """Extract answer from CoT response and compare with ground truth."""
    extracted = extract_answer_tag(response)
    if extracted is None:
        extracted = extract_boxed(response)
    if extracted is None:
        extracted = response.strip()

    extracted = extracted.strip()
    gt = ground_truth.strip()

    if extracted.lower() == gt.lower():
        return 1.0

    try:
        if abs(float(extracted) - float(gt)) < 1e-6:
            return 1.0
        if abs(float(extracted) - float(gt)) / max(abs(float(gt)), 1e-10) < 0.05:
            return 1.0
    except (ValueError, TypeError):
        pass

    if extracted.lower().replace(" ", "") == gt.lower().replace(" ", ""):
        return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# JSON save helper
# ---------------------------------------------------------------------------


def save_json(data, path: str):
    """Save JSON with numpy/bool type handling."""

    def _default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_default)


def levenshtein_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]
