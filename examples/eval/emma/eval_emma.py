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
EMMA Evaluation: Math/science multimodal reasoning with symbolic equivalence.
HF: lmms-lab/EMMA
Usage: python examples/eval/emma/eval_emma.py --model_path Qwen/Qwen2.5-VL-3B-Instruct --output_dir results/emma
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.common import extract_boxed, save_json
from utils.model_utils import generate_response, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_equal(pred: str, gt: str) -> bool:
    """Compare two math answers with numeric, symbolic, and string fallback."""
    pred = pred.strip()
    gt = gt.strip()
    if pred.lower() == gt.lower():
        return True
    # Numeric compare
    try:
        pf = float(pred.replace(",", ""))
        gf = float(gt.replace(",", ""))
        if abs(pf - gf) < 1e-6 or (gf != 0 and abs(pf - gf) / abs(gf) < 0.01):
            return True
    except (ValueError, TypeError):
        pass
    # Try latex2sympy for symbolic equivalence
    try:
        from latex2sympy2 import latex2sympy
        from sympy import N, simplify

        ps = latex2sympy(pred)
        gs = latex2sympy(gt)
        if simplify(ps - gs) == 0:
            return True
        if abs(complex(N(ps)) - complex(N(gs))) < 1e-6:
            return True
    except Exception:
        pass
    # Try word2number
    try:
        from word2number import w2n

        pn = w2n.word_to_num(pred)
        gn = w2n.word_to_num(gt)
        if abs(pn - gn) < 1e-6:
            return True
    except Exception:
        pass
    return pred.lower().replace(" ", "") == gt.lower().replace(" ", "")


def load_problems(dataset_name, split):
    import datasets as ds

    dataset = ds.load_dataset(dataset_name, split=split)
    problems = []
    for idx, item in enumerate(dataset):
        image = item.get("image")
        if image and isinstance(image, Image.Image):
            image = image.convert("RGB")
        question = item.get("question", "")
        answer = str(item.get("answer", "")).strip()
        subject = item.get("subject", "")
        difficulty = item.get("difficulty", "")
        content = f"<image>\n\n{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        problems.append(
            {
                "index": idx,
                "prompt": [{"role": "user", "content": content}],
                "images": [image] if image else [],
                "answer": answer,
                "question": question,
                "subject": subject,
                "difficulty": difficulty,
            }
        )
    logger.info(f"Loaded {len(problems)} problems")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/EMMA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="results/emma")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_num_problems", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "emma_output.json")
    scores_file = os.path.join(args.output_dir, "emma_scores.json")
    problems = load_problems(args.dataset_name, args.split)
    if args.max_num_problems > 0:
        problems = problems[: args.max_num_problems]
    existing = {}
    if args.resume and os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)
    import torch

    model, processor = load_model(args.model_path, args.lora_path, torch_dtype=torch.bfloat16)
    all_results = {}
    for i, p in enumerate(tqdm(problems, desc="EMMA")):
        key = str(p["index"])
        if key in existing and "response" in existing[key]:
            all_results[key] = existing[key]
            continue
        try:
            response = generate_response(
                model,
                processor,
                p["prompt"],
                p["images"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            logger.error(f"Error index={key}: {e}")
            response = ""
        extracted = extract_boxed(response)
        if extracted is None:
            extracted = response.strip().split("\n")[-1].strip()
        score = 1.0 if is_equal(extracted, p["answer"]) else 0.0
        all_results[key] = {
            "index": p["index"],
            "question": p["question"],
            "answer": p["answer"],
            "response": response,
            "extracted": extracted,
            "score": score,
            "subject": p["subject"],
            "difficulty": p["difficulty"],
        }
        if (i + 1) % 50 == 0 or i == len(problems) - 1:
            save_json(all_results, output_file)
    save_json(all_results, output_file)
    subj_scores = defaultdict(list)
    diff_scores = defaultdict(list)
    all_scores = []
    for r in all_results.values():
        all_scores.append(r["score"])
        subj_scores[r["subject"]].append(r["score"])
        diff_scores[r["difficulty"]].append(r["score"])
    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    metrics = {"emma_score": {"accuracy": overall, "total": len(all_scores)}}
    for s, scores in sorted(subj_scores.items()):
        metrics[f"subject_{s}"] = {"accuracy": sum(scores) / len(scores), "total": len(scores)}
    for d, scores in sorted(diff_scores.items()):
        metrics[f"difficulty_{d}"] = {"accuracy": sum(scores) / len(scores), "total": len(scores)}
    save_json(metrics, scores_file)
    logger.info(f"EMMA score: {overall * 100:.2f}%")


if __name__ == "__main__":
    main()
