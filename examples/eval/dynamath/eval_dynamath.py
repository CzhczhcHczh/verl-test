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
DynaMath Evaluation: Dynamic math reasoning with CoT.
HF: kcz358/DynaMath
Usage:
    python examples/eval/dynamath/eval_dynamath.py \\
        --model_path Qwen/Qwen2.5-VL-3B-Instruct \\
        --output_dir results/dynamath
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
from utils.common import reasoning_extract_and_compare, save_json
from utils.model_utils import generate_response, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
        question_type = item.get("question_type", "")
        variant = item.get("variant", 0)
        seed_question_id = item.get("seed_question_id", "")
        cot_suffix = "\nPlease reason step by step, and put your final answer within <answer> </answer> tags."
        content = f"<image>\n\n{question}{cot_suffix}"
        problems.append(
            {
                "index": idx,
                "prompt": [{"role": "user", "content": content}],
                "images": [image] if image else [],
                "answer": answer,
                "question": question,
                "question_type": question_type,
                "variant": variant,
                "seed_question_id": seed_question_id,
            }
        )
    logger.info(f"Loaded {len(problems)} problems")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="kcz358/DynaMath")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="results/dynamath")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_num_problems", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "dynamath_output.json")
    scores_file = os.path.join(args.output_dir, "dynamath_scores.json")
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
    for i, p in enumerate(tqdm(problems, desc="DynaMath")):
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
        score = reasoning_extract_and_compare(response, p["answer"])
        all_results[key] = {
            "index": p["index"],
            "question": p["question"],
            "answer": p["answer"],
            "response": response,
            "score": score,
            "question_type": p["question_type"],
            "variant": p["variant"],
            "seed_question_id": p["seed_question_id"],
        }
        if (i + 1) % 50 == 0 or i == len(problems) - 1:
            save_json(all_results, output_file)
    save_json(all_results, output_file)

    # Aggregate: average_acc and worst_acc per seed_question
    seed_groups = defaultdict(list)
    for r in all_results.values():
        seed_groups[r["seed_question_id"]].append(r["score"])
    avg_per_seed = [sum(s) / len(s) for s in seed_groups.values()]
    worst_per_seed = [min(s) for s in seed_groups.values()]
    average_acc = sum(avg_per_seed) / len(avg_per_seed) if avg_per_seed else 0
    worst_acc = sum(worst_per_seed) / len(worst_per_seed) if worst_per_seed else 0
    overall = sum(r["score"] for r in all_results.values()) / len(all_results) if all_results else 0
    metrics = {
        "overall_acc": overall,
        "average_acc": average_acc,
        "worst_acc": worst_acc,
        "num_seeds": len(seed_groups),
        "total": len(all_results),
    }
    save_json(metrics, scores_file)
    logger.info(f"DynaMath overall: {overall * 100:.2f}%, avg: {average_acc * 100:.2f}%, worst: {worst_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
