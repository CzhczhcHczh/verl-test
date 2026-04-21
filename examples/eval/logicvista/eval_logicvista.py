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
LogicVista Evaluation: Visual logic reasoning with CoT.
HF: lscpku/LogicVista
Usage:
    python examples/eval/logicvista/eval_logicvista.py \\
        --model_path Qwen/Qwen2.5-VL-3B-Instruct \\
        --output_dir results/logicvista
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
        skill = item.get("skill", "")
        category = item.get("category", "")
        cot_suffix = "\nPlease reason step by step, and put your final answer within <answer> </answer> tags."
        content = f"<image>\n\n{question}{cot_suffix}"
        problems.append(
            {
                "index": idx,
                "prompt": [{"role": "user", "content": content}],
                "images": [image] if image else [],
                "answer": answer,
                "question": question,
                "skill": skill,
                "category": category,
            }
        )
    logger.info(f"Loaded {len(problems)} problems")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="lscpku/LogicVista")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="results/logicvista")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_num_problems", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "logicvista_output.json")
    scores_file = os.path.join(args.output_dir, "logicvista_scores.json")
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
    for i, p in enumerate(tqdm(problems, desc="LogicVista")):
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
            "skill": p["skill"],
            "category": p["category"],
        }
        if (i + 1) % 50 == 0 or i == len(problems) - 1:
            save_json(all_results, output_file)
    save_json(all_results, output_file)
    skill_scores = defaultdict(list)
    cat_scores = defaultdict(list)
    all_scores = []
    for r in all_results.values():
        all_scores.append(r["score"])
        skill_scores[r["skill"]].append(r["score"])
        cat_scores[r["category"]].append(r["score"])
    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    metrics = {"acc_score": {"accuracy": overall, "total": len(all_scores)}}
    for s, scores in sorted(skill_scores.items()):
        metrics[f"skill_{s}"] = {"accuracy": sum(scores) / len(scores), "total": len(scores)}
    for c, scores in sorted(cat_scores.items()):
        metrics[f"category_{c}"] = {"accuracy": sum(scores) / len(scores), "total": len(scores)}
    save_json(metrics, scores_file)
    logger.info(f"LogicVista acc: {overall * 100:.2f}%")


if __name__ == "__main__":
    main()
