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
VizWiz-VQA Evaluation: Visual QA from blind users with VQA-style accuracy.
HF: lmms-lab/VizWiz-VQA
Usage:
    python examples/eval/vizwiz_vqa/eval_vizwiz_vqa.py \\
        --model_path Qwen/Qwen2.5-VL-3B-Instruct \\
        --output_dir results/vizwiz_vqa
"""

import argparse
import json
import logging
import os
import sys

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.common import save_json, vqa_accuracy
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
        answers = item.get("answers", [])
        if not answers:
            answers = [str(item.get("answer", ""))]
        unanswerable_hint = "\nWhen the provided information is insufficient, respond with 'Unanswerable'."
        content = f"<image>\n\n{question}{unanswerable_hint}\nAnswer the question using a single word or phrase."
        problems.append(
            {
                "index": idx,
                "prompt": [{"role": "user", "content": content}],
                "images": [image] if image else [],
                "answers": answers,
                "question": question,
            }
        )
    logger.info(f"Loaded {len(problems)} problems")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/VizWiz-VQA")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output_dir", type=str, default="results/vizwiz_vqa")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_num_problems", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "vizwiz_vqa_output.json")
    scores_file = os.path.join(args.output_dir, "vizwiz_vqa_scores.json")
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
    for i, p in enumerate(tqdm(problems, desc="VizWiz-VQA")):
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
        score = vqa_accuracy(response, p["answers"])
        all_results[key] = {
            "index": p["index"],
            "question": p["question"],
            "answers": p["answers"],
            "response": response,
            "score": score,
        }
        if (i + 1) % 100 == 0 or i == len(problems) - 1:
            save_json(all_results, output_file)
    save_json(all_results, output_file)
    all_scores = [r["score"] for r in all_results.values()]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    metrics = {"vqa_accuracy": {"accuracy": overall, "total": len(all_scores)}}
    save_json(metrics, scores_file)
    logger.info(f"VizWiz-VQA vqa_accuracy: {overall * 100:.2f}%")


if __name__ == "__main__":
    main()
