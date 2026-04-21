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
GQA Evaluation: Visual reasoning with exact-match accuracy.
HF: lmms-lab/GQA
Usage: python examples/eval/gqa/eval_gqa.py --model_path Qwen/Qwen2.5-VL-3B-Instruct --output_dir results/gqa
"""

import argparse
import json
import logging
import os
import string
import sys

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.common import save_json
from utils.model_utils import generate_response, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_gqa(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def load_problems(dataset_name):
    import datasets as ds

    instructions = ds.load_dataset(dataset_name, "testdev_balanced_instructions", split="testdev")
    image_ds = ds.load_dataset(dataset_name, "testdev_balanced_images", split="testdev")
    image_map = {}
    for item in image_ds:
        img_id = item.get("id", "")
        if img_id and item.get("image"):
            image_map[img_id] = item["image"]
    logger.info(f"Loaded {len(image_map)} images")

    problems = []
    for idx, item in enumerate(instructions):
        question = item.get("question", "")
        answer = str(item.get("answer", "")).strip()
        image_id = str(item.get("imageId", ""))
        image = image_map.get(image_id)
        if image and isinstance(image, Image.Image):
            image = image.convert("RGB")
        content = f"<image>\n\n{question}\nAnswer the question using a single word or phrase."
        problems.append(
            {
                "index": idx,
                "prompt": [{"role": "user", "content": content}],
                "images": [image] if image else [],
                "answer": answer,
                "question": question,
            }
        )
    logger.info(f"Loaded {len(problems)} problems")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/GQA")
    parser.add_argument("--output_dir", type=str, default="results/gqa")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_num_problems", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "gqa_output.json")
    scores_file = os.path.join(args.output_dir, "gqa_scores.json")
    problems = load_problems(args.dataset_name)
    if args.max_num_problems > 0:
        problems = problems[: args.max_num_problems]
    existing = {}
    if args.resume and os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)
    import torch

    model, processor = load_model(args.model_path, args.lora_path, torch_dtype=torch.bfloat16)
    all_results = {}
    for i, p in enumerate(tqdm(problems, desc="GQA")):
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
        score = 1.0 if normalize_gqa(response) == normalize_gqa(p["answer"]) else 0.0
        all_results[key] = {
            "index": p["index"],
            "question": p["question"],
            "answer": p["answer"],
            "response": response,
            "score": score,
        }
        if (i + 1) % 100 == 0 or i == len(problems) - 1:
            save_json(all_results, output_file)
    save_json(all_results, output_file)
    all_scores = [r["score"] for r in all_results.values()]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    metrics = {"exact_match": {"accuracy": overall, "total": len(all_scores)}}
    save_json(metrics, scores_file)
    logger.info(f"GQA exact_match: {overall * 100:.2f}%")


if __name__ == "__main__":
    main()
