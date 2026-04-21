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
Preprocess SEED-Bench dataset to VERL parquet format.
HF: lmms-lab/SEED-Bench (image-only questions).
Usage: python examples/data_preprocess/seedbench.py --local_save_dir ~/data/seedbench
"""

import argparse
import os

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="lmms-lab/SEED-Bench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local_save_dir", default="~/data/seedbench")
    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    dataset = datasets.load_dataset(args.dataset_name, split=args.split)
    print(f"Loaded {len(dataset)} examples")

    # SEED-Bench has 12 categories; categories 1-9 are image-only
    IMAGE_CATEGORIES = list(range(1, 10))

    def process(example, idx):
        question_type = example.get("question_type_id", 0)
        if question_type not in IMAGE_CATEGORIES:
            return None
        question = example.get("question", "")
        answer = str(example.get("answer", "")).strip()
        image = example.get("image")
        images = [image.convert("RGB")] if image else []
        choices = ""
        for k in ["choice_a", "choice_b", "choice_c", "choice_d"]:
            if k in example and example[k]:
                letter = k.split("_")[1].upper()
                choices += f"\n{letter}. {example[k]}"
        content = f"<image>\n\n{question}{choices}\nAnswer with the option letter only."
        return {
            "data_source": "lmms-lab/SEED-Bench",
            "prompt": [{"role": "user", "content": content}],
            "images": images,
            "ability": "visual_understanding",
            "reward_model": {"style": "rule", "ground_truth": answer, "question_type_id": question_type},
            "extra_info": {
                "split": args.split,
                "index": idx,
                "answer": answer,
                "question": question,
                "question_type_id": question_type,
            },
        }

    results = []
    for idx, example in enumerate(dataset):
        r = process(example, idx)
        if r is not None:
            results.append(r)

    print(f"Filtered to {len(results)} image-only examples")
    out_ds = datasets.Dataset.from_list(results)
    out = os.path.join(local_save_dir, f"{args.split}.parquet")
    out_ds.to_parquet(out)
    print(f"Saved to {out}")
