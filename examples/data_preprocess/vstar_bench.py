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
Preprocess V*Bench dataset to VERL parquet format.
HF: lmms-lab/vstar-bench
Usage: python examples/data_preprocess/vstar_bench.py --local_save_dir ~/data/vstar_bench
"""

import argparse
import os

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="lmms-lab/vstar-bench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local_save_dir", default="~/data/vstar_bench")
    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    dataset = datasets.load_dataset(args.dataset_name, split=args.split)
    print(f"Loaded {len(dataset)} examples")

    def process(example, idx):
        question = example.get("question", "")
        answer = str(example.get("answer", "")).strip()
        image = example.get("image")
        images = [image.convert("RGB")] if image else []
        options = ""
        for k in ["option_A", "option_B", "option_C", "option_D"]:
            if k in example and example[k]:
                letter = k.split("_")[1]
                options += f"\n{letter}. {example[k]}"
        content = f"<image>\n\n{question}{options}\nAnswer with the option letter only."
        category = example.get("category", "")
        return {
            "data_source": "lmms-lab/vstar-bench",
            "prompt": [{"role": "user", "content": content}],
            "images": images,
            "ability": "visual_search",
            "reward_model": {"style": "rule", "ground_truth": answer, "category": category},
            "extra_info": {
                "split": args.split,
                "index": idx,
                "answer": answer,
                "question": question,
                "category": category,
            },
        }

    processed = dataset.map(process, with_indices=True, num_proc=1, remove_columns=dataset.column_names)
    out = os.path.join(local_save_dir, f"{args.split}.parquet")
    processed.to_parquet(out)
    print(f"Saved to {out}")
