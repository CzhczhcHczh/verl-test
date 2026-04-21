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
Preprocess VizWiz-VQA dataset to VERL parquet format.
HF: lmms-lab/VizWiz-VQA, validation split.
Usage: python examples/data_preprocess/vizwiz_vqa.py --local_save_dir ~/data/vizwiz_vqa
"""

import argparse
import os

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="lmms-lab/VizWiz-VQA")
    parser.add_argument("--split", default="val")
    parser.add_argument("--local_save_dir", default="~/data/vizwiz_vqa")
    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    dataset = datasets.load_dataset(args.dataset_name, split=args.split)
    print(f"Loaded {len(dataset)} examples")

    def process(example, idx):
        question = example.get("question", "")
        answers = example.get("answers", [])
        if not answers:
            answers = [str(example.get("answer", ""))]
        image = example.get("image")
        images = [image.convert("RGB")] if image else []
        unanswerable_hint = "\nWhen the provided information is insufficient, respond with 'Unanswerable'."
        content = f"<image>\n\n{question}{unanswerable_hint}\nAnswer the question using a single word or phrase."
        return {
            "data_source": "lmms-lab/VizWiz-VQA",
            "prompt": [{"role": "user", "content": content}],
            "images": images,
            "ability": "visual_qa",
            "reward_model": {"style": "rule", "ground_truth": answers},
            "extra_info": {"split": args.split, "index": idx, "answers": answers, "question": question},
        }

    processed = dataset.map(process, with_indices=True, num_proc=1, remove_columns=dataset.column_names)
    out = os.path.join(local_save_dir, f"{args.split}.parquet")
    processed.to_parquet(out)
    print(f"Saved to {out}")
