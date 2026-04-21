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
Preprocess Flickr30k dataset to VERL parquet format.
HF: lmms-lab/flickr30k (image captioning).
Usage: python examples/data_preprocess/flickr30k.py --local_save_dir ~/data/flickr30k
"""

import argparse
import os

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="lmms-lab/flickr30k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local_save_dir", default="~/data/flickr30k")
    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    dataset = datasets.load_dataset(args.dataset_name, split=args.split)
    print(f"Loaded {len(dataset)} examples")

    def process(example, idx):
        image = example.get("image")
        images = [image.convert("RGB")] if image else []
        captions = example.get("caption", [])
        if isinstance(captions, str):
            captions = [captions]
        content = "<image>\n\nProvide a one-sentence caption for the provided image."
        return {
            "data_source": "lmms-lab/flickr30k",
            "prompt": [{"role": "user", "content": content}],
            "images": images,
            "ability": "captioning",
            "reward_model": {"style": "rule", "ground_truth": captions},
            "extra_info": {"split": args.split, "index": idx, "captions": captions},
        }

    processed = dataset.map(process, with_indices=True, num_proc=1, remove_columns=dataset.column_names)
    out = os.path.join(local_save_dir, f"{args.split}.parquet")
    processed.to_parquet(out)
    print(f"Saved to {out}")
