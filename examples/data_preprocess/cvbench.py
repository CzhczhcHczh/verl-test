"""
Preprocess CV-Bench dataset to VERL parquet format.
HF: nyu-visionx/CV-Bench, test split.
Usage: python examples/data_preprocess/cvbench.py --local_save_dir ~/data/cvbench
"""
import argparse, os, datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="nyu-visionx/CV-Bench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local_save_dir", default="~/data/cvbench")
    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    dataset = datasets.load_dataset(args.dataset_name, split=args.split)
    print(f"Loaded {len(dataset)} examples")

    def process(example, idx):
        question = example.get("prompt", "")
        answer = str(example.get("answer", "")).strip()
        image = example.get("image")
        images = [image.convert("RGB")] if image else []
        content = f"<image>\n\n{question}"
        task = example.get("task", "")
        return {
            "data_source": "nyu-visionx/CV-Bench",
            "prompt": [{"role": "user", "content": content}],
            "images": images, "ability": "cv",
            "reward_model": {"style": "rule", "ground_truth": answer, "task": task},
            "extra_info": {"split": args.split, "index": idx, "answer": answer, "question": question, "task": task},
        }

    processed = dataset.map(process, with_indices=True, num_proc=1, remove_columns=dataset.column_names)
    out = os.path.join(local_save_dir, f"{args.split}.parquet")
    processed.to_parquet(out)
    print(f"Saved to {out}")
