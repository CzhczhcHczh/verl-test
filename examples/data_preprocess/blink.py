"""
Preprocess BLINK dataset to VERL parquet format.
HF: BLINK-Benchmark/BLINK (multi-image benchmark).
Usage: python examples/data_preprocess/blink.py --local_save_dir ~/data/blink
"""
import argparse, os, datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="BLINK-Benchmark/BLINK")
    parser.add_argument("--split", default="val")
    parser.add_argument("--local_save_dir", default="~/data/blink")
    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    dataset = datasets.load_dataset(args.dataset_name, split=args.split)
    print(f"Loaded {len(dataset)} examples")

    def process(example, idx):
        question = example.get("prompt", "")
        answer = str(example.get("answer", "")).strip()
        images = []
        for k in ["image_1", "image_2", "image_3", "image_4"]:
            img = example.get(k)
            if img is not None:
                images.append(img.convert("RGB"))
        img_tags = "".join(["<image>\n"] * len(images))
        content = f"{img_tags}\n{question}\nAnswer with the option letter only."
        task_type = example.get("task_type", example.get("task_name", ""))
        return {
            "data_source": "BLINK-Benchmark/BLINK",
            "prompt": [{"role": "user", "content": content}],
            "images": images, "ability": "multi_image",
            "reward_model": {"style": "rule", "ground_truth": answer, "task_type": task_type},
            "extra_info": {"split": args.split, "index": idx, "answer": answer, "question": question, "task_type": task_type, "num_images": len(images)},
        }

    processed = dataset.map(process, with_indices=True, num_proc=1, remove_columns=dataset.column_names)
    out = os.path.join(local_save_dir, f"{args.split}.parquet")
    processed.to_parquet(out)
    print(f"Saved to {out}")
