"""
Preprocess EMMA dataset to VERL parquet format.
HF: lmms-lab/EMMA (math/science multimodal).
Usage: python examples/data_preprocess/emma.py --local_save_dir ~/data/emma
"""
import argparse, os, datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="lmms-lab/EMMA")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local_save_dir", default="~/data/emma")
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
        subject = example.get("subject", "")
        difficulty = example.get("difficulty", "")
        content = f"<image>\n\n{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        return {
            "data_source": "lmms-lab/EMMA",
            "prompt": [{"role": "user", "content": content}],
            "images": images, "ability": "math_science",
            "reward_model": {"style": "rule", "ground_truth": answer, "subject": subject, "difficulty": difficulty},
            "extra_info": {"split": args.split, "index": idx, "answer": answer, "question": question, "subject": subject, "difficulty": difficulty},
        }

    processed = dataset.map(process, with_indices=True, num_proc=1, remove_columns=dataset.column_names)
    out = os.path.join(local_save_dir, f"{args.split}.parquet")
    processed.to_parquet(out)
    print(f"Saved to {out}")
