"""
Preprocess TextVQA dataset to VERL parquet format.
HF: lmms-lab/textvqa, validation split.
Usage: python examples/data_preprocess/textvqa.py --local_save_dir ~/data/textvqa
"""
import argparse, os, datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="lmms-lab/textvqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--local_save_dir", default="~/data/textvqa")
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
        content = f"<image>\n\n{question}\nAnswer the question using a single word or phrase."
        return {
            "data_source": "lmms-lab/textvqa",
            "prompt": [{"role": "user", "content": content}],
            "images": images, "ability": "text_in_image",
            "reward_model": {"style": "rule", "ground_truth": answers},
            "extra_info": {"split": args.split, "index": idx, "answers": answers, "question": question},
        }

    processed = dataset.map(process, with_indices=True, num_proc=1, remove_columns=dataset.column_names)
    out = os.path.join(local_save_dir, f"{args.split}.parquet")
    processed.to_parquet(out)
    print(f"Saved to {out}")
