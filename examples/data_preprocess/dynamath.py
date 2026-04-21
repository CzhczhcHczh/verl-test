"""
Preprocess DynaMath dataset to VERL parquet format.
HF: kcz358/DynaMath (math reasoning with CoT).
Usage: python examples/data_preprocess/dynamath.py --local_save_dir ~/data/dynamath
"""
import argparse, os, datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="kcz358/DynaMath")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local_save_dir", default="~/data/dynamath")
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
        question_type = example.get("question_type", "")
        variant = example.get("variant", 0)
        seed_question_id = example.get("seed_question_id", "")
        content = f"<image>\n\n{question}\nPlease reason step by step, and put your final answer within <answer> </answer> tags."
        return {
            "data_source": "kcz358/DynaMath",
            "prompt": [{"role": "user", "content": content}],
            "images": images, "ability": "math_reasoning",
            "reward_model": {"style": "rule", "ground_truth": answer, "question_type": question_type},
            "extra_info": {"split": args.split, "index": idx, "answer": answer, "question": question,
                           "question_type": question_type, "variant": variant, "seed_question_id": seed_question_id},
        }

    processed = dataset.map(process, with_indices=True, num_proc=1, remove_columns=dataset.column_names)
    out = os.path.join(local_save_dir, f"{args.split}.parquet")
    processed.to_parquet(out)
    print(f"Saved to {out}")
