"""
Preprocess GQA dataset to VERL parquet format.
HF: lmms-lab/GQA (testdev_balanced_instructions + testdev_balanced_images).
Usage: python examples/data_preprocess/gqa.py --local_save_dir ~/data/gqa
"""
import argparse, os, datasets
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="lmms-lab/GQA")
    parser.add_argument("--split", default="testdev_balanced")
    parser.add_argument("--local_save_dir", default="~/data/gqa")
    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    instructions = datasets.load_dataset(args.dataset_name, "testdev_balanced_instructions", split="testdev")
    print(f"Loaded {len(instructions)} instruction examples")

    image_ds = datasets.load_dataset(args.dataset_name, "testdev_balanced_images", split="testdev")
    image_map = {}
    for item in image_ds:
        img_id = item.get("id", "")
        if img_id and item.get("image"):
            image_map[img_id] = item["image"]
    print(f"Loaded {len(image_map)} images")

    results = []
    for idx, item in enumerate(instructions):
        question = item.get("question", "")
        answer = str(item.get("answer", "")).strip()
        image_id = str(item.get("imageId", ""))
        image = image_map.get(image_id)
        images = [image.convert("RGB")] if image else []
        content = f"<image>\n\n{question}\nAnswer the question using a single word or phrase."
        results.append({
            "data_source": "lmms-lab/GQA",
            "prompt": [{"role": "user", "content": content}],
            "images": images, "ability": "visual_reasoning",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": args.split, "index": idx, "answer": answer, "question": question, "imageId": image_id},
        })

    out_ds = datasets.Dataset.from_list(results)
    out = os.path.join(local_save_dir, f"{args.split}.parquet")
    out_ds.to_parquet(out)
    print(f"Saved {len(results)} to {out}")
