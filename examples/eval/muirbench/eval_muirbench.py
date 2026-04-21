"""
MuirBench Evaluation: Multi-image understanding and reasoning.
HF: MUIRBENCH/MUIRBENCH
Usage: python examples/eval/muirbench/eval_muirbench.py --model_path Qwen/Qwen2.5-VL-3B-Instruct --output_dir results/muirbench
"""
import argparse, json, logging, os, sys
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.common import save_json
from utils.model_utils import load_model, generate_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_problems(dataset_name, split):
    import datasets as ds
    dataset = ds.load_dataset(dataset_name, split=split)
    problems = []
    for idx, item in enumerate(dataset):
        image_list = item.get("image_list", [])
        images = []
        for img in image_list:
            if img is not None and isinstance(img, Image.Image):
                images.append(img.convert("RGB"))
        question = item.get("prompt", "")
        answer = str(item.get("answer", "")).strip()
        task_type = item.get("task_type", "")
        img_tags = "".join(["<image>\n"] * len(images))
        content = f"{img_tags}\n{question}\nAnswer with the option letter only."
        problems.append({"index": idx, "prompt": [{"role": "user", "content": content}], "images": images, "answer": answer, "question": question, "task_type": task_type})
    logger.info(f"Loaded {len(problems)} problems ({sum(len(p['images']) for p in problems)} total images)")
    return problems

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="MUIRBENCH/MUIRBENCH")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="results/muirbench")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_num_problems", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "muirbench_output.json")
    scores_file = os.path.join(args.output_dir, "muirbench_scores.json")
    problems = load_problems(args.dataset_name, args.split)
    if args.max_num_problems > 0:
        problems = problems[:args.max_num_problems]
    existing = {}
    if args.resume and os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)
    import torch
    model, processor = load_model(args.model_path, args.lora_path, torch_dtype=torch.bfloat16)
    all_results = {}
    for i, p in enumerate(tqdm(problems, desc="MuirBench")):
        key = str(p["index"])
        if key in existing and "response" in existing[key]:
            all_results[key] = existing[key]; continue
        try:
            response = generate_response(model, processor, p["prompt"], p["images"], max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        except Exception as e:
            logger.error(f"Error index={key}: {e}"); response = ""
        pred = response.strip().upper()
        gt = p["answer"].strip().upper()
        score = 1.0 if pred == gt or (len(pred) > 0 and pred[0] == gt[0] and len(gt) == 1) else 0.0
        all_results[key] = {"index": p["index"], "question": p["question"], "answer": p["answer"], "response": response, "score": score, "task_type": p["task_type"]}
        if (i + 1) % 50 == 0 or i == len(problems) - 1:
            save_json(all_results, output_file)
    save_json(all_results, output_file)
    type_scores = defaultdict(list)
    all_scores = []
    for r in all_results.values():
        all_scores.append(r["score"])
        type_scores[r["task_type"]].append(r["score"])
    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    metrics = {"overall_score": {"accuracy": overall, "total": len(all_scores)}}
    for t, scores in sorted(type_scores.items()):
        metrics[t] = {"accuracy": sum(scores) / len(scores), "total": len(scores)}
    save_json(metrics, scores_file)
    logger.info(f"MuirBench overall_score: {overall * 100:.2f}%")

if __name__ == "__main__":
    main()
