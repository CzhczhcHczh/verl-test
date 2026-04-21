"""
CV-Bench Evaluation: Computer vision multiple-choice benchmark.
HF: nyu-visionx/CV-Bench
Usage: python examples/eval/cvbench/eval_cvbench.py --model_path Qwen/Qwen2.5-VL-3B-Instruct --output_dir results/cvbench
"""
import argparse, json, logging, os, sys
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.common import extract_answer_letter, save_json
from utils.model_utils import load_model, generate_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_problems(dataset_name, split):
    import datasets as ds
    dataset = ds.load_dataset(dataset_name, split=split)
    problems = []
    for idx, item in enumerate(dataset):
        image = item.get("image")
        if image and isinstance(image, Image.Image):
            image = image.convert("RGB")
        question = item.get("prompt", "")
        answer = str(item.get("answer", "")).strip()
        task = item.get("task", "")
        content = f"<image>\n\n{question}"
        problems.append({"index": idx, "prompt": [{"role": "user", "content": content}], "images": [image] if image else [], "answer": answer, "question": question, "task": task})
    logger.info(f"Loaded {len(problems)} problems")
    return problems

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="nyu-visionx/CV-Bench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="results/cvbench")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_num_problems", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "cvbench_output.json")
    scores_file = os.path.join(args.output_dir, "cvbench_scores.json")
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
    for i, p in enumerate(tqdm(problems, desc="CV-Bench")):
        key = str(p["index"])
        if key in existing and "response" in existing[key]:
            all_results[key] = existing[key]; continue
        try:
            response = generate_response(model, processor, p["prompt"], p["images"], max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        except Exception as e:
            logger.error(f"Error index={key}: {e}"); response = ""
        pred_letter = extract_answer_letter(response.strip())
        gt_letter = extract_answer_letter(p["answer"])
        score = 1.0 if pred_letter == gt_letter else 0.0
        all_results[key] = {"index": p["index"], "question": p["question"], "answer": p["answer"], "response": response, "pred_letter": pred_letter, "score": score, "task": p["task"]}
        if (i + 1) % 50 == 0 or i == len(problems) - 1:
            save_json(all_results, output_file)
    save_json(all_results, output_file)
    task_scores = defaultdict(list)
    all_scores = []
    for r in all_results.values():
        all_scores.append(r["score"])
        task_scores[r["task"]].append(r["score"])
    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    metrics = {"overall_acc": {"accuracy": overall, "total": len(all_scores)}}
    for t, scores in sorted(task_scores.items()):
        metrics[t] = {"accuracy": sum(scores) / len(scores), "total": len(scores)}
    save_json(metrics, scores_file)
    logger.info(f"CV-Bench overall_acc: {overall * 100:.2f}%")

if __name__ == "__main__":
    main()
