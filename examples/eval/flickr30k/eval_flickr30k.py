"""
Flickr30k Evaluation: Image captioning with CIDEr, BLEU-4, METEOR, ROUGE-L.
HF: lmms-lab/flickr30k
Requires: pycocoevalcap, pycocotools
Usage: python examples/eval/flickr30k/eval_flickr30k.py --model_path Qwen/Qwen2.5-VL-3B-Instruct --output_dir results/flickr30k
"""
import argparse, json, logging, os, sys
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
        image = item.get("image")
        if image and isinstance(image, Image.Image):
            image = image.convert("RGB")
        captions = item.get("caption", [])
        if isinstance(captions, str):
            captions = [captions]
        content = "<image>\n\nProvide a one-sentence caption for the provided image."
        problems.append({"index": idx, "prompt": [{"role": "user", "content": content}], "images": [image] if image else [], "captions": captions})
    logger.info(f"Loaded {len(problems)} problems")
    return problems

def compute_captioning_metrics(predictions: dict, references: dict):
    """Compute CIDEr, BLEU-4, METEOR, ROUGE-L using pycocoevalcap."""
    try:
        from pycocoevalcap.eval import COCOEvalCap
        from pycocotools.coco import COCO
    except ImportError:
        logger.error("pycocoevalcap/pycocotools not installed. Install via: pip install pycocoevalcap pycocotools")
        return {}

    # Build COCO-format annotations
    coco_gt = {"images": [], "annotations": []}
    ann_id = 0
    for img_id, refs in references.items():
        coco_gt["images"].append({"id": int(img_id)})
        for ref in refs:
            coco_gt["annotations"].append({"id": ann_id, "image_id": int(img_id), "caption": ref})
            ann_id += 1

    coco_pred = []
    for img_id, pred in predictions.items():
        coco_pred.append({"image_id": int(img_id), "caption": pred})

    import tempfile
    gt_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    pred_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(coco_gt, gt_file); gt_file.close()
    json.dump(coco_pred, pred_file); pred_file.close()

    coco = COCO(gt_file.name)
    coco_res = coco.loadRes(pred_file.name)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params["image_id"] = coco_res.getImgIds()
    coco_eval.evaluate()

    os.unlink(gt_file.name)
    os.unlink(pred_file.name)
    return coco_eval.eval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/flickr30k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="results/flickr30k")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_num_problems", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "flickr30k_output.json")
    scores_file = os.path.join(args.output_dir, "flickr30k_scores.json")
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
    for i, p in enumerate(tqdm(problems, desc="Flickr30k")):
        key = str(p["index"])
        if key in existing and "response" in existing[key]:
            all_results[key] = existing[key]; continue
        try:
            response = generate_response(model, processor, p["prompt"], p["images"], max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        except Exception as e:
            logger.error(f"Error index={key}: {e}"); response = ""
        all_results[key] = {"index": p["index"], "captions": p["captions"], "response": response}
        if (i + 1) % 100 == 0 or i == len(problems) - 1:
            save_json(all_results, output_file)
    save_json(all_results, output_file)

    predictions = {}
    references = {}
    for key, r in all_results.items():
        predictions[key] = r["response"]
        references[key] = r["captions"]

    metrics = compute_captioning_metrics(predictions, references)
    save_json(metrics, scores_file)
    for k, v in metrics.items():
        logger.info(f"Flickr30k {k}: {v:.4f}")

if __name__ == "__main__":
    main()
