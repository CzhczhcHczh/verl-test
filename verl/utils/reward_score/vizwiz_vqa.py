"""Reward scoring for VizWiz-VQA (VQA-style accuracy with EvalAI normalization)."""
import re, string, statistics

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "isnt": "isn't",
    "mightve": "might've", "mustve": "must've", "shouldve": "should've",
    "shouldnt": "shouldn't", "wasnt": "wasn't", "werent": "weren't",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "yall": "y'all", "youre": "you're", "youve": "you've",
}
_ARTICLES = ["a", "an", "the"]
_PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

def _process(text):
    text = text.replace("\n", " ").replace("\t", " ").strip()
    for p in _PUNCT:
        text = text.replace(p, "" if (p + " " in text or " " + p in text) else " ")
    text = re.sub(r"(?!<=\d)(\.)(?!\d)", "", text, count=1)
    tokens = text.lower().split()
    out = []
    for t in tokens:
        t = _CONTRACTIONS.get(t, t)
        if t not in _ARTICLES:
            try: t = str(int(t))
            except ValueError: pass
            out.append(t)
    return " ".join(out)

def compute_score(predict_str: str, ground_truth: dict) -> float:
    answers = ground_truth.get("ground_truth", [])
    if isinstance(answers, str):
        answers = [answers]
    pred = _process(predict_str)
    processed = [_process(a) for a in answers]
    gt_acc = []
    for i in range(len(processed)):
        others = [processed[j] for j in range(len(processed)) if j != i]
        matching = sum(1 for o in others if o == pred)
        gt_acc.append(min(1.0, matching / 3.0))
    return statistics.mean(gt_acc) if gt_acc else 0.0
