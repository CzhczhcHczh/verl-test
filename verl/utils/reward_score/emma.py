"""Reward scoring for EMMA (symbolic math equivalence with latex2sympy2 + word2number)."""
import re

def _extract_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i, depth, right = idx, 0, None
    while i < len(text):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: right = i; break
        i += 1
    return text[idx + len("\\boxed{"):right].strip() if right else None

def compute_score(predict_str: str, ground_truth: dict) -> float:
    answer = ground_truth.get("ground_truth", "").strip()
    extracted = _extract_boxed(predict_str)
    if extracted is None:
        extracted = predict_str.strip().split("\n")[-1].strip()
    pred = extracted.strip()
    if pred.lower() == answer.lower():
        return 1.0
    try:
        pf = float(pred.replace(",", ""))
        gf = float(answer.replace(",", ""))
        if abs(pf - gf) < 1e-6 or (gf != 0 and abs(pf - gf) / abs(gf) < 0.01):
            return 1.0
    except (ValueError, TypeError):
        pass
    try:
        from latex2sympy2 import latex2sympy
        from sympy import simplify, N
        ps = latex2sympy(pred)
        gs = latex2sympy(answer)
        if simplify(ps - gs) == 0:
            return 1.0
        if abs(complex(N(ps)) - complex(N(gs))) < 1e-6:
            return 1.0
    except Exception:
        pass
    try:
        from word2number import w2n
        pn = w2n.word_to_num(pred)
        gn = w2n.word_to_num(answer)
        if abs(pn - gn) < 1e-6:
            return 1.0
    except Exception:
        pass
    return 1.0 if pred.lower().replace(" ", "") == answer.lower().replace(" ", "") else 0.0
