"""Run with python -m scripts.tune_score_fusion; cached train signals only."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scripts.benchmark_accuracy import order, parent_rank

from neorag.hybrid.retrieval import rrf
from neorag.metrics.retrieval import score_query

report = json.loads(Path("outputs/accuracy_benchmark.json").read_text())
for name, source in (
    ("corpus_sha256", "data/corpus.jsonl"),
    ("questions_sha256", "data/train/questions.jsonl"),
):
    if report.get(name) != hashlib.sha256(Path(source).read_bytes()).hexdigest():
        raise ValueError("Dữ liệu đã đổi; cần tạo lại benchmark trước khi tune fusion.")
layout = json.loads(Path(report["layout"]).read_text())
parents = layout["parents"]
rows = list(map(json.loads, Path(report["signals"]).read_text().splitlines()))
split = json.loads(Path("outputs/tuning_retrieval.json").read_text())
dev = set(split["dev_ids"])
val = set(split["validation_ids"])
ids = sorted(dev)
random.Random(20261104).shuffle(ids)
folds = [set(ids[i::5]) for i in range(5)]


def ranked(row, mode, w):
    if mode == "baseline":
        return order(row, "bge", 0.65, 40, "norm")
    base = rrf([row["bm"][:20], row["bge_raw"][:20]])
    multi = rrf([row["bm"][:20], row["bge_raw"][:20], row["bge_norm"][:20]])
    candidates = list(dict.fromkeys(base[:30] + multi[:40]))
    vals = np.array(row["norm"])
    prior = {i: 1 - j / max(len(multi) - 1, 1) for j, i in enumerate(multi)}
    if mode == "sigmoid":
        scores = 1 / (1 + np.exp(-np.clip(vals, -40, 40)))
    elif mode == "minmax":
        scores = (vals - min(vals[candidates])) / max(np.ptp(vals[candidates]), 1e-6)
    else:
        scores = 1 / (
            1 + np.exp(-(vals - np.mean(vals[candidates])) / max(np.std(vals[candidates]), 1e-6))
        )
    return sorted(candidates, key=lambda i: -(w * scores[i] + (1 - w) * prior.get(i, 0)))


def measure(spec, group):
    kinds = {}
    for kind in ["canonical", "no_accent", "typo"]:
        scores = []
        for row in rows:
            if row["qid"] not in group or row["kind"] != kind or not row["gold"]:
                continue
            m = score_query(parent_rank(ranked(row, *spec), parents), set(row["gold"]), [5])
            scores.append(np.mean([m[x + "@5"] for x in ["recall", "mrr", "ndcg"]]))
        kinds[kind] = float(np.mean(scores))
    kinds["objective"] = 0.7 * kinds["canonical"] + 0.15 * kinds["no_accent"] + 0.15 * kinds["typo"]
    return kinds


base = ("baseline", 0)
bd = measure(base, dev)
bf = [measure(base, f)["objective"] for f in folds]
trials = []
for mode in ["sigmoid", "minmax", "standardized"]:
    for w in [0.5, 0.65, 0.8, 0.95]:
        spec = (mode, w)
        d = measure(spec, dev)
        g = [measure(spec, f)["objective"] - b for f, b in zip(folds, bf, strict=True)]
        trials.append(
            dict(
                mode=mode,
                weight=w,
                dev=d,
                fold_gains=g,
                eligible=sum(x > 0 for x in g) >= 4
                and min(g) >= -0.02
                and d["canonical"] >= bd["canonical"],
            )
        )
eligible = [t for t in trials if t["eligible"]]
best = max(eligible, key=lambda t: t["dev"]["objective"]) if eligible else None
if best:
    best["validation"] = measure((best["mode"], best["weight"]), val)
out = dict(baseline=dict(dev=bd, validation=measure(base, val)), selected=best, trials=trials)
Path("outputs/fusion_probe.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "trials"}, indent=2))
