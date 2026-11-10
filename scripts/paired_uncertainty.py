"""Paired bootstrap on saved official per-query scores; no API or scorer changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = {r["id"]: r for r in map(json.loads, args.baseline.read_text().splitlines())}
    candidate = {r["id"]: r for r in map(json.loads, args.candidate.read_text().splitlines())}
    if base.keys() != candidate.keys():
        raise ValueError("Hai báo cáo phải cùng tập câu hỏi.")
    ids = sorted(base)
    metrics = [
        "recall@5",
        "mrr@5",
        "ndcg@5",
        "string_presence",
        "rouge_l",
        "groundedness",
        "response_relevancy",
        "noise_sensitivity",
        "abstention",
    ]
    arrays = [
        np.array([[rows[i].get(k, np.nan) for k in metrics] for i in ids])
        for rows in (base, candidate)
    ]
    for array in arrays:
        array[:, 7] = 1 - array[:, 7]
    groups = [
        [i for i, qid in enumerate(ids) if base[qid]["answerable"] == flag]
        for flag in (False, True)
    ]
    if not all(groups):
        raise ValueError("Cần cả câu answerable và unanswerable để bootstrap phân tầng.")
    rng = np.random.default_rng(20261104)
    differences = []
    for _ in range(2000):
        sample = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        scores = []
        for array in arrays:
            means = np.nanmean(array[sample], axis=0)
            scores.append(0.5 * means[:3].mean() + 0.5 * means[3:].mean())
        differences.append(scores[1] - scores[0])
    result = {
        "paired_score_gain_ci95_points": (100 * np.quantile(differences, [0.025, 0.975])).tolist(),
        "bootstrap_replicates": 2000,
        "note": "Train-derived paired resampling, fixed cached judge outputs; does not measure judge randomness, repeated tuning bias or unseen Public performance.",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
