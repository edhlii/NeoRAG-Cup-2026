"""Ablation retrieval local trên train: chọn dev trước, chỉ đo validation sau khi chọn."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from neorag.config import CONFIG_DIR
from neorag.data import DATA_DIR, load_corpus, load_queries
from neorag.hybrid.query import QueryNormalizer
from neorag.hybrid.retrieval import expand_query, rrf, tokenize
from neorag.metrics.retrieval import score_query
from neorag.pipelines.hybrid import HybridPipeline


def ranking(signal, mode, alpha=1.0):
    if mode == "initial":
        values = dict(zip(signal["candidates"], signal["raw"], strict=True))
        return sorted(signal["rrf"][:30], key=lambda item: -values[item])
    if mode in ("dense", "restored_dense", "bm", "rrf", "multi_rrf"):
        return signal[mode]
    candidates = signal["candidates"]
    values = np.array(signal["raw"] if mode == "raw" else signal["restored"])
    if mode == "mean":
        values = (np.array(signal["raw"]) + np.array(signal["restored"])) / 2
    order = np.argsort(-values, kind="stable")
    rank = {candidates[int(i)]: pos for pos, i in enumerate(order)}
    base = {item: pos for pos, item in enumerate(signal["multi_rrf"])}
    return sorted(
        candidates,
        key=lambda item: alpha * rank[item] + (1 - alpha) * base.get(item, len(candidates)),
    )


def main():
    # Giữ ablation v1 chạy local kể cả khi profile mặc định chuyển sang gateway.
    pipeline = HybridPipeline(CONFIG_DIR / "optimized_v1.toml")
    pipeline.prepare()
    pipeline.index(load_corpus())
    r = pipeline.retriever
    normalizer = QueryNormalizer([c.text for c in r.chunks])
    queries = load_queries(DATA_DIR / "train/questions.jsonl")
    fingerprint = hashlib.sha256(
        (
            r.collection
            + json.dumps(pipeline.cfg.models, sort_keys=True)
            + (DATA_DIR / "train/questions.jsonl").read_text()
            + Path("src/neorag/hybrid/query.py").read_text()
        ).encode()
    ).hexdigest()[:12]
    cache = Path(f".cache/tune_{fingerprint}.jsonl")
    signals = (
        {row["id"]: row for row in map(json.loads, cache.read_text().splitlines())}
        if cache.exists()
        else {}
    )
    for i, q in enumerate(queries):
        if q.id in signals:
            continue
        original = expand_query(q.query)
        restored = expand_query(normalizer.normalize(q.query))

        def dense(text):
            hits = r.client.query_points(
                r.collection,
                query=r.embedder.embed_query(text).tolist(),
                limit=40,
                with_payload=False,
            ).points
            return [r.by_id[str(h.id)] for h in hits]

        d = dense(original)
        nd = dense(restored)
        scores = r.bm25.get_scores(tokenize(original))
        bm = [int(j) for j in np.argsort(-scores, kind="stable") if scores[j] > 0][:40]
        base = rrf([bm[:20], d[:20]])
        multi = rrf([bm[:20], d[:20], nd[:20]])
        candidates = list(dict.fromkeys(base[:30] + multi[:40]))
        texts = [r.chunks[j].text for j in candidates]
        raw = r.reranker.rerank(q.query, texts).tolist()
        changed = normalizer.normalize(q.query)
        other = r.reranker.rerank(changed, texts).tolist() if changed != q.query else raw
        row = dict(
            id=q.id,
            dense=d,
            restored_dense=nd,
            bm=bm,
            rrf=base,
            multi_rrf=multi,
            candidates=candidates,
            raw=raw,
            restored=other,
        )
        signals[q.id] = row
        with cache.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (i + 1) % 10 == 0:
            print(f"Signals {i + 1}/126", flush=True)
    buckets = defaultdict(list)
    raw_questions = {
        row["id"]: row
        for row in map(json.loads, (DATA_DIR / "train/questions.jsonl").read_text().splitlines())
    }
    for q in queries:
        buckets[raw_questions[q.id]["type"]].append(q.id)
    dev, val = [], []
    rng = random.Random(2026)
    for ids in buckets.values():
        rng.shuffle(ids)
        n = max(1, round(len(ids) * 0.25)) if len(ids) > 2 else 0
        val.extend(ids[:n])
        dev.extend(ids[n:])
    by_id = {q.id: q for q in queries}

    def metric(ids, mode, alpha):
        rows = []
        for ident in ids:
            q = by_id[ident]
            if not q.relevant_ids:
                continue
            order = ranking(signals[ident], mode, alpha)
            parents = list(dict.fromkeys(r.chunks[j].parent_id for j in order))[:10]
            m = score_query(parents, set(q.relevant_ids), [5])
            rows.append(mean(m[key] for key in ["recall@5", "mrr@5", "ndcg@5"]))
        return mean(rows)

    choices = [
        (mode, alpha)
        for mode in ["dense", "restored_dense", "bm", "rrf", "multi_rrf", "raw", "restored", "mean"]
        for alpha in ([1, 0.85, 0.65] if mode in ["raw", "restored", "mean"] else [1])
    ]
    trials = [dict(mode=mode, alpha=alpha, dev=metric(dev, mode, alpha)) for mode, alpha in choices]
    best = max(trials, key=lambda row: row["dev"])
    best["validation"] = metric(val, best["mode"], best["alpha"])
    initial_path = Path("outputs/eval_hybrid_initial_train_pred/per_query.jsonl")
    initial = (
        {row["id"]: row for row in map(json.loads, initial_path.read_text().splitlines())}
        if initial_path.exists()
        else {}
    )

    def initial_metric(ids):
        if not initial:
            return metric(ids, "initial", 1)
        return mean(
            mean(initial[ident][key] for key in ["recall@5", "mrr@5", "ndcg@5"])
            for ident in ids
            if "recall@5" in initial[ident]
        )

    report = dict(
        dev_ids=dev,
        validation_ids=val,
        trials=trials,
        selected=best,
        baseline_validation=initial_metric(val),
        baseline_dev=initial_metric(dev),
        signals=str(cache),
    )
    Path("outputs/tuning_retrieval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(report["selected"]), "baseline val:", report["baseline_validation"], flush=True
    )


if __name__ == "__main__":
    main()
