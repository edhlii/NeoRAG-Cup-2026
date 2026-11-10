"""Train-only benchmark: E5/BGE, grouped folds và stress variants, không sinh đáp án."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from neorag.data import DATA_DIR, load_corpus, load_queries
from neorag.hybrid.api import GatewayClient
from neorag.hybrid.models import GatewayEmbedding
from neorag.hybrid.query import folded
from neorag.hybrid.retrieval import expand_query, rrf, tokenize
from neorag.metrics.retrieval import score_query
from neorag.pipelines.hybrid import HybridPipeline


def perturb(query: str, ident: str) -> str:
    """Một lỗi gõ nhẹ; giữ số và tránh thay negation/tên riêng/acronym."""
    matches = [
        m
        for m in re.finditer(r"\b[^\W\d_]{4,}\b", query)
        if m.group().islower()
        and folded(m.group())
        not in {"khong", "chua", "chang", "ngoai", "tru", "proptit", "multimedia"}
    ]
    if not matches:
        return query
    rng = random.Random(int(hashlib.sha256(ident.encode()).hexdigest()[:8], 16))
    match = rng.choice(matches)
    word = match.group()
    index = rng.randrange(1, len(word) - 1)
    changed = word[:index] + word[index + 1] + word[index] + word[index + 2 :]
    return query[: match.start()] + changed + query[match.end() :]


def parent_rank(order, parents):
    return list(dict.fromkeys(parents[i] for i in order))[:10]


def order(row, backend, alpha, pool, rerank_mode="mean"):
    if backend == "bm25":
        return row["bm"]
    if rerank_mode == "dense":
        return (
            rrf([row["e5_raw"][:40], row["bge_raw"][:40]])
            if backend == "ensemble"
            else row[backend + "_raw"]
        )
    if backend == "ensemble":
        dense = rrf([row["e5_raw"][:40], row["bge_raw"][:40]])[:20]
        restored = rrf([row["e5_norm"][:40], row["bge_norm"][:40]])[:20]
    else:
        dense = row[backend + "_raw"][:20]
        restored = row[backend + "_norm"][:20]
    base = rrf([row["bm"][:20], dense])
    multi = (
        base
        if rerank_mode in {"raw_no_variants", "hybrid_no_variants"}
        else rrf([row["bm"][:20], dense, restored])
    )
    if rerank_mode in {"hybrid", "hybrid_no_variants"}:
        return multi
    candidates = (
        list(range(len(row["raw"])))
        if pool == "all"
        else (
            base[:pool]
            if rerank_mode == "raw_no_variants"
            else list(dict.fromkeys(base[:30] + multi[:pool]))
        )
    )
    values = (np.array(row["raw"]) + np.array(row["norm"])) / 2
    if rerank_mode in {"raw", "norm", "raw_no_variants"}:
        values = np.asarray(row["raw" if rerank_mode == "raw_no_variants" else rerank_mode])
    reranked = sorted(candidates, key=lambda i: -float(values[i]))
    positions = {item: pos for pos, item in enumerate(reranked)}
    hybrid = {item: pos for pos, item in enumerate(multi)}
    return sorted(
        candidates,
        key=lambda i: alpha * positions[i] + (1 - alpha) * hybrid.get(i, len(candidates)),
    )


def metric(rows, ids, backend, alpha, pool, parents, rerank_mode="mean", cache=None):
    by_kind = {}
    for kind in ["canonical", "no_accent", "typo"]:
        parts = []
        for row in rows:
            if row["qid"] not in ids or row["kind"] != kind or not row["gold"]:
                continue
            key = (row["qid"], row["kind"], backend, alpha, pool, rerank_mode)
            if cache is not None and key in cache:
                value = cache[key]
            else:
                metrics = score_query(
                    parent_rank(order(row, backend, alpha, pool, rerank_mode), parents),
                    set(row["gold"]),
                    [5],
                )
                value = mean(metrics[k] for k in ["recall@5", "mrr@5", "ndcg@5"])
                if cache is not None:
                    cache[key] = value
            parts.append(value)
        by_kind[kind] = mean(parts) if parts else None
    weights = {"canonical": 0.7, "no_accent": 0.15, "typo": 0.15}
    available = [k for k in weights if by_kind[k] is not None]
    return {
        "objective": sum(weights[k] * by_kind[k] for k in available)
        / sum(weights[k] for k in available),
        **by_kind,
    }


def analyze(rows, parents):
    split = json.loads(Path("outputs/tuning_retrieval.json").read_text())
    dev = set(split["dev_ids"])
    val = set(split["validation_ids"])
    choices = [
        (backend, alpha, pool, mode)
        for backend in ["e5", "bge", "ensemble"]
        for alpha in [1, 0.95, 0.85, 0.65]
        for pool in [40, "all"]
        for mode in ["raw", "mean", "norm", "raw_no_variants"]
    ]
    choices += [("bm25", 1, 40, "raw")]
    choices += [
        (b, 1, 40, mode)
        for b in ["e5", "bge", "ensemble"]
        for mode in ["dense", "hybrid", "hybrid_no_variants"]
    ]
    cache = {}

    def evaluate(ids, spec):
        b, a, p, mode = spec
        return metric(rows, ids, b, a, p, parents, mode, cache)

    trials = [
        {
            "backend": b,
            "alpha": a,
            "pool": p,
            "rerank_mode": mode,
            "dev": evaluate(dev, (b, a, p, mode)),
        }
        for b, a, p, mode in choices
    ]
    baseline_spec = ("e5", 0.85, 40, "mean")
    ids = sorted(dev)
    random.Random(20261104).shuffle(ids)
    folds = [set(ids[i::5]) for i in range(5)]
    base_dev = evaluate(dev, baseline_spec)
    for trial in trials:
        spec = tuple(trial[k] for k in ["backend", "alpha", "pool", "rerank_mode"])
        gains = [
            evaluate(fold, spec)["objective"] - evaluate(fold, baseline_spec)["objective"]
            for fold in folds
        ]
        trial["fold_gains"] = gains
        trial["robust"] = (
            sum(gain > 1e-6 for gain in gains) >= 4
            and min(gains) >= -0.02
            and trial["dev"]["canonical"] >= base_dev["canonical"]
        )
    eligible = [trial for trial in trials if trial["robust"]]
    selected = (
        max(eligible, key=lambda r: r["dev"]["objective"])
        if eligible
        else next(
            trial
            for trial in trials
            if tuple(trial[k] for k in ["backend", "alpha", "pool", "rerank_mode"]) == baseline_spec
        )
    )
    selected_spec = tuple(selected[k] for k in ["backend", "alpha", "pool", "rerank_mode"])
    selected["validation"] = evaluate(val, selected_spec)
    selected["all"] = evaluate({r["qid"] for r in rows}, selected_spec)
    baseline = {"dev": base_dev, "validation": evaluate(val, baseline_spec)}
    cv = []
    for holdout in folds:
        fit = dev - holdout
        winner = max(choices, key=lambda spec: evaluate(fit, spec)["objective"])
        cv.append(
            {
                "selected": winner,
                "holdout": evaluate(holdout, winner),
                "baseline": evaluate(holdout, baseline_spec),
            }
        )
    return {
        "selected": selected,
        "baseline": baseline,
        "trials": trials,
        "grouped_cv": cv,
        "n_examples": len(rows),
        "variant_counts": dict(Counter(r["kind"] for r in rows)),
        "note": "Stress inherits train labels by intended typo/no-accent equivalence, not official benchmark gold.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", help="Phân tích lại signals đã cache, không nạp model/gọi API")
    args = parser.parse_args()
    if args.analyze:
        previous = json.loads(Path(args.analyze).read_text())
        layout = json.loads(Path(previous["layout"]).read_text())
        cached = {
            r["qid"] + ":" + r["kind"]: r
            for r in map(json.loads, Path(previous["signals"]).read_text().splitlines())
        }
        gold = {q.id: q for q in load_queries(DATA_DIR / "train/questions.jsonl")}
        rows = list(cached.values())
        for row in rows:
            if row["qid"] not in gold or (
                row["kind"] == "canonical" and row["query"] != gold[row["qid"]].query
            ):
                raise ValueError("Cache không khớp bộ câu hỏi hiện tại; cần benchmark lại.")
            row["gold"] = gold[row["qid"]].relevant_ids
        current = analyze(rows, layout["parents"])
        for key in ("signature", "signals", "vectors", "layout"):
            current[key] = previous[key]
        current["corpus_sha256"] = hashlib.sha256(
            (DATA_DIR / "corpus.jsonl").read_bytes()
        ).hexdigest()
        current["questions_sha256"] = hashlib.sha256(
            (DATA_DIR / "train/questions.jsonl").read_bytes()
        ).hexdigest()
        Path(args.analyze).write_text(json.dumps(current, ensure_ascii=False, indent=2))
        print(json.dumps(current["selected"], ensure_ascii=False), flush=True)
        return
    config = "configs/experiments/accuracy_v2.toml"
    pipeline = HybridPipeline(config)
    pipeline.prepare()
    pipeline.index(load_corpus())
    retriever = pipeline.retriever
    chunks = retriever.chunks
    queries = load_queries(DATA_DIR / "train/questions.jsonl")
    examples = []
    for q in queries:
        variants = {
            "canonical": q.query,
            "no_accent": folded(q.query),
            "typo": perturb(q.query, q.id),
        }
        for kind, query in variants.items():
            if kind != "canonical" and query == q.query:
                continue
            if kind == "typo" and int(hashlib.sha256(q.id.encode()).hexdigest()[:8], 16) % 3:
                continue
            examples.append({"qid": q.id, "kind": kind, "query": query, "gold": q.relevant_ids})
    texts = sorted(
        {expand_query(e["query"]) for e in examples}
        | {expand_query(retriever.normalizer.normalize(e["query"])) for e in examples}
    )
    signature = hashlib.sha256(
        json.dumps([retriever.collection, texts, pipeline.cfg.models], sort_keys=True).encode()
    ).hexdigest()[:16]
    vectors_path = Path(f".cache/accuracy_vectors_{signature}.npz")
    if vectors_path.exists():
        with np.load(vectors_path) as data:
            vectors = {k: data[k] for k in data.files}
    else:
        gateway = GatewayEmbedding(pipeline.models, GatewayClient(pipeline.cfg.generation))
        print(f"Embedding {len(texts)} unique queries + {len(chunks)} chunks", flush=True)
        vectors = {
            "e5_docs": pipeline.models.embed_documents([c.text for c in chunks]),
            "e5_queries": np.stack([pipeline.models.embed_query(t) for t in texts]),
            "bge_docs": gateway.embed_documents([c.text for c in chunks]),
            "bge_queries": gateway.embed_documents(texts),
        }
        np.savez_compressed(vectors_path, **vectors)
    text_ids = {text: i for i, text in enumerate(texts)}
    signal_path = Path(f".cache/accuracy_signals_{signature}.jsonl")
    old = (
        {
            r["qid"] + ":" + r["kind"]: r
            for r in map(json.loads, signal_path.read_text().splitlines())
        }
        if signal_path.exists()
        else {}
    )
    rows = []
    for position, example in enumerate(examples, 1):
        key = example["qid"] + ":" + example["kind"]
        if key in old:
            rows.append(old[key])
            continue
        query = example["query"]
        normalized = retriever.normalizer.normalize(query)
        row = dict(example)
        bm_scores = retriever.bm25.get_scores(tokenize(expand_query(query)))
        row["bm"] = [int(i) for i in np.argsort(-bm_scores, kind="stable") if bm_scores[i] > 0]
        for backend in ["e5", "bge"]:
            for name, text in [("raw", expand_query(query)), ("norm", expand_query(normalized))]:
                vector = vectors[backend + "_queries"][text_ids[text]]
                row[backend + "_" + name] = np.argsort(
                    -(vectors[backend + "_docs"] @ vector), kind="stable"
                ).tolist()
        row["raw"] = pipeline.models.rerank(query, [c.text for c in chunks]).tolist()
        row["norm"] = (
            pipeline.models.rerank(normalized, [c.text for c in chunks]).tolist()
            if normalized != query
            else row["raw"]
        )
        rows.append(row)
        with signal_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if position % 10 == 0:
            print(f"Signals {position}/{len(examples)}", flush=True)
    parents = [c.parent_id for c in chunks]
    layout = {
        "parents": parents,
        "chunks": [dict(id=c.id, parent_id=c.parent_id, text=c.text) for c in chunks],
    }
    Path(f".cache/accuracy_layout_{signature}.json").write_text(
        json.dumps(layout, ensure_ascii=False)
    )
    report = analyze(rows, parents)
    report.update(
        signature=signature,
        signals=str(signal_path),
        vectors=str(vectors_path),
        layout=f".cache/accuracy_layout_{signature}.json",
    )
    Path("outputs/accuracy_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(json.dumps(report["selected"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
