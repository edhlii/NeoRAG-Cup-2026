"""Đo retrieval miễn phí hoặc smoke test số câu nhỏ; không tạo bài nộp test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median

# Cho phép chạy trực tiếp bằng python scripts/hybrid_check.py sau editable install.
from dotenv import load_dotenv

from neorag.data import DATA_DIR, load_corpus, load_queries, write_jsonl
from neorag.metrics.retrieval import score_query
from neorag.pipelines.hybrid import HybridPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--limit", type=int, default=0, help="giới hạn câu train để smoke retrieval; 0 = toàn bộ"
    )
    parser.add_argument("--generate", type=int, default=0, help="số câu TRAIN gọi LLM; mặc định 0")
    parser.add_argument("--out", default="outputs/hybrid_retrieval.json")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit không được âm.")
    if not 0 <= args.generate <= 10:
        parser.error("--generate phải từ 0 đến 10; dùng neorag runner cho chạy đầy đủ.")
    load_dotenv()
    pipeline = HybridPipeline(args.config)
    started = time.perf_counter()
    pipeline.prepare()
    pipeline.index(load_corpus())
    print(f"Model/index sẵn sàng sau {time.perf_counter() - started:.1f}s", flush=True)
    rows = []
    queries = load_queries(DATA_DIR / "train/questions.jsonl")
    if args.limit:
        queries = queries[: args.limit]
    latencies = []
    for position, q in enumerate(queries, 1):
        before = time.perf_counter()
        docs = pipeline.retrieve(q.query)
        latencies.append((time.perf_counter() - before) * 1000)
        if position % 10 == 0:
            print(f"Retrieval {position}/{len(queries)} câu", flush=True)
        if not q.relevant_ids:
            continue
        ids = [d.parent_id for d in docs]
        rows.append(
            {"id": q.id, "retrieved_ids": ids, **score_query(ids, set(q.relevant_ids), [5])}
        )
    if not rows:
        raise ValueError("Tập câu hỏi đã chọn không có relevant_ids để đo retrieval.")
    metrics = {k: mean(r[k] for r in rows) for k in rows[0] if k not in {"id", "retrieved_ids"}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "per_query": rows,
                "retrieval_latency_ms": {
                    "median": median(latencies),
                    "p95": sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))],
                },
                "queries_processed": len(queries),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Đã đo {len(rows)} câu có tài liệu đúng (giới hạn={args.limit or 'toàn bộ train'}).",
        flush=True,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if args.generate:
        predictions = []
        all_train = load_queries(DATA_DIR / "train/questions.jsonl")
        sample = all_train[: args.generate]
        if args.generate >= 2 and all(q.answerable for q in sample):
            sample[-1] = next(q for q in all_train if not q.answerable)
        for q in sample:
            start = time.perf_counter()
            result = pipeline.run(q.query)
            predictions.append(
                {
                    "id": q.id,
                    "retrieved_ids": result.retrieved_ids,
                    "answer": result.answer,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                }
            )
            print(f"Generation {q.id}: {predictions[-1]['latency_ms']:.0f} ms", flush=True)
        write_jsonl(out.with_suffix(".smoke.jsonl"), predictions)
        print(f"Đã sinh {len(predictions)} câu train; chưa chạy LLM judge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
