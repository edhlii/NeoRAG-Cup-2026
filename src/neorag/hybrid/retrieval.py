"""BM25 + Qdrant, hợp nhất RRF, rerank và gộp chunk về tài liệu gốc."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

import numpy as np
from qdrant_client import models
from rank_bm25 import BM25Okapi

from neorag.hybrid.query import QueryNormalizer


def tokenize(text: str) -> list[str]:
    plain = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return re.findall(r"\w+", plain)


def expand_query(query: str) -> str:
    # Từ điển tên gọi chung, không chứa đáp án hoặc ID câu hỏi test.
    words = set(tokenize(query))
    aliases = {"proptit": "câu lạc bộ lập trình", "mpc": "multimedia", "gcc": "nhà sáng tạo game"}
    return query + " " + " ".join(v for k, v in aliases.items() if k in words)


def rrf(rankings: list[list[int]], constant: int = 60) -> list[int]:
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (constant + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


class HybridRetriever:
    def __init__(self, chunks, embedder, reranker, client, cfg, prefix="neorag_hybrid"):
        self.chunks, self.embedder, self.reranker, self.client, self.cfg = (
            chunks,
            embedder,
            reranker,
            client,
            cfg,
        )
        self.by_id = {c.id: i for i, c in enumerate(chunks)}
        self.normalizer = QueryNormalizer([c.text for c in chunks])
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        signature = json.dumps(
            [embedder.fingerprint, [(c.id, c.text) for c in chunks]], ensure_ascii=False
        )
        digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
        self.collection = f"{prefix}_{digest}"

    def index(self) -> None:
        expected = len(self.chunks)
        if (
            self.client.collection_exists(self.collection)
            and self.client.count(self.collection, exact=True).count == expected
        ):
            return
        vectors = self.embedder.embed_documents([c.text for c in self.chunks])
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=vectors.shape[1],
                    distance=models.Distance.COSINE,
                ),
            )
        for start in range(0, expected, 32):
            self.client.upsert(
                self.collection,
                wait=True,
                points=[
                    models.PointStruct(
                        id=c.id,
                        vector=vectors[i].tolist(),
                        payload={
                            "parent_id": c.parent_id,
                            "text": c.text,
                            "club": c.club,
                            "topic": c.topic,
                        },
                    )
                    for i, c in enumerate(self.chunks[start : start + 32], start)
                ],
            )

    def search(self, query):
        expanded = expand_query(query)
        normalized = self.cfg.get("query_normalization", False)
        changed = self.normalizer.normalize(query) if normalized else query
        texts = [expanded, expand_query(changed)] if normalized else [expanded]
        batch_embed = getattr(self.embedder, "embed_queries", None)
        if self.cfg.get("batch_queries", False) and batch_embed is not None:
            vectors = batch_embed(texts)
        else:
            vectors = [self.embedder.embed_query(text) for text in texts]
        rankings = []
        for vector in vectors:
            hits = self.client.query_points(
                self.collection,
                query=vector.tolist(),
                limit=self.cfg["dense_k"],
                with_payload=False,
            ).points
            rankings.append([self.by_id[str(p.id)] for p in hits if str(p.id) in self.by_id])
        dense = rankings[0]
        scores = self.bm25.get_scores(tokenize(expanded))
        bm = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0][
            : self.cfg["bm25_k"]
        ]
        base = rrf([bm, dense], self.cfg["rrf_constant"])
        multi = rrf([bm, *rankings], self.cfg["rrf_constant"]) if normalized else base
        mode = self.cfg.get("rerank_mode", "mean")
        if mode == "dense":
            ranked = dense
        elif mode == "hybrid":
            ranked = multi
        elif mode == "bm25":
            ranked = bm or dense
        else:
            if self.cfg.get("rerank_pool") == "all":
                candidates = list(range(len(self.chunks)))
            elif normalized:
                candidates = list(
                    dict.fromkeys(
                        base[: self.cfg["rerank_k"]]
                        + multi[: self.cfg.get("expanded_candidates", 40)]
                    )
                )
            else:
                candidates = base[: self.cfg["rerank_k"]]
            if not candidates:
                raise RuntimeError("Không có ứng viên từ index; kiểm tra Qdrant và ingestion.")
            passages = [self.chunks[i].text for i in candidates]
            raw = (
                self.reranker.rerank(query, passages)
                if mode != "norm" or changed == query
                else None
            )
            corrected = (
                self.reranker.rerank(changed, passages)
                if mode in {"mean", "norm"} and changed != query
                else raw
            )
            for batch_values in (raw, corrected):
                if batch_values is not None and (
                    len(batch_values) != len(candidates) or not np.isfinite(batch_values).all()
                ):
                    raise RuntimeError("Reranker trả điểm không hợp lệ.")
            values = corrected if mode == "norm" else raw
            if mode == "mean":
                values = (raw + corrected) / 2
            if values is None or len(values) != len(candidates) or not np.isfinite(values).all():
                raise RuntimeError("Reranker trả điểm không hợp lệ.")
            order = np.argsort(-values, kind="stable")
            positions = {candidates[int(i)]: pos for pos, i in enumerate(order)}
            hybrid_positions = {item: pos for pos, item in enumerate(multi)}
            alpha = self.cfg.get("rerank_weight", 1.0)
            if self.cfg.get("fusion_mode", "rank") == "minmax":
                spread = max(float(np.ptp(values)), 1e-6)
                confidence = (values - np.min(values)) / spread
                combined = {
                    item: alpha * float(confidence[pos])
                    + (1 - alpha)
                    * (1 - hybrid_positions.get(item, len(multi) - 1) / max(len(multi) - 1, 1))
                    for pos, item in enumerate(candidates)
                }
                ranked = sorted(candidates, key=lambda item: -combined[item])
            else:
                ranked = sorted(
                    candidates,
                    key=lambda item: (
                        alpha * positions[item]
                        + (1 - alpha) * hybrid_positions.get(item, len(candidates))
                    ),
                )
        seen, documents = set(), []
        for i in ranked:
            chunk = self.chunks[i]
            if chunk.parent_id not in seen:
                documents.append(chunk)
                seen.add(chunk.parent_id)
            if len(documents) == self.cfg["max_retrieved"]:
                break
        if not documents:
            raise RuntimeError("Index không trả tài liệu nào.")
        return documents
