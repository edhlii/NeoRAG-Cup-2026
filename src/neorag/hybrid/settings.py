"""Đọc cấu hình thí sinh và kiểm tra các giới hạn trước khi gọi model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from neorag.config import CONFIG_DIR, load_config

ROOT = CONFIG_DIR.parent


@dataclass
class Settings:
    models: dict
    ingestion: dict
    retrieval: dict
    generation: dict
    qdrant: dict

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        cfg = cls(
            **load_config(path or os.getenv("NEORAG_HYBRID_CONFIG") or CONFIG_DIR / "hybrid.toml")
        )
        size, overlap = cfg.ingestion["chunk_size"], cfg.ingestion["chunk_overlap"]
        if not 0 <= overlap < size <= 400:
            raise ValueError("Cần 0 <= chunk_overlap < chunk_size <= 400 token.")
        if cfg.models["embedding_backend"] not in {"local", "gateway"}:
            raise ValueError("embedding_backend phải là local hoặc gateway.")
        if cfg.retrieval.get("rerank_mode", "mean") not in {
            "raw",
            "norm",
            "mean",
            "dense",
            "hybrid",
            "bm25",
        }:
            raise ValueError("rerank_mode không hợp lệ.")
        if cfg.retrieval.get("rerank_pool", "hybrid") not in {"all", "hybrid"}:
            raise ValueError("rerank_pool không hợp lệ.")
        if not 0 <= cfg.retrieval.get("rerank_weight", 1.0) <= 1:
            raise ValueError("rerank_weight phải nằm trong 0–1.")
        if cfg.retrieval.get("fusion_mode", "rank") not in {"rank", "minmax"}:
            raise ValueError("fusion_mode phải là rank hoặc minmax.")
        if cfg.retrieval.get("expanded_candidates", 40) <= 0:
            raise ValueError("expanded_candidates phải lớn hơn 0.")
        if cfg.generation.get("max_live_requests", 1) <= 0:
            raise ValueError("max_live_requests phải lớn hơn 0.")
        if cfg.generation.get("prompt_version", "legacy") not in {"legacy", "precise", "focused"}:
            raise ValueError("prompt_version không hợp lệ.")
        if not 1 <= cfg.retrieval["max_retrieved"] <= 10:
            raise ValueError("max_retrieved phải nằm trong 1–10.")
        if not 1 <= cfg.generation["context_docs"] <= cfg.retrieval["max_retrieved"]:
            raise ValueError("context_docs không được lớn hơn max_retrieved.")
        if not 1 <= cfg.generation["max_answer_chars"] <= 2000:
            raise ValueError("max_answer_chars phải nằm trong 1–2000.")
        for group, keys in (
            (cfg.models, ("batch_size", "cpu_threads")),
            (cfg.retrieval, ("dense_k", "bm25_k", "rerank_k", "rrf_constant")),
            (
                cfg.generation,
                ("context_token_budget", "max_tokens", "request_timeout", "max_attempts"),
            ),
        ):
            if any(group[k] <= 0 for k in keys):
                raise ValueError(f"Các giá trị {keys} phải lớn hơn 0.")
        if cfg.generation["request_interval"] < 0.6:
            raise ValueError("request_interval cần >= 0.6 giây để không vượt 100 RPM.")
        return cfg
