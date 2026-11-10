"""Model local nhỏ, tải một định dạng trọng số; embedding gateway dùng để đối chứng."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from neorag.hybrid.settings import ROOT

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def download_model(name: str, revision: str = "main") -> str:
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "model.safetensors", "sentencepiece.bpe.model", "1_Pooling/*"]
    try:
        path = snapshot_download(
            name, revision=revision, allow_patterns=patterns, local_files_only=True
        )
        if (Path(path) / "model.safetensors").is_file() and (
            Path(path) / "tokenizer.json"
        ).is_file():
            return path
    except (FileNotFoundError, OSError):
        pass
    return snapshot_download(
        name,
        revision=revision,
        allow_patterns=patterns,
        max_workers=2,
        ignore_patterns=["onnx/*", "openvino/*"],
    )


def token_windows(tokens: list[int], size: int = 128, overlap: int = 32) -> list[list[int]]:
    if not 0 <= overlap < size:
        raise ValueError("Cần 0 <= overlap < size.")
    if len(tokens) <= size:
        return [tokens]
    starts = list(range(0, len(tokens) - size + 1, size - overlap))
    if starts[-1] != len(tokens) - size:
        starts.append(len(tokens) - size)
    return [tokens[start : start + size] for start in starts]


class LocalModels:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(cfg["cpu_threads"])
        embedding_path = download_model(cfg["embedding"], cfg.get("embedding_revision", "main"))
        reranker_path = download_model(cfg["reranker"], cfg.get("reranker_revision", "main"))
        self.device = cfg["device"]
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_path, local_files_only=True)
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_path, local_files_only=True)
        self.embedding_path = embedding_path
        self.encoder = None
        self.reranker = (
            AutoModelForSequenceClassification.from_pretrained(
                reranker_path,
                local_files_only=True,
                use_safetensors=True,
            )
            .to(self.device)
            .eval()
        )
        self.batch_size = cfg["batch_size"]
        self.fingerprint = f"{cfg['embedding']}:{Path(embedding_path).name}:e5-mean-pool-v1"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([f"passage: {text}" for text in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_queries([query])[0]

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([f"query: {text}" for text in texts])

    def _embed(self, texts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        if self.encoder is None:
            from transformers import AutoModel

            self.encoder = (
                AutoModel.from_pretrained(
                    self.embedding_path, local_files_only=True, use_safetensors=True
                )
                .to(self.device)
                .eval()
            )
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[start : start + self.batch_size],
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.encoder(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors.append(functional.normalize(pooled, p=2, dim=1).cpu().numpy())
        return np.concatenate(vectors).astype(np.float32)

    def rerank(self, query: str, texts: list[str]) -> np.ndarray:
        import torch

        ids = self.rerank_tokenizer.encode(query, add_special_tokens=False)
        views = (
            [query]
            if len(ids) <= 128
            else [
                self.rerank_tokenizer.decode(window, skip_special_tokens=True)
                for window in token_windows(ids)
            ]
        )
        by_view = []
        with torch.inference_mode():
            for view in views:
                scores = []
                for start in range(0, len(texts), self.batch_size):
                    batch = texts[start : start + self.batch_size]
                    pairs = self.rerank_tokenizer(
                        [view] * len(batch),
                        batch,
                        padding=True,
                        truncation="only_second",
                        max_length=512,
                        return_tensors="pt",
                    ).to(self.device)
                    logits = self.reranker(**pairs).logits
                    if logits.shape[-1] != 1:
                        raise ValueError("Reranker phải có một logit relevance cho mỗi cặp.")
                    scores.append(logits[:, 0].cpu().numpy())
                by_view.append(np.concatenate(scores))
        return np.max(np.stack(by_view), axis=0)


class GatewayEmbedding:
    """Giữ tokenizer/reranker local, chỉ thay nhánh dense bằng bge-m3 gateway."""

    def __init__(self, models, client):
        self.models = models
        self.client = client
        self.fingerprint = f"gateway:bge-m3:{client.endpoint}"

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, query):
        return self.embed_queries([query])[0]

    def embed_queries(self, texts):
        return self._embed(texts)

    def _embed(self, texts):
        vectors = np.asarray(self.client.embed(texts), dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
