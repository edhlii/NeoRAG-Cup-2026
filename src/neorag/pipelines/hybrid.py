"""Adapter pipeline hybrid cho runner/chabot có sẵn, không sửa bộ chấm."""

from __future__ import annotations

import threading

from neorag.data import Document
from neorag.hybrid.settings import Settings
from neorag.pipelines.base import RAGOutput, RAGPipeline


class HybridPipeline(RAGPipeline):
    def __init__(self, config_path=None, *, models=None, client=None, qdrant=None):
        self.cfg = Settings.load(config_path)
        self.models, self.client, self.qdrant = models, client, qdrant
        self.lock = threading.Lock()
        self.retriever = None

    def prepare(self) -> None:
        if self.models is None:
            from neorag.hybrid.models import LocalModels

            self.models = LocalModels(self.cfg.models)
        if self.qdrant is None:
            from qdrant_client import QdrantClient

            self.qdrant = QdrantClient(url=self.cfg.qdrant["url"], timeout=10)

    def index(self, corpus: list[Document]) -> None:
        from neorag.hybrid.ingestion import ingest
        from neorag.hybrid.retrieval import HybridRetriever

        self.prepare()
        embedder = self.models
        if self.cfg.models["embedding_backend"] == "gateway":
            from neorag.hybrid.models import GatewayEmbedding

            embedder = GatewayEmbedding(self.models, self._gateway())
        chunks = ingest(corpus, self.models.tokenizer, **self.cfg.ingestion)
        self.retriever = HybridRetriever(
            chunks,
            embedder,
            self.models,
            self.qdrant,
            self.cfg.retrieval,
            self.cfg.qdrant["collection_prefix"],
        )
        self.retriever.index()

    def _gateway(self):
        if self.client is None:
            from neorag.hybrid.api import GatewayClient

            self.client = GatewayClient(self.cfg.generation)
        return self.client

    def retrieve(self, query):
        """Truy xuất/rerank không gọi LLM; dùng đo metric retrieval trên train."""
        if self.retriever is None:
            raise RuntimeError("Cần gọi index(corpus) trước khi truy xuất.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Câu hỏi không được rỗng.")
        limit = self.cfg.retrieval.get("max_query_tokens", 128)
        if len(self.models.tokenizer.encode(query, add_special_tokens=True)) > limit:
            raise ValueError(f"Câu hỏi vượt {limit} token; hãy chia thành các câu hỏi ngắn hơn.")
        return self.retriever.search(query)

    def run(self, query: str) -> RAGOutput:
        from neorag.hybrid.generation import Generator

        with self.lock:
            documents = self.retrieve(query)
            answer = Generator(self._gateway(), self.cfg.generation).answer(query, documents)
            return RAGOutput(answer=answer, retrieved_ids=[d.parent_id for d in documents])
