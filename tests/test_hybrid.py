"""Kiểm tra hybrid offline với Qdrant in-memory, không tải model/gọi gateway."""

# ruff: noqa: E402
# Stack hybrid là optional; starter kit cơ bản vẫn chạy được test riêng.
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from openai import APIConnectionError

for optional in (
    "qdrant_client",
    "langchain_core",
    "langchain_text_splitters",
    "rank_bm25",
    "tenacity",
):
    pytest.importorskip(optional)

from qdrant_client import QdrantClient

from conftest import FakeEmbedder
from neorag.hybrid import api
from neorag.hybrid.generation import ABSTENTION, build_context, parse_answer
from neorag.hybrid.ingestion import Chunk, ingest
from neorag.hybrid.retrieval import HybridRetriever, rrf
from neorag.hybrid.settings import Settings
from neorag.pipelines.hybrid import HybridPipeline


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text)


class Models:
    tokenizer = Tokenizer()
    fingerprint = "offline-test-v1"

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return FakeEmbedder().embed(texts)

    def embed_query(self, query):
        return FakeEmbedder().embed([query])[0]

    def rerank(self, query, texts):
        return np.array([1.0 if "09/10/2011" in text else 0.0 for text in texts])


@pytest.fixture
def cfg():
    return Settings.load()


def test_ingestion_keeps_parent_and_short_documents(corpus):
    chunks = ingest(corpus, Tokenizer(), chunk_size=60, chunk_overlap=10)
    assert {c.parent_id for c in chunks} == {d.id for d in corpus}
    assert all(c.club and c.topic and c.id for c in chunks)
    assert chunks == ingest(corpus, Tokenizer(), chunk_size=60, chunk_overlap=10)


def test_ingestion_splits_long_document_and_preserves_ids(corpus):
    from neorag.data import Document

    docs = [Document(99, "CLB", "Chủ đề", "Thông tin. " * 80)]
    chunks = ingest(docs, Tokenizer(), chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.parent_id == 99 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_rrf_rewards_agreement_and_does_not_double_count_duplicates():
    assert rrf([[1, 1, 2], [2, 3]]) == [2, 1, 3]


def test_qdrant_reuse_rerank_and_parent_dedup(corpus, cfg):
    models = Models()
    chunks = ingest(corpus, models.tokenizer)
    duplicate = Chunk("00000000-0000-0000-0000-000000000001", 1, "09/10/2011.", "CLB", "Ngày")
    client = QdrantClient(":memory:")
    retriever = HybridRetriever([*chunks, duplicate], models, models, client, cfg.retrieval)
    retriever.index()
    retriever.index()
    assert models.calls == 1
    hits = retriever.search("ProPTIT thành lập ngày nào?")
    assert hits[0].parent_id == 1
    assert len({h.parent_id for h in hits}) == len(hits)
    changed = HybridRetriever(chunks, models, models, client, cfg.retrieval)
    assert changed.collection != retriever.collection
    client.close()


def test_pipeline_adapter_with_real_inmemory_qdrant(corpus):
    gateway = SimpleNamespace(
        chat=lambda messages: json.dumps(
            {
                "answer": "Ngày 09/10/2011.",
                "citations": [1],
                "insufficient": False,
            }
        )
    )
    qdrant = QdrantClient(":memory:")
    pipeline = HybridPipeline(
        "configs/initial.toml", models=Models(), client=gateway, qdrant=qdrant
    )
    pipeline.index(corpus)
    out = pipeline.run("ProPTIT thành lập khi nào?")
    assert out.answer == "Ngày 09/10/2011. [1]"
    assert out.retrieved_ids[0] == 1
    with pytest.raises(ValueError, match="128"):
        pipeline.retrieve("a" * 129)
    qdrant.close()


def test_context_budget_and_citation_validation(cfg):
    docs = [Chunk("a", 1, "Thông tin.", "CLB", "Topic"), Chunk("b", 2, "x" * 1000, "CLB", "Topic")]
    generation = {**cfg.generation, "context_token_budget": 100}
    context, ids = build_context(docs, generation)
    assert len(context.encode()) <= 100
    assert ids == [1]
    raw = json.dumps({"answer": "Có đáp án", "citations": [2], "insufficient": False})
    with pytest.raises(ValueError, match="Citation"):
        parse_answer(raw, ids)
    with pytest.raises(ValueError, match="JSON"):
        parse_answer("not json", ids)
    with pytest.raises(ValueError, match="budget"):
        build_context(docs, {**generation, "context_token_budget": 1})


def test_abstention_is_explicit_and_not_from_transport_errors():
    raw = json.dumps({"answer": "", "citations": [], "insufficient": True})
    assert parse_answer(raw, []) == ABSTENTION
    raw = json.dumps({"answer": "x" * 2000, "citations": [1], "insufficient": False})
    with pytest.raises(ValueError, match="ký tự"):
        parse_answer(raw, [1])


def test_gateway_retry_cache_usage_and_output_limit(tmp_path, cfg, monkeypatch):
    attempts, params = [], []

    def create(**kwargs):
        attempts.append(1)
        params.append(kwargs)
        if len(attempts) == 1:
            raise APIConnectionError(request=None)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="result"))
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )

    sdk = SimpleNamespace(
        base_url="http://offline.test/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(api, "retry_wait", lambda state: 0)
    monkeypatch.setattr(api._LIMITER, "acquire", lambda *args: None)
    client = api.GatewayClient(
        {**cfg.generation, "cache_path": str(tmp_path / "cache.sqlite")}, sdk=sdk
    )
    messages = [{"role": "user", "content": "test"}]
    assert client.chat(messages) == client.chat(messages) == "result"
    assert len(attempts) == 2
    assert params[-1]["max_tokens"] == cfg.generation["max_tokens"]
    assert "temperature" not in params[-1]
    assert client.cache._conn.execute("SELECT SUM(input_tokens) FROM usage").fetchone()[0] == 10


def test_gateway_does_not_retry_authentication_or_budget():
    from openai import APIStatusError

    for status in [400, 401, 403, 429, 503]:
        response = SimpleNamespace(status_code=status, request=None, headers={})
        exc = APIStatusError("test", response=response, body=None)
        assert api.is_retryable(exc) is (status in [429, 503])


def test_invalid_overlap_rejected_before_loading_models(tmp_path):
    from neorag.config import CONFIG_DIR

    text = (
        (CONFIG_DIR / "hybrid.toml")
        .read_text()
        .replace("chunk_overlap = 40", "chunk_overlap = 300")
    )
    path = tmp_path / "invalid.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings.load(path)


def test_retry_after_is_honored_and_excessive_wait_fails():
    from openai import APIStatusError

    response = SimpleNamespace(status_code=429, request=None, headers={"retry-after": "30"})
    exc = APIStatusError("rate limited", response=response, body=None)
    state = SimpleNamespace(attempt_number=1, outcome=SimpleNamespace(exception=lambda: exc))
    assert api.retry_wait(state) >= 30
    response.headers["retry-after"] = "120"
    with pytest.raises(RuntimeError, match="60"):
        api.retry_wait(state)


def test_truncated_answer_does_not_enter_cache(tmp_path, cfg, monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="partial"))
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=768),
    )
    sdk = SimpleNamespace(
        base_url="http://offline.test/v1",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response),
        ),
    )
    monkeypatch.setattr(api._LIMITER, "acquire", lambda *args: None)
    client = api.GatewayClient(
        {**cfg.generation, "cache_path": str(tmp_path / "cache.sqlite")}, sdk=sdk
    )
    with pytest.raises(ValueError, match="max_tokens"):
        client.chat([{"role": "user", "content": "test"}])
    assert client.cache._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0


def test_invalid_generation_is_evicted_instead_of_becoming_abstention(cfg):
    from neorag.hybrid.generation import Generator

    invalidated = []
    client = SimpleNamespace(chat=lambda messages: "bad JSON", invalidate_chat=invalidated.append)
    with pytest.raises(ValueError, match="JSON"):
        Generator(client, cfg.generation).answer(
            "Câu hỏi?", [Chunk("x", 1, "Thông tin.", "CLB", "Topic")]
        )
    assert len(invalidated) == 2


def test_generator_repairs_invalid_citation_once(cfg):
    from neorag.hybrid.generation import Generator

    replies = iter(
        [
            json.dumps({"answer": "Thông tin", "citations": [999], "insufficient": False}),
            json.dumps({"answer": "Thông tin", "citations": ["1"], "insufficient": False}),
        ]
    )
    client = SimpleNamespace(
        chat=lambda messages: next(replies), invalidate_chat=lambda messages: None
    )
    answer = Generator(client, cfg.generation).answer(
        "Câu hỏi?", [Chunk("x", 1, "Thông tin", "CLB", "Topic")]
    )
    assert answer == "Thông tin [1]"


def test_query_normalization_uses_corpus_not_question_labels():
    from neorag.hybrid.query import QueryNormalizer

    normalizer = QueryNormalizer(["Ban chủ nhiệm câu lạc bộ lập trình được thành lập."])
    restored = normalizer.normalize("ban chu nhiem clb lap trinh dc thanh lap?")
    assert "ban chủ nhiệm" in restored
    assert "lập trình" in restored
    assert "được" in restored
    assert "2025" in normalizer.normalize("nam 2025")


def test_context_budget_increase_keeps_more_evidence(cfg):
    docs = [Chunk(str(i), i, "thông tin " * 80, "CLB", "topic") for i in range(1, 7)]
    _, small = build_context(docs, {**cfg.generation, "context_token_budget": 3000})
    _, large = build_context(docs, {**cfg.generation, "context_token_budget": 9000})
    assert len(large) > len(small)
    assert large[: len(small)] == small


def test_generation_quota_allows_cache_but_blocks_new_calls(tmp_path, cfg, monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="ok"))],
        usage=None,
    )
    sdk = SimpleNamespace(
        base_url="http://offline.test",
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)),
    )
    monkeypatch.setattr(api._LIMITER, "acquire", lambda *args: None)
    client = api.GatewayClient(
        {
            **cfg.generation,
            "cache_path": str(tmp_path / "quota.sqlite"),
            "budget_start": "2000-01-01",
            "max_live_requests": 1,
        },
        sdk=sdk,
    )
    messages = [{"role": "user", "content": "one"}]
    assert client.chat(messages) == "ok"
    assert client.chat(messages) == "ok"
    with pytest.raises(RuntimeError, match="quota"):
        client.chat([{"role": "user", "content": "two"}])


def test_initial_profile_remains_reproducible(corpus):
    from neorag.config import CONFIG_DIR

    class CountingModels(Models):
        def __init__(self):
            super().__init__()
            self.reranks = 0

        def rerank(self, query, texts):
            self.reranks += 1
            return super().rerank(query, texts)

    model = CountingModels()
    client = QdrantClient(":memory:")
    pipeline = HybridPipeline(CONFIG_DIR / "initial.toml", models=model, qdrant=client)
    pipeline.index(corpus)
    pipeline.retrieve("ProPTIT thành lập khi nào?")
    assert model.reranks == 1
    client.close()


def test_query_normalizer_handles_decomposed_unicode():
    import unicodedata

    from neorag.hybrid.query import QueryNormalizer

    normalizer = QueryNormalizer(["Ban chủ nhiệm câu lạc bộ lập trình."])
    query = "Ban chủ nhiệm clb lập trình?"
    assert normalizer.normalize(query) == normalizer.normalize(unicodedata.normalize("NFD", query))


def test_train_stress_augmentation_preserves_numbers_names_and_negation():
    from scripts.benchmark_accuracy import perturb

    query = "CLB Multimedia không tuyển thành viên năm 2026?"
    changed = perturb(query, "generic-example")
    assert "Multimedia" in changed
    assert "không" in changed
    assert "2026" in changed
    assert perturb(query, "generic-example") == changed


def test_benchmark_parent_ranking_is_unique():
    from scripts.benchmark_accuracy import parent_rank

    assert parent_rank([2, 0, 1, 3], [10, 10, 20, 30]) == [20, 10, 30]


def test_explanatory_abstention_preserves_cited_correction():
    raw = json.dumps(
        {
            "answer": "Chưa có lịch năm mới; tài liệu chỉ nêu đợt cũ.",
            "citations": [1],
            "insufficient": True,
        }
    )
    assert parse_answer(raw, [1]) == ABSTENTION
    assert parse_answer(raw, [1], explanatory_abstention=True).endswith("[1]")
    wrong = json.dumps(
        {"answer": "Năm mới bắt đầu ngày 1/1.", "citations": [1], "insufficient": True}
    )
    with pytest.raises(ValueError, match="giới hạn"):
        parse_answer(wrong, [1], explanatory_abstention=True)


def test_diverse_context_covers_explicit_clubs_without_changing_legacy(cfg):
    docs = [Chunk(str(i), i, "Thông tin.", "CLB Lập Trình PTIT", "Topic") for i in range(1, 5)]
    docs.append(Chunk("5", 5, "Thông tin GCC.", "CLB Nhà Sáng Tạo Game PTIT (GCC)", "Topic"))
    settings = {**cfg.generation, "context_docs": 2, "diversify_context": False}
    _, legacy = build_context(docs, settings, "So sánh ProPTIT với GCC?")
    _, diverse = build_context(
        docs, {**settings, "diversify_context": True}, "So sánh ProPTIT với GCC?"
    )
    assert legacy == [1, 2]
    assert diverse == [1, 5]


def test_embedding_item_cache_reuses_across_batch_boundaries(tmp_path, cfg, monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs["input"])
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=[float(len(t)), 1.0])
                for i, t in enumerate(kwargs["input"])
            ],
            usage=None,
        )

    sdk = SimpleNamespace(base_url="http://offline.test", embeddings=SimpleNamespace(create=create))
    monkeypatch.setattr(api._LIMITER, "acquire", lambda *args: None)
    client = api.GatewayClient(
        {**cfg.generation, "cache_path": str(tmp_path / "embed.sqlite")}, sdk=sdk
    )
    vectors = client.embed(["abc", "hello", "abc"])
    assert calls == [["abc", "hello"]]
    assert vectors[0] == vectors[2]
    assert client.embed(["hello", "abc"]) == [vectors[1], vectors[0]]
    assert len(calls) == 1


def test_all_candidate_pool_and_batched_query_embeddings(corpus, cfg):
    class BatchedModels(Models):
        def __init__(self):
            super().__init__()
            self.batches = 0
            self.seen = 0

        def embed_queries(self, texts):
            self.batches += 1
            return FakeEmbedder().embed(texts)

        def embed_query(self, query):
            raise AssertionError("must batch queries")

        def rerank(self, query, texts):
            self.seen = len(texts)
            return super().rerank(query, texts)

    model = BatchedModels()
    client = QdrantClient(":memory:")
    retriever = HybridRetriever(
        ingest(corpus, model.tokenizer),
        model,
        model,
        client,
        {
            **cfg.retrieval,
            "batch_queries": True,
            "rerank_pool": "all",
            "rerank_mode": "raw",
            "rerank_weight": 1.0,
            "dense_k": 1,
            "bm25_k": 1,
        },
    )
    retriever.index()
    hits = retriever.search("Multimedia thành lập khi nào?")
    assert model.batches == 1
    assert model.seen == 3
    assert hits[0].parent_id == 1
    client.close()


def test_old_embedding_batch_cache_migrates_without_network(tmp_path, cfg):
    def fail(**kwargs):
        raise AssertionError("cached embeddings must not call gateway")

    sdk = SimpleNamespace(base_url="http://offline.test", embeddings=SimpleNamespace(create=fail))
    client = api.GatewayClient(
        {**cfg.generation, "cache_path": str(tmp_path / "legacy.sqlite")}, sdk=sdk
    )
    client.cache.set(
        api._key("hybrid-embed-v1", client.endpoint, "bge-m3", ["abc", "x"]),
        json.dumps([[3.0, 1.0], [1.0, 1.0]]),
    )
    assert client.embed(["abc", "x"]) == [[3.0, 1.0], [1.0, 1.0]]
    assert client.embed(["x"]) == [[1.0, 1.0]]


def test_gateway_profile_pipeline_adapter_without_network(corpus):
    class FakeGateway:
        endpoint = "http://offline.test/v1"

        def embed(self, texts):
            return FakeEmbedder().embed(texts).tolist()

        def chat(self, messages):
            return json.dumps(
                {"answer": "Ngày 09/10/2011.", "citations": [1], "insufficient": False}
            )

    client = QdrantClient(":memory:")
    pipeline = HybridPipeline(
        "configs/accuracy_selected.toml", models=Models(), client=FakeGateway(), qdrant=client
    )
    pipeline.index(corpus)
    result = pipeline.run("ProPTIT thành lập khi nào?")
    assert result.retrieved_ids[0] == 1
    assert result.answer.endswith("[1]")
    client.close()


def test_query_token_windows_cover_all_input_without_oversized_pairs():
    from neorag.hybrid.models import token_windows

    ids = list(range(511))
    parts = token_windows(ids)
    assert all(len(part) <= 128 for part in parts)
    assert {i for part in parts for i in part} == set(ids)
    assert parts[0][0] == 0 and parts[-1][-1] == 510
    assert token_windows([1, 2, 3]) == [[1, 2, 3]]


def test_minmax_fusion_is_affine_invariant_and_handles_flat_scores(corpus, cfg):
    class ScaledModels(Models):
        scale = 1.0
        offset = 0.0

        def rerank(self, query, texts):
            return self.scale * super().rerank(query, texts) + self.offset

    model = ScaledModels()
    client = QdrantClient(":memory:")
    retriever = HybridRetriever(
        ingest(corpus, model.tokenizer),
        model,
        model,
        client,
        {**cfg.retrieval, "fusion_mode": "minmax", "rerank_weight": 0.5},
    )
    retriever.index()
    original = [d.parent_id for d in retriever.search("Multimedia thành lập khi nào?")]
    model.scale, model.offset = 7.0, -31.0
    assert original == [d.parent_id for d in retriever.search("Multimedia thành lập khi nào?")]
    model.scale = 0.0
    flat = retriever.search("Multimedia thành lập khi nào?")
    assert len(flat) == len(corpus)
    assert len({d.parent_id for d in flat}) == len(corpus)
    client.close()


def test_gateway_models_defer_unused_encoder(monkeypatch, cfg):
    import sys

    import torch

    from neorag.hybrid import models as module

    calls = []

    class Network:
        def to(self, device):
            return self

        def eval(self):
            return self

    fake = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        AutoModelForSequenceClassification=SimpleNamespace(
            from_pretrained=lambda *a, **k: Network()
        ),
        AutoModel=SimpleNamespace(from_pretrained=lambda *a, **k: calls.append("encoder")),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(module, "download_model", lambda *a: "/tmp/model")
    previous = torch.get_num_threads()
    try:
        models = module.LocalModels(cfg.models)
        assert models.encoder is None
        assert calls == []
        assert models.tokenizer is not None
        assert models.reranker is not None
    finally:
        torch.set_num_threads(previous)


def test_unknown_prompt_profile_is_rejected(tmp_path):
    from neorag.config import CONFIG_DIR

    text = (CONFIG_DIR / "accuracy_v3.toml").read_text()
    profile = tmp_path / "bad_prompt.toml"
    profile.write_text(text.replace('prompt_version = "precise"', 'prompt_version = "typo"'))
    with pytest.raises(ValueError, match="prompt_version"):
        Settings.load(profile)
