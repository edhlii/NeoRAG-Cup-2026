"""Gateway có cache, một tầng retry, rate limiter và nhật ký token sử dụng."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from openai import APIConnectionError, APIStatusError, OpenAI
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from neorag.llm import _Cache, _key


class RateLimiter:
    """Giới hạn dùng chung trong một process; không thay thế limiter của gateway."""

    def __init__(self):
        self.lock = threading.Lock()
        self.history = deque()
        self.last = 0.0

    def acquire(self, tokens, interval):
        if tokens > 250_000:
            raise ValueError("Một request vượt ngân sách token/phút.")
        with self.lock:
            while True:
                now = time.monotonic()
                while self.history and now - self.history[0][0] >= 60:
                    self.history.popleft()
                delay = max(0.0, self.last + interval - now)
                if len(self.history) >= 90 or sum(n for _, n in self.history) + tokens > 250_000:
                    delay = max(delay, self.history[0][0] + 60 - now)
                if delay <= 0:
                    self.last = now
                    self.history.append((now, tokens))
                    return
                time.sleep(delay)


_LIMITER = RateLimiter()


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        # Không thử lại 400 (budget/tham số), 401 hoặc 403.
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return False


def retry_wait(state) -> float:
    exc = state.outcome.exception()
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    delay = wait_random_exponential(multiplier=1, min=1, max=20)(state)
    if header:
        try:
            after = float(header)
        except ValueError:
            try:
                after = (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                after = 0
        # Không gửi sớm hơn Retry-After. Dừng thay vì sleep vô hạn.
        if after > 60:
            raise RuntimeError("Gateway yêu cầu chờ hơn 60 giây; hãy thử lại sau.") from exc
        delay = max(delay, after)
    return delay


class GatewayClient:
    MODEL = "claude-haiku-5-5"

    def __init__(self, cfg, sdk=None):
        import os

        self.cfg = cfg
        self.sdk = sdk or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://proptit.dekthe.dev/v1"),
            max_retries=0,
            timeout=cfg["request_timeout"],
        )
        self.endpoint = str(self.sdk.base_url)
        self.cache = _Cache(cfg["cache_path"])
        self.lock = threading.Lock()
        with self.cache._conn:
            self.cache._conn.execute(
                "CREATE TABLE IF NOT EXISTS usage "
                "(created_at TEXT, model TEXT, input_tokens INTEGER, output_tokens INTEGER)"
            )

    def _call(self, func, estimated_tokens):
        for attempt in Retrying(
            stop=stop_after_attempt(self.cfg["max_attempts"]),
            wait=retry_wait,
            retry=retry_if_exception(is_retryable),
            reraise=True,
        ):
            with attempt:
                if self.cfg.get("budget_start") and self.cfg.get("max_live_requests"):
                    with self.cache._lock:
                        count = self.cache._conn.execute(
                            "SELECT COUNT(*) FROM usage WHERE created_at > ?",
                            (self.cfg["budget_start"],),
                        ).fetchone()[0]
                    if count >= self.cfg["max_live_requests"]:
                        raise RuntimeError(
                            "Đã chạm quota request thử nghiệm; cache vẫn dùng được, cần xem lại ngân sách trước khi gọi thêm."
                        )
                _LIMITER.acquire(estimated_tokens, self.cfg["request_interval"])
                return func()
        raise RuntimeError("Không nhận được phản hồi gateway.")

    def _usage(self, response, model):
        usage = getattr(response, "usage", None)
        with self.cache._lock, self.cache._conn:
            self.cache._conn.execute(
                "INSERT INTO usage VALUES (?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    model,
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0),
                ),
            )

    def chat(self, messages: list[dict]) -> str:
        key = _key("hybrid-chat-v1", self.endpoint, self.MODEL, self.cfg["max_tokens"], messages)
        with self.lock:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
            # Proxy byte UTF-8 để lập ngân sách bảo thủ; không phải tokenizer Claude.
            reserve = (
                len(json.dumps(messages, ensure_ascii=False).encode()) + self.cfg["max_tokens"]
            )
            response = self._call(
                lambda: self.sdk.chat.completions.create(
                    model=self.MODEL,
                    messages=messages,
                    max_tokens=self.cfg["max_tokens"],
                ),
                reserve,
            )
            self._usage(response, self.MODEL)
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise ValueError(
                    "LLM bị cắt vì max_tokens; tăng giới hạn có kiểm soát rồi chạy lại."
                )
            answer = (choice.message.content or "").strip()
            if not answer:
                raise ValueError("Gateway trả câu trả lời rỗng.")
            self.cache.set(key, answer)
            return answer

    def invalidate_chat(self, messages: list[dict]) -> None:
        key = _key("hybrid-chat-v1", self.endpoint, self.MODEL, self.cfg["max_tokens"], messages)
        with self.lock, self.cache._lock, self.cache._conn:
            self.cache._conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Cache từng text, giữ tương thích cache batch cũ và gộp các cache miss."""
        vectors = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            keys = [_key("hybrid-embed-item-v1", self.endpoint, "bge-m3", text) for text in batch]
            batch_key = _key("hybrid-embed-v1", self.endpoint, "bge-m3", batch)
            with self.lock:
                old = self.cache.get(batch_key)
                if old is not None:
                    result = json.loads(old)
                    if len(result) != len(batch):
                        raise ValueError("Cache embedding batch không khớp số đầu vào.")
                else:
                    hits = {
                        text: self.cache.get(key) for text, key in zip(batch, keys, strict=True)
                    }
                    missing = list(dict.fromkeys(text for text in batch if hits[text] is None))
                    fresh = {}
                    if missing:
                        response = self._call(
                            lambda missing=missing: self.sdk.embeddings.create(
                                model="bge-m3",
                                input=missing,
                                encoding_format="float",
                            ),
                            sum(len(t.encode()) for t in missing),
                        )
                        self._usage(response, "bge-m3")
                        ordered = sorted(response.data, key=lambda item: item.index)
                        if [item.index for item in ordered] != list(range(len(missing))):
                            raise ValueError("Gateway trả số/index embedding không khớp request.")
                        fresh = {
                            text: item.embedding
                            for text, item in zip(missing, ordered, strict=True)
                        }
                    result = [
                        json.loads(hits[text]) if hits[text] is not None else fresh[text]
                        for text in batch
                    ]
                entries = [
                    (key, json.dumps(vector)) for key, vector in zip(keys, result, strict=True)
                ]
                with self.cache._lock, self.cache._conn:
                    self.cache._conn.executemany(
                        "INSERT OR REPLACE INTO cache VALUES (?, ?)", entries
                    )
                vectors.extend(result)
        return vectors
